import numpy as np
from decord import VideoReader, cpu

def compute_frame_features(frames):
    """
    Compute simple features for frames to measure visual diversity.
    We'll use downsampled grayscale images as features.
    frames: (N, H, W, 3)
    returns: (N, feature_dim)
    """
    # Downsample to 32x32 to reduce noise and dimensionality
    # We can use simple slicing for speed: frames[:, ::step, ::step, :]
    # Let's just take the mean of channels to get grayscale
    N, H, W, C = frames.shape
    
    # Simple downsampling by slicing (approx 32x32)
    h_step = max(1, H // 32)
    w_step = max(1, W // 32)
    
    small_frames = frames[:, ::h_step, ::w_step, :] # (N, h, w, 3)
    
    # Convert to grayscale (mean over channels) and flatten
    features = small_frames.mean(axis=-1).reshape(N, -1) # (N, h*w)
    
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms
    
    return features

def sample_frames(video_path, num_frames=16, adaptive=True, oversample_factor=4):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return np.zeros((0, 224, 224, 3), dtype='uint8')
        
    if not adaptive or total <= num_frames:
        # Uniform sampling (fallback or requested)
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        frames = vr.get_batch(idxs).asnumpy()
    else:
        # Adaptive sampling using Farthest Point Sampling (FPS)
        # 1. Sample a larger pool of candidate frames uniformly
        pool_size = min(total, num_frames * oversample_factor)
        candidate_idxs = np.linspace(0, total - 1, pool_size).astype(int)
        candidate_frames = vr.get_batch(candidate_idxs).asnumpy()
        
        # 2. Compute features
        features = compute_frame_features(candidate_frames) # (pool_size, dim)
        
        # 3. Farthest Point Sampling
        selected_indices_in_pool = []
        
        # Start with the first frame (usually good to anchor start)
        selected_indices_in_pool.append(0)
        
        # Initialize distances to the closest selected point
        # (Start with distance to the first point)
        # dist(x, y) = ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        # Since normalized, ||x||=1, so dist = 2 - 2<x, y>. Max dist <=> Min dot product.
        # We want to maximize distance, so minimize dot product.
        
        # Current distances to the set of selected points (initially just point 0)
        # We want to find point i that maximizes min_dist(i, selected)
        
        # Using dot product as similarity. Distance ~ 1 - similarity.
        # Maximize (1 - similarity) <=> Minimize similarity.
        
        current_sims = np.dot(features, features[0]) # (pool_size,)
        min_sims = current_sims # Track the max similarity to ANY selected point (we want to minimize this max similarity)
        
        for _ in range(num_frames - 1):
            # Find the point that has the LOWEST 'max similarity to any selected point'
            # i.e. it is farthest from the set of selected points
            
            # Note: For standard FPS with Euclidean distance, we maximize the minimum distance.
            # Here with cosine sim, 'distance' is (1 - sim).
            # So we want to maximize (1 - max_sim_to_selected).
            # This is equivalent to minimizing max_sim_to_selected.
            
            # However, we need to update the 'max_sim_to_selected' for all points.
            # Let 'min_sims' be the array where min_sims[i] = max(sim(i, s) for s in selected)
            # We want to choose next p = argmin(min_sims)
            
            next_idx = np.argmin(min_sims)
            selected_indices_in_pool.append(next_idx)
            
            # Update min_sims
            new_sims = np.dot(features, features[next_idx])
            min_sims = np.maximum(min_sims, new_sims)
            
        # 4. Retrieve original indices and sort
        selected_indices_in_pool = sorted(selected_indices_in_pool)
        final_idxs = candidate_idxs[selected_indices_in_pool]
        
        # We already have the frames in memory, just pick them
        frames = candidate_frames[selected_indices_in_pool]

    # if decord returns float, force uint8:
    if frames.dtype != np.uint8:
        frames = frames.astype('uint8')
    return frames
