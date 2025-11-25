# scripts/embed_video.py
import torch
import numpy as np
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEModel
from transformers import CLIPVisionModel, CLIPImageProcessor

# choose device if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading VideoMAE model on {device}...")
videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(device)
videomae_model.eval()

print(f"Loading CLIP model on {device}...")
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

def frames_to_pil(frames):
    """
    frames: numpy array (N, H, W, 3) dtype uint8 usually from decord
    returns: list of PIL Images
    """
    return [Image.fromarray(f) for f in frames]

def embed_video(frames, num_frames_to_use=16):
    """
    frames: numpy array (N, H, W, 3)
    returns: 1D numpy embedding vector (float32)
    
    Combines VideoMAE (action/content) + CLIP (aesthetics/style) embeddings.
    - VideoMAE: 768-dim (temporal understanding, actions)
    - CLIP: 512-dim (visual style, aesthetics)
    - Combined: 1280-dim normalized vector
    """
    # Ensure dtype uint8
    frames = frames.astype("uint8")
    
    if len(frames) == 0:
        raise ValueError("No frames provided to embed_video. Video might be empty or unreadable.")

    pil_frames = frames_to_pil(frames)
    
    # Sample frames if we have more than needed
    if len(pil_frames) > num_frames_to_use:
        indices = np.linspace(0, len(pil_frames) - 1, num_frames_to_use).astype(int)
        pil_frames = [pil_frames[i] for i in indices]
    elif len(pil_frames) < num_frames_to_use:
        # Pad by duplicating the last frame
        while len(pil_frames) < num_frames_to_use:
            pil_frames.append(pil_frames[-1])

    with torch.no_grad():
        # ============ VideoMAE Embedding ============
        videomae_inputs = videomae_processor(pil_frames, return_tensors="pt").to(device)
        videomae_outputs = videomae_model(**videomae_inputs)
        
        # Mean pool across patches for video-level embedding
        videomae_emb = videomae_outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()  # (768,)
        
        # Normalize VideoMAE embedding
        videomae_norm = np.linalg.norm(videomae_emb)
        if videomae_norm > 0:
            videomae_emb = videomae_emb / videomae_norm
        
        # ============ CLIP Embedding ============
        clip_inputs = clip_processor(images=pil_frames, return_tensors="pt").to(device)
        clip_outputs = clip_model(**clip_inputs)
        
        # Average pooling across frames for video-level embedding
        clip_emb = clip_outputs.pooler_output.mean(dim=0).cpu().numpy()  # (512,)
        
        # Normalize CLIP embedding
        clip_norm = np.linalg.norm(clip_emb)
        if clip_norm > 0:
            clip_emb = clip_emb / clip_norm
        
        # ============ Combine Embeddings ============
        # Concatenate normalized embeddings
        combined_emb = np.concatenate([videomae_emb, clip_emb])  # (1280,)
        
        # Normalize the combined embedding for cosine similarity
        combined_norm = np.linalg.norm(combined_emb)
        if combined_norm > 0:
            combined_emb = combined_emb / combined_norm

    return combined_emb.astype("float32")
