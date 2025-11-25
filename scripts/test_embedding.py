import sys
import os
import numpy as np
from embed_video import embed_video
from extract_frames import sample_frames

def test_embedding():
    video_path = "../videos/HP.mp4"
    if not os.path.exists(video_path):
        # Try to find any video
        video_dir = "../videos"
        videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        if videos:
            video_path = os.path.join(video_dir, videos[0])
        else:
            print("No videos found.")
            return

    print(f"Testing embedding with video: {video_path}")
    
    # Sample frames (using our adaptive sampling)
    frames = sample_frames(video_path, num_frames=16, adaptive=True)
    print(f"Sampled frames shape: {frames.shape}")
    
    # Embed video
    try:
        emb = embed_video(frames)
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
        
        if emb.shape == (512,):
            print("SUCCESS: Embedding has correct shape (512,).")
        else:
            print(f"FAILURE: Unexpected embedding shape {emb.shape}")
            
    except Exception as e:
        print(f"FAILURE: Error during embedding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embedding()
