# scripts/embed_video.py
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# choose device if available
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

def frames_to_pil(frames):
    """
    frames: numpy array (N, H, W, 3) dtype uint8 usually from decord
    returns: list of PIL Images
    """
    return [Image.fromarray(f) for f in frames]

def embed_video(frames, batch_size=8):
    """
    frames: numpy array (N, H, W, 3)
    returns: 1D numpy embedding vector (float32)
    """
    # Ensure dtype uint8
    frames = frames.astype("uint8")
    pil_frames = frames_to_pil(frames)

    all_feats = []
    model.eval()
    with torch.no_grad():
        # process in small batches to avoid OOM
        for i in range(0, len(pil_frames), batch_size):
            batch = pil_frames[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            image_feats = model.get_image_features(**inputs)  # (B, dim)
            # move to cpu and to numpy
            image_feats = image_feats.cpu().numpy()
            all_feats.append(image_feats)

    if len(all_feats) == 0:
        raise RuntimeError("No frames processed (empty frame list).")

    all_feats = np.vstack(all_feats).astype("float32")  # (N, dim)

    # Normalize per-vector then mean-pool (cosine-friendly)
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_feats = all_feats / norms

    video_emb = all_feats.mean(axis=0)
    # final normalize
    vnorm = np.linalg.norm(video_emb)
    if vnorm > 0:
        video_emb = video_emb / vnorm

    return video_emb.astype("float32")
