import os
import numpy as np
import faiss
from embed_video import embed_video
from extract_frames import sample_frames

VIDEOS_DIR = "videos"
EMBEDDINGS_DIR = "embeddings"
INDEX_FILE = "faiss_video_index.bin"
META_FILE = "video_meta.txt"

def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".avi", ".mkv"))]
    embeddings = []
    meta = []

    for i, vf in enumerate(video_files):
        path = os.path.join(VIDEOS_DIR, vf)
        frames = sample_frames(path)
        emb = embed_video(frames)
        embeddings.append(emb)
        meta.append(vf)
        np.save(os.path.join(EMBEDDINGS_DIR, vf + ".npy"), emb)
        print(f"[{i+1}/{len(video_files)}] Processed {vf}")

    embeddings = np.array(embeddings).astype('float32')
    d = embeddings.shape[1]  # dimension, should be 512
    index = faiss.IndexFlatIP(d)  # cosine similarity using inner product (assumes normalized vectors)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    # Save video filenames in order
    with open(META_FILE, "w") as f:
        for m in meta:
            f.write(m + "\n")

    print(f"Built FAISS index with {len(video_files)} videos.")

if __name__ == "__main__":
    main()
