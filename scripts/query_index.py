import numpy as np
import faiss
from embed_video import embed_video
from extract_frames import sample_frames

INDEX_FILE = "faiss_video_index.bin"
META_FILE = "video_meta.txt"

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r") as f:
        meta = [line.strip() for line in f.readlines()]
    return index, meta

def query(video_path, top_k=5):
    frames = sample_frames(video_path)
    q_emb = embed_video(frames)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb.reshape(1, -1))

    index, meta = load_index()
    D, I = index.search(q_emb.reshape(1, -1), top_k)  # distances and indices

    results = [(meta[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return results

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    results = query(video_path)
    print("Top similar videos:")
    for filename, score in results:
        print(f"{filename} | similarity: {score:.4f}")
