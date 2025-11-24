# 🎥 Video Similarity Search Engine

A **local, efficient, and scalable video similarity search** pipeline leveraging frame-level embeddings and vector search - all running **offline and cost-free**! Perfect for content-based video retrieval in diverse collections of varying video lengths.

---

## 🔍 Overview

This project builds a video similarity search system that:

* Extracts **evenly sampled frames** from videos of any length 📹
* Embeds each frame using **OpenAI's CLIP ViT-B/32 model** to capture rich visual semantics 🧠
* Aggregates frame embeddings into a single **video-level embedding vector** 🎯
* Indexes video embeddings using **FAISS**, a fast similarity search library ⚡
* Enables querying by embedding any new video and retrieving the most similar videos by content 🎞️

All processing and search run **locally** - no cloud costs, no external APIs — fully open and customizable.

---

## 🚀 Features

* **Whole-video representation**: Robust similarity that accounts for the entire video content
* **Flexible video lengths**: Works from short clips to hour-long videos seamlessly
* **Fast similarity search**: FAISS index ensures millisecond nearest-neighbor retrieval
* **Open-source tools**: Built with PyTorch, Transformers, Decord, PIL, and FAISS
* **Modular scripts**: Easy to extend or swap components like frame extractor or embedding model

---

## 🛠️ How it works

1. **Frame Extraction**
   Using `decord`, the system samples a fixed number of frames evenly distributed across the video's duration. This ensures a representative snapshot regardless of length.

2. **Frame Embedding**
   Each sampled frame is converted to a PIL image and passed through the **CLIP ViT-B/32 image encoder** to generate a 512-dimensional semantic embedding.

3. **Video Embedding Aggregation**
   Frame embeddings are normalized and averaged to form a single embedding vector representing the entire video.

4. **Indexing**
   All video embeddings are normalized and stored in a **FAISS flat index** for efficient similarity queries based on cosine similarity.

5. **Querying**
   A query video is embedded using the same pipeline and matched against the FAISS index to retrieve the top-k most similar videos ranked by content.

---

## 📁 Project Structure

```
video-similarity-search/
├── videos/                   # Video files for indexing and queries
├── embeddings/               # Stored video embedding vectors (.npy files)
├── scripts/
│   ├── extract_frames.py     # Extracts evenly spaced frames from videos
│   ├── embed_video.py        # Embeds frames and aggregates video embeddings
│   ├── build_index.py        # Builds FAISS index from video embeddings
│   ├── query_index.py        # Queries the FAISS index for similar videos
├── requirements.txt          # Python dependencies
└── README.md                 # This document
```

---

## ⚙️ Setup Instructions

1. **Clone the repo and enter the project folder**

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your videos** to the `videos/` folder (any format like `.mp4`, `.mkv`, `.avi`)

5. **Build the index**

```bash
python scripts/build_index.py
```

6. **Query for similar videos**

```bash
python scripts/query_index.py videos/your_query_video.mp4
```

---

## 📊 Performance Notes

* Frame sampling number can be adjusted in `extract_frames.py` (default 16)
* FAISS index is a flat (exact) search for simplicity; scalable to millions with IVF or HNSW indices
* Embeddings are 512-dimensional vectors from CLIP, effective for visual semantic similarity
* Running on GPU accelerates embedding computation significantly

---

## 🧠 Future Enhancements

* Integrate temporal models for better motion and sequence understanding (e.g., XCLIP, VideoMAE)
* Add audio embedding fusion for multimodal similarity
* Build a web UI or API for interactive similarity search
* Support incremental index updates for dynamic video collections

---

## 🙏 Acknowledgments

* [OpenAI CLIP](https://github.com/openai/CLIP) for powerful image embeddings
* [FAISS](https://github.com/facebookresearch/faiss) for state-of-the-art vector search
* [Decord](https://github.com/dmlc/decord) for efficient video frame decoding
* Huggingface Transformers for seamless model integration

