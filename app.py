from flask import Flask, request, jsonify
from embed_video import embed_video
from store_vector import store_vector_s3
from query_vectors import query_similar_vectors
import json

app = Flask(__name__)

VECTOR_BUCKET = "your-vector-bucket"
INDEX_NAME = "video-index"


@app.route("/index", methods=["POST"])
def index_video():
    video_path = request.json.get("video_path")
    video_id = request.json.get("video_id")
    if not video_path or not video_id:
        return jsonify({"error": "Missing video_path or video_id"}), 400

    emb = embed_video(video_path)
    store_vector_s3(VECTOR_BUCKET, INDEX_NAME, video_id, emb)
    return jsonify({"message": f"Video {video_id} indexed successfully."})


@app.route("/query", methods=["POST"])
def query_video():
    video_path = request.json.get("video_path")
    if not video_path:
        return jsonify({"error": "Missing video_path"}), 400

    emb = embed_video(video_path)
    results = query_similar_vectors(VECTOR_BUCKET, INDEX_NAME, emb, top_k=5)
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
