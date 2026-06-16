# from sentence_transformers import CrossEncoder
# import os

# MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# MODEL_DIR = "models/cross-encoder-ms-marco-MiniLM-L-6-v2"

# # Download model from Hugging Face
# reranker = CrossEncoder(MODEL_NAME)

# # Save locally
# os.makedirs(MODEL_DIR, exist_ok=True)
# reranker.save_pretrained(MODEL_DIR)

# print(f"Model saved to: {MODEL_DIR}")

from sentence_transformers import CrossEncoder

MODEL_DIR = "models/cross-encoder-ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(
    MODEL_DIR,
    local_files_only=True
)

print("Model loaded successfully!")

