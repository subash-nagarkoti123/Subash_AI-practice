import json
import numpy as np
import faiss

# -------------------------------
# Configuration
# -------------------------------
EMBEDDINGS_FILE = "output.jsonl"  # Input: embedding-ready JSONL file
INDEX_FILE = "faiss.index"                   # Output: FAISS index file
# -------------------------------

def load_embeddings(jsonl_file):
    """
    Load embeddings from a JSONL file.
    Returns:
        ids: List of chunk IDs
        texts: List of chunk texts
        embeddings: numpy array of embeddings
    """
    ids, texts, embeddings = [], [], []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ids.append(record["id"])
            texts.append(record["text"])
            embeddings.append(record["embedding"])
    embeddings = np.array(embeddings, dtype="float32")
    return ids, texts, embeddings

def build_and_save_index(embeddings, index_file=INDEX_FILE):
    """
    Build FAISS index for fast similarity search.
    - L2 distance (Euclidean) used.
    - Saves the index to disk for reuse.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print(f"✅ FAISS index saved to {index_file}")

if __name__ == "__main__":
    # Step 1: Load embeddings from JSONL
    ids, texts, embeddings = load_embeddings(EMBEDDINGS_FILE)
    
    # Step 2: Build and save FAISS index
    build_and_save_index(embeddings)
