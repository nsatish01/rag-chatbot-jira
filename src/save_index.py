import faiss
import numpy as np
import pickle
import os

EMBEDDINGS_PATH = "data/embeddings.npy"
CHUNKS_PATH = "data/chunks.pkl"
FAISS_INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.pkl"

def save_faiss_index():
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    # Load chunks metadata
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Save metadata to map back to chunks
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Metadata saved to: {METADATA_PATH}")

if __name__ == "__main__":
    save_faiss_index()
