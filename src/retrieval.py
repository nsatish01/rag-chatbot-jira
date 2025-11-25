import faiss
import numpy as np
import pickle
import ollama
import os

FAISS_INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.pkl"


def embed_query(query: str, model="llama3.1") -> np.ndarray:
    """
    Generates an embedding for the query using Ollama.
    """
    response = ollama.embeddings(model=model, prompt=query)
    embedding = np.array(response["embedding"]).astype("float32")
    return embedding


def load_faiss_and_metadata():
    """
    Loads FAISS index and metadata (chunks list).
    """
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def search(query: str, k=5):
    """
    Performs semantic search on FAISS and returns top-k chunks.
    """
    # Lo
