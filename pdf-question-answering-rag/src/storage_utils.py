import faiss
import pickle
import os

def save_faiss_index(index, index_path: str):
    faiss.write_index(index, index_path)

def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

def save_chunks(chunks: list, chunks_path: str):
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(chunks_path: str):
    with open(chunks_path, "rb") as f:
        return pickle.load(f)
