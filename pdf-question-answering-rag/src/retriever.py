import numpy as np

def retrieve_top_chunks_faiss(
    index,
    question_embedding: np.ndarray,
    chunks: list,
    top_k: int = 3
):
    """
    Retrieves top-k chunks using FAISS index.
    """
    # FAISS expects 2D array
    question_embedding = question_embedding.reshape(1, -1)

    distances, indices = index.search(question_embedding, top_k)

    top_chunks = [chunks[i] for i in indices[0]]

    return top_chunks
