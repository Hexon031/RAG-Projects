import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS index from embeddings.
    """
    dimension = embeddings.shape[1]

    # Create index (L2 distance)
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    index.add(embeddings)

    return index
