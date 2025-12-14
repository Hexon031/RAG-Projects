from sentence_transformers import SentenceTransformer

# Load model once (important for performance)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list):
    """
    Converts a list of text chunks into embeddings.
    """
    return _model.encode(texts)
