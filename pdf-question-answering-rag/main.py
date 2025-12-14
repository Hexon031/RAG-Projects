import os
from src.pdf_loader import load_pdf_text
from src.chunker import chunk_text
from src.embedder import embed_texts
from src.faiss_index import build_faiss_index
from src.retriever import retrieve_top_chunks_faiss
from src.generator import generate_answer
from src.storage_utils import (
    save_faiss_index,
    load_faiss_index,
    save_chunks,
    load_chunks
)

INDEX_PATH = "storage/faiss.index"
CHUNKS_PATH = "storage/chunks.pkl"
PDF_PATH = "data/sample.pdf"

# -------- Build or Load Index --------
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("Loading existing FAISS index...")
    index = load_faiss_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)
else:
    print("Building FAISS index for first time...")
    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    index = build_faiss_index(embeddings)

    os.makedirs("storage", exist_ok=True)
    save_faiss_index(index, INDEX_PATH)
    save_chunks(chunks, CHUNKS_PATH)

# -------- Question --------
question = "What is the name of the project's guide?"
question_embedding = embed_texts([question])[0]

# -------- Retrieval --------
top_chunks = retrieve_top_chunks_faiss(
    index,
    question_embedding,
    chunks,
    top_k=3
)

# -------- Answer --------
answer = generate_answer(question, top_chunks)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)
