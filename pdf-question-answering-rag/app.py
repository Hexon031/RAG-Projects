import streamlit as st
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

st.set_page_config(page_title="PDF RAG App", layout="centered")

st.title("📄 PDF Question Answering System")

UPLOAD_DIR = "uploads"
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# ---------------- PDF Upload ----------------
st.header("1️⃣ Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)

    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    with st.spinner("Processing PDF and building index..."):
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        index = build_faiss_index(embeddings)

        save_faiss_index(index, INDEX_PATH)
        save_chunks(chunks, CHUNKS_PATH)

    st.success("Document indexed successfully.")

# ---------------- Question Answering ----------------
st.header("2️⃣ Ask a Question")

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    question = st.text_input("Enter your question")

    if st.button("Get Answer") and question.strip():
        index = load_faiss_index(INDEX_PATH)
        chunks = load_chunks(CHUNKS_PATH)

        question_embedding = embed_texts([question])[0]

        top_chunks = retrieve_top_chunks_faiss(
            index,
            question_embedding,
            chunks,
            top_k=3
        )

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, top_chunks)

        st.subheader("Answer")
        st.write(answer)
else:
    st.info("Please upload and index a PDF first.")
