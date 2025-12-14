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

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="PDF Question Answering (RAG)",
    layout="centered"
)

# ---------------- Custom Styling ----------------
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Constants ----------------
UPLOAD_DIR = "uploads"
STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

# ---------------- Title ----------------
st.title("📄 PDF Question Answering System")
st.caption(
    "Retrieval-Augmented Generation (RAG) using FAISS and a local LLM"
)

st.divider()

# ======================================================
# 1️⃣ PDF UPLOAD & INDEXING
# ======================================================
st.subheader("📤 Upload PDF Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)

    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    # Save PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    with st.spinner("Processing PDF and building FAISS index..."):
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        index = build_faiss_index(embeddings)

        save_faiss_index(index, INDEX_PATH)
        save_chunks(chunks, CHUNKS_PATH)

    st.success("Document indexed successfully. You can now ask questions.")

st.divider()

# ======================================================
# 2️⃣ QUESTION ANSWERING (RAG)
# ======================================================
st.subheader("❓ Ask a Question")

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):

    question = st.text_input(
        "Enter your question about the document"
    )

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

        # Save to chat history
        st.session_state.chat_history.append(
            {"question": question, "answer": answer}
        )

        # Display answer
        st.subheader("💡 Answer")
        st.success(answer)

        # Optional: show retrieved context
        with st.expander("🔍 Retrieved Context"):
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk[:500] + "...")
                st.divider()

else:
    st.info("Please upload and index a PDF first.")

# ======================================================
# 3️⃣ CHAT HISTORY
# ======================================================
if st.session_state.chat_history:
    st.divider()
    st.subheader("🗨️ Chat History")

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.divider()
