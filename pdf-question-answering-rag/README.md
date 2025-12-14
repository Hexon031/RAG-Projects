# 📄 PDF Question Answering System (RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions grounded strictly in the document content.

## 🚀 Features
- PDF upload and text extraction
- Chunking with overlap for semantic accuracy
- Sentence Transformer embeddings
- FAISS vector database for fast similarity search
- Local LLM (FLAN-T5) for answer generation
- Hallucination control (document-grounded answers only)
- Streamlit-based interactive web interface
- Persistent FAISS index for faster startup
- Chat history for conversational Q&A

## 🧠 Architecture
PDF → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer

## 🛠️ Tech Stack
- Python
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- Streamlit
- Retrieval-Augmented Generation (RAG)

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
🌐 Live Demo

🔗 Deployed on Streamlit Community Cloud

📌 Use Cases

Document-based Q&A

Knowledge assistants

Educational notes search

Enterprise document intelligence