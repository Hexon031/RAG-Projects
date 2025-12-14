from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

_qa_pipeline = load_qa_pipeline()

def generate_answer(question: str, context_chunks: list) -> str:
    context = "\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present in the context, say "Answer not found in document".

Context:
{context}

Question:
{question}
"""

    result = _qa_pipeline(
        prompt,
        max_length=256,
        do_sample=False
    )
    return result[0]["generated_text"]
