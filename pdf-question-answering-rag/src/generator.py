import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load model once
_llm = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(question: str, context_chunks: list) -> str:
    """
    Generates an answer using retrieved context chunks.
    """
    context = "\n".join(context_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present in the context, say "Answer not found in document".

Context:
{context}

Question:
{question}
"""

    response = _llm.generate_content(prompt)
    return response.text.strip()
