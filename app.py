import streamlit as st
import faiss
import numpy as np
import os
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🏢")

st.title("🏢 HR Policy Assistant (Free Version)")
st.caption("Powered by open-source models • No API key required")

# ---------------- PDF PATH ----------------
PDF_FILE = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_FILE):
    st.error("HR Policy PDF not found. Please add Sample_HR_Policy_Document.pdf")
    st.stop()

# ---------------- CLEAN POLICY TEXT ----------------
def clean_policy_text(text):
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)          # remove 2.1, 3.4 etc.
    text = re.sub(r"^\s*\d+\s*", "", text, flags=re.M)  # remove line numbers
    return text.strip()

# ---------------- LOAD RAG PIPELINE ----------------
@st.cache_resource
def load_rag():
    # Load PDF
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

    # Embeddings (FREE)
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts)

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    # LLM (FREE)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return embedder, index, texts, llm

embedder, index, texts, llm = load_rag()

# ---------------- ANSWER FUNCTION ----------------
def answer_question(question):
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=3)

    # Use only top chunk
    raw_context = texts[idx[0][0]]
    context = clean_policy_text(raw_context)

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer ONLY using the policy content below
- Provide a SHORT and NATURAL SUMMARY (2–3 sentences)
- Do NOT include numbers, clauses, or section references
- Do NOT assume information

If the answer is not available, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Final Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=100,
        temperature=0.2,
        do_sample=False
    )[0]["generated_text"]

    return result.strip()

# ---------------- UI ----------------
st.subheader("💬 Ask a question")

question = st.text_input("Enter your HR policy question")

if question:
    if question.lower() in ["hi", "hello", "hey"]:
        st.info("Hello 👋 I’m your HR Policy Assistant.")
    else:
        answer = answer_question(question)
        st.success(answer)
