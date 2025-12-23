import streamlit as st
import faiss
import numpy as np
import os
import pandas as pd
from datetime import datetime

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Enterprise HR Assistant", page_icon="🏢")

DATA_DIR = "data"

# ---------------- LOGIN SYSTEM ----------------
USERS = {
    "employee": {"password": "employee123", "role": "Employee"},
    "hr": {"password": "hr123", "role": "HR"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login – HR Policy Assistant")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        username = username.strip()
        password = password.strip()

        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- HEADER ----------------
st.title("🏢 Enterprise RAG-based HR Chatbot")
st.caption(f"Logged in as: **{st.session_state.role}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- CHECK PDF ----------------
if not os.path.exists(DATA_DIR):
    st.error("❌ data folder not found")
    st.stop()

pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

if not pdf_files:
    st.error("❌ No HR policy PDF found in data folder")
    st.stop()

# ---------------- LOAD RAG PIPELINE ----------------
@st.cache_resource
def load_pipeline(pdf_files):
    texts, sources = [], []

    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        for c in chunks:
            texts.append(c.page_content)
            sources.append(f"{file} - page {c.metadata.get('page', 0)}")

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, sources, llm

embedder, index, texts, sources, llm = load_pipeline(pdf_files)

# ---------------- RAG QUERY ----------------
def answer_query(query):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, idx = index.search(np.array(q_emb), k=5)

    filtered = [
        (texts[i], sources[i])
        for i, s in zip(idx[0], scores[0])
        if s > 0.35
    ]

    if not filtered:
        return "This information is not mentioned in the HR policy.", None

    context = "\n".join([t for t, _ in filtered])

    prompt = f"""
Answer ONLY from HR policy document.
If not found, say it is not mentioned.

HR Content:
{context}

Question:
{query}
"""

    answer = llm(prompt, max_length=200, temperature=0.2)[0]["generated_text"]
    return answer, [s for _, s in filtered]

# ---------------- CHAT UI ----------------
st.subheader("💬 Ask HR Policy Question")
query = st.text_input("Enter your question")

if query:
    answer, src = answer_query(query)
    st.success(answer)

    with st.expander("📄 Source"):
        if src:
            for s in src:
                st.write(s)
