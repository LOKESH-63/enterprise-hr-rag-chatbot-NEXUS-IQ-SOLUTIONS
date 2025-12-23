import streamlit as st
import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🏢")

PDF_FILE = "Sample_HR_Policy_Document.pdf"

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
st.title("🏢 Enterprise HR Policy Chatbot")
st.caption(f"Logged in as: **{st.session_state.role}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- CHECK PDF ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, PDF_FILE)

if not os.path.exists(PDF_PATH):
    st.error(f"❌ {PDF_FILE} not found in repository root.")
    st.stop()

# ---------------- LOAD RAG PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm

embedder, index, texts, llm = load_pipeline()

# ---------------- QUERY FUNCTION ----------------
def answer_query(question):
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb), k=3)

    context = " ".join([texts[i] for i in idx[0]])

    prompt = f"""
Answer ONLY from the HR policy document.
If the answer is not present, say politely:
"I checked the HR policy document, but this information is not mentioned."

Context:
{context}

Question:
{question}
"""

    return llm(prompt, max_length=200, temperature=0.2)[0]["generated_text"]

# ---------------- CHAT UI ----------------
st.subheader("💬 Ask HR Policy Question")
question = st.text_input("Enter your question")

if question:
    answer = answer_query(question)
    st.success(answer)

