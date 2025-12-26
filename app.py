import streamlit as st
import faiss
import numpy as np
import os
from PIL import Image

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🏢")

# ---------------- FILE PATHS ----------------
PDF_FILE = "Sample_HR_Policy_Document.pdf"
LOGO_FILE = "nexus_iq_logo.png"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, PDF_FILE)
LOGO_PATH = os.path.join(BASE_DIR, LOGO_FILE)

# ---------------- LOGIN USERS ----------------
USERS = {
    "employee": {"password": "employee123", "role": "Employee"},
    "hr": {"password": "hr123", "role": "HR"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN FUNCTION ----------------
def login():
    st.title("🔐 Login – HR Policy Assistant")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = USERS[username]["role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- HEADER ----------------
if os.path.exists(LOGO_PATH):
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(Image.open(LOGO_PATH), width=70)
    with col2:
        st.markdown("## **NEXUS IQ SOLUTIONS**")
else:
    st.markdown("## **NEXUS IQ SOLUTIONS**")

st.caption(f"Logged in as: **{st.session_state.role}**")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- PDF CHECK ----------------
if not os.path.exists(PDF_PATH):
    st.error("❌ HR Policy PDF not found.")
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
    index.add(np.array(embeddings).astype("float32"))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm

embedder, index, texts, llm = load_pipeline()

# ---------------- GREETING CHECK ----------------
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    text = text.lower()
    return any(greet in text for greet in greetings)

# ---------------- ANSWER FUNCTION (SUMMARY ONLY) ----------------
def answer_query(question):
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=3)

    # Use ONLY top 1 chunk to avoid dumping
    context = texts[idx[0][0]]

    prompt = f"""
You are a professional HR assistant.

TASK:
- Answer using ONLY the policy content below
- Provide a SHORT and NATURAL SUMMARY
- Limit to 2–3 sentences
- Do NOT copy policy text
- Do NOT repeat information
- Do NOT mention clauses or page numbers

If information is not available, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Final Answer (summary only):
"""

    response = llm(
        prompt,
        max_new_tokens=80,
        temperature=0.2,
        do_sample=False
    )[0]["generated_text"]

    return response.strip()

# ---------------- UI ----------------
st.subheader("💬 Ask HR Policy Question")
question = st.text_input("Enter your question")

if question:
    if is_greeting(question):
        st.info(
            "Hello 👋 I’m your HR Policy Assistant.\n\n"
            "You can ask questions like:\n"
            "- What is the leave policy?\n"
            "- What is the notice period?\n"
            "- How many casual leaves are allowed?"
        )
    else:
        st.success(answer_query(question))
