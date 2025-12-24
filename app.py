import streamlit as st
import faiss
import numpy as np
import os
import re
from PIL import Image
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# ---------------- LOGIN ----------------
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

# ---------------- CLEAN POLICY TEXT ----------------
def clean_policy_text(text):
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)
    text = re.sub(r"^\s*\d+\s*", "", text, flags=re.MULTILINE)
    return text.strip()

# ---------------- OPENAI EMBEDDING ----------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ---------------- LOAD RAG PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

    embeddings = np.vstack([get_embedding(t) for t in texts])

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts

index, texts = load_pipeline()

# ---------------- GREETING ----------------
def is_greeting(text):
    return any(g in text.lower() for g in ["hi", "hello", "hey", "good morning"])

# ---------------- ANSWER FUNCTION ----------------
def answer_query(question):
    q_emb = get_embedding(question)
    _, idx = index.search(np.array([q_emb]), k=3)

    raw_context = texts[idx[0][0]]
    context = clean_policy_text(raw_context)

    prompt = f"""
You are a professional HR assistant.

Rules:
- Answer ONLY from the policy content
- Provide a SHORT summary (2–3 sentences)
- Do NOT include numbers, clauses, or sections
- Do NOT assume information

If the answer is missing, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Final Answer:
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120
    )

    return completion.choices[0].message.content.strip()

# ---------------- UI ----------------
st.subheader("💬 Ask HR Policy Question")
question = st.text_input("Enter your question")

if question:
    if is_greeting(question):
        st.info("Hello 👋 I’m your HR Policy Assistant.")
    else:
        st.success(answer_query(question))
