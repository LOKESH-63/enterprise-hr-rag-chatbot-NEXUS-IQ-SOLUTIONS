import streamlit as st
import faiss
import numpy as np
import os
import re

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="NEXUS IQ HR Chatbot", page_icon="🏢")
st.markdown("## 🏢 NEXUS IQ SOLUTIONS")
st.caption("RAG-based • Clean • Accurate HR answers")
st.markdown("### 💬 Ask an HR policy question")

# ======================================================
# PDF FILE
# ======================================================
PDF_FILE = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_FILE):
    st.error("HR Policy PDF not found.")
    st.stop()

# ======================================================
# CLEAN TEXT (REMOVE STRUCTURE, KEEP MEANING)
# ======================================================
def clean_policy_text(text: str) -> str:
    # Remove section headers like "4. Work From Home (WFH) Policy"
    text = re.sub(
        r"\n?\d+\.\s*[A-Za-z &()]+Policy\n?",
        "\n",
        text
    )

    # Remove leading numbering like "1 •", "2 •"
    text = re.sub(
        r"^\s*\d+\s*[•.-]?\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove bullets only
    text = re.sub(r"[•–—]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ======================================================
# DETECT SECTION (FOR ACCURATE RETRIEVAL)
# ======================================================
def detect_section_keywords(question: str) -> list:
    q = question.lower()

    if "work from home" in q or "wfh" in q:
        return ["work from home", "wfh"]

    if "leave" in q:
        return ["leave"]

    if "security" in q or "it" in q:
        return ["security", "password", "device", "data"]

    if "working hours" in q or "office time" in q:
        return ["working hours"]

    return []

# ======================================================
# LOAD DOCUMENTS + EMBEDDINGS
# ======================================================
@st.cache_resource
def load_data():
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embeddings = embedder.encode(texts)

    return texts, embeddings, embedder

texts, embeddings, embedder = load_data()

# ======================================================
# ANSWER FUNCTION (SECTION-ACCURATE, COMPLETE)
# ======================================================
def answer_question(question: str) -> list:
    keywords = detect_section_keywords(question)

    # Filter chunks by section keywords
    if keywords:
        filtered_texts = [
            t for t in texts
            if any(k in t.lower() for k in keywords)
        ]
    else:
        filtered_texts = texts

    if not filtered_texts:
        return []

    filtered_embeddings = embedder.encode(filtered_texts)
    index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    index.add(np.array(filtered_embeddings).astype("float32"))

    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=1)

    raw_context = filtered_texts[idx[0][0]]
    cleaned = clean_policy_text(raw_context)

    # Split into sentences (KEEP ALL RULES)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)

    return [s.strip() for s in sentences if len(s.strip()) > 10]

# ======================================================
# UI
# ======================================================
question = st.text_input("Enter your question")

if question:
    sentences = answer_question(question)

    if not sentences:
        st.warning(
            "I checked the HR policy document, but this information is not mentioned."
        )
    else:
        st.success("Here is the policy information:")
        for s in sentences:
            st.markdown(s)

