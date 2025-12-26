import streamlit as st
import faiss
import numpy as np
import os
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="NEXUS IQ HR Chatbot", page_icon="🏢")

st.markdown("## 🏢 NEXUS IQ SOLUTIONS")
st.caption("RAG-based • Clean HR answers • Section-aware retrieval")

st.markdown("### 💬 Ask an HR policy question")

# ======================================================
# PDF FILE
# ======================================================
PDF_FILE = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_FILE):
    st.error("HR Policy PDF not found. Please add Sample_HR_Policy_Document.pdf")
    st.stop()

# ======================================================
# CLEAN POLICY TEXT
# ======================================================
def clean_policy_text(text: str) -> str:
    # Remove clause numbers (1, 2.3, 6.1.2)
    text = re.sub(r"\b\d+(\.\d+)*\b", "", text)

    # Remove roman numerals
    text = re.sub(
        r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b\.?",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove policy labels
    text = re.sub(
        r"\b(working hours|leave|wfh|work from home|it and security|information security|security)\s+policy\b",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\bpolicy\b", "", text, flags=re.IGNORECASE)

    # Remove bullets/dashes
    text = re.sub(r"[•–—-]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ======================================================
# SECTION KEYWORD DETECTION
# ======================================================
def detect_section_keywords(question: str) -> list:
    q = question.lower()

    if "security" in q or "it" in q:
        return ["security", "password", "device", "data"]

    if "leave" in q:
        return ["leave", "casual", "sick"]

    if "working hours" in q or "office time" in q:
        return ["working hours", "office", "time"]

    if "work from home" in q or "wfh" in q:
        return ["work from home", "wfh", "remote"]

    return []

# ======================================================
# EXTRACT RELEVANT SENTENCES
# ======================================================
def extract_relevant_sentences(context: str, question: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    q_words = [w for w in question.lower().split() if len(w) > 3]

    relevant = [
        s.strip()
        for s in sentences
        if any(w in s.lower() for w in q_words)
    ]

    return relevant[:4]

# ======================================================
# LOAD RAG PIPELINE
# ======================================================
@st.cache_resource
def load_rag():
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

    return embedder, texts, embeddings

embedder, texts, embeddings = load_rag()

# ======================================================
# ANSWER FUNCTION (SECTION-AWARE)
# ======================================================
def answer_question(question: str) -> list:
    keywords = detect_section_keywords(question)

    # Filter chunks by detected section
    if keywords:
        filtered = [
            t for t in texts
            if any(k in t.lower() for k in keywords)
        ]
    else:
        filtered = texts

    if not filtered:
        return []

    filtered_embeddings = embedder.encode(filtered)

    index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    index.add(np.array(filtered_embeddings).astype("float32"))

    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=1)

    raw_context = filtered[idx[0][0]]
    cleaned_context = clean_policy_text(raw_context)

    sentences = extract_relevant_sentences(cleaned_context, question)

    return sentences

# ======================================================
# UI INPUT & DISPLAY
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
        for sentence in sentences:
            st.markdown(sentence)
