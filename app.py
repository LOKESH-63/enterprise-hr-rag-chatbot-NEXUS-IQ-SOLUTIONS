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
st.caption("RAG-based • Clean & step-by-step answers • Free models")

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
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)
    text = re.sub(
        r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b\.?",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"[•–—-]", " ", text)
    text = re.sub(r"\b[a-zA-Z]\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================================================
# EXTRACT RELEVANT SENTENCES
# ======================================================
def extract_relevant_sentences(context: str, question: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    q_words = [w for w in question.lower().split() if len(w) > 3]

    relevant = [
        s.strip() for s in sentences
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

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm

embedder, index, texts, llm = load_rag()

# ======================================================
# ANSWER FUNCTION (STEP-BY-STEP OUTPUT)
# ======================================================
def answer_question(question: str) -> str:
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=3)

    raw_context = texts[idx[0][0]]
    cleaned_context = clean_policy_text(raw_context)

    extracted_sentences = extract_relevant_sentences(
        cleaned_context, question
    )

    if not extracted_sentences:
        return "I checked the HR policy document, but this information is not mentioned."

    steps = []
    for i, sentence in enumerate(extracted_sentences, start=1):
        steps.append(f"Step {i}: {sentence}")

    return "\n".join(steps)

# ======================================================
# UI INPUT
# ======================================================
question = st.text_input("Enter your question")

if question:
    answer = answer_question(question)
    st.success(answer)

