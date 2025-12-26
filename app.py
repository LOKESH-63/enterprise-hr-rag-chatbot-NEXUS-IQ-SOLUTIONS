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
st.caption("RAG-based • Clean & factual answers • Free models")

# ======================================================
# INSTRUCTION TEXT (YOUR REQUEST)
# ======================================================
st.markdown(
    """
### 💬 Ask an HR policy question  
**Examples:**  
- *What is the leave policy?*  
- *How to avail medical insurance?*  
- *How many sick leaves are allowed?*  
"""
)

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
def extract_relevant_sentences(context: str, question: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", context)
    q_words = [w for w in question.lower().split() if len(w) > 3]

    relevant = [
        s for s in sentences
        if any(w in s.lower() for w in q_words)
    ]

    return " ".join(relevant[:4])

# ======================================================
# DETECT POLICY OVERVIEW QUESTIONS
# ======================================================
def is_policy_overview(question: str) -> bool:
    keywords = [
        "leave policy",
        "what is the leave policy",
        "medical insurance",
        "insurance policy",
        "policy details",
        "types of leave"
    ]
    q = question.lower()
    return any(k in q for k in keywords)

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

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    return embedder, index, texts, llm

embedder, index, texts, llm = load_rag()

# ======================================================
# ANSWER FUNCTION
# ======================================================
def answer_question(question: str) -> str:
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=3)

    raw_context = texts[idx[0][0]]
    cleaned_context = clean_policy_text(raw_context)

    extracted = extract_relevant_sentences(cleaned_context, question)

    if not extracted.strip():
        return "I checked the HR policy document, but this information is not mentioned."

    overview = is_policy_overview(question)

    if overview:
        prompt = f"""
You are an HR assistant.

Convert the information below into clear bullet points.

RULES:
- Use bullet points (•)
- One fact per bullet
- Include numbers if present
- Do NOT include headings or section titles
- Do NOT include roman numerals

Information:
{extracted}

Bullet Point Answer:
"""
    else:
        prompt = f"""
You are an HR assistant.

Rewrite the information below into a clean, natural answer.

RULES:
- Write 2–3 professional sentences
- Do NOT include headings or numbering
- Use simple employee-friendly language

Information:
{extracted}

Final Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=100,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    result = re.sub(
        r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
        "",
        result,
        flags=re.IGNORECASE
    )
    result = re.sub(r"\s+", " ", result)

    return result.strip()

# ======================================================
# UI INPUT
# ======================================================
question = st.text_input("Enter your question")

if question:
    answer = answer_question(question)
    st.success(answer)
