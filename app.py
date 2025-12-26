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
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 HR Policy Assistant")
st.caption("RAG-based • Clean & factual answers • Free models")

# ======================================================
# PDF FILE
# ======================================================
PDF_FILE = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_FILE):
    st.error("HR Policy PDF not found. Please add Sample_HR_Policy_Document.pdf")
    st.stop()

# ======================================================
# STRONG CLEANING FUNCTION (NUMBERS + ROMAN NUMERALS)
# ======================================================
def clean_policy_text(text: str) -> str:
    # Remove numeric clauses (2.2, 3.4.1)
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)

    # Remove roman numerals (i., ii., iii., iv.)
    text = re.sub(
        r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove list markers like a), 1)
    text = re.sub(r"\b[a-zA-Z]\)", "", text)
    text = re.sub(r"\b\d+\)", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ======================================================
# DETECT FACTUAL QUESTIONS (KEY FEATURE)
# ======================================================
def is_factual_question(question: str) -> bool:
    keywords = [
        "how many", "number of", "entitled", "per year",
        "paid leave", "paid leaves",
        "casual leave", "sick leave",
        "days", "leave count"
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
    llm = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return embedder, index, texts, llm

embedder, index, texts, llm = load_rag()

# ======================================================
# ANSWER FUNCTION (SUMMARY + FACTUAL MODES)
# ======================================================
def answer_question(question: str) -> str:
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=3)

    # Use only top chunk
    raw_context = texts[idx[0][0]]
    context = clean_policy_text(raw_context)

    factual = is_factual_question(question)

    if factual:
        prompt = f"""
You are an HR policy assistant.

TASK:
- Answer using EXACT facts from the policy
- Include numbers if they are mentioned
- Be clear and direct
- Do NOT include clause numbers or headings

If the information is not available, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Direct Answer:
"""
    else:
        prompt = f"""
You are an HR policy assistant.

TASK:
- SUMMARIZE the policy information
- Use natural, professional language
- Limit to 2–3 short sentences
- Do NOT include numbers unless necessary
- Do NOT include clause numbers or headings

If the information is not available, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Summary Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=100,
        temperature=0.0,   # deterministic output
        do_sample=False
    )[0]["generated_text"]

    # FINAL SAFETY CLEAN
    answer = re.sub(r"\b\d+(\.\d+)+\b", "", result)
    answer = re.sub(
        r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\b",
        "",
        answer,
        flags=re.IGNORECASE
    )
    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()

# ======================================================
# UI
# ======================================================
st.subheader("💬 Ask an HR policy question")

question = st.text_input("Enter your question")

if question:
    if question.lower() in ["hi", "hello", "hey"]:
        st.info("Hello 👋 I’m your HR Policy Assistant.")
    else:
        answer = answer_question(question)
        st.success(answer)

