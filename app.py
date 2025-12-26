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
st.title("🏢 NEXUS IQ SOLUTIONS")
st.caption("Free • RAG-based • Clean summarized answers")

# ======================================================
# PDF FILE
# ======================================================
PDF_FILE = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_FILE):
    st.error("HR Policy PDF not found. Please add Sample_HR_Policy_Document.pdf")
    st.stop()

# ======================================================
# STRONG CLEANING FUNCTION (KEY FIX)
# ======================================================
def clean_policy_text(text: str) -> str:
    # Remove numbering like 2.2, 3.4.1
    text = re.sub(r"\b\d+(\.\d+)+\b", "", text)

    # Remove line-start numbers
    text = re.sub(r"^\s*\d+\s*", "", text, flags=re.MULTILINE)

    # Remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ======================================================
# LOAD RAG PIPELINE
# ======================================================
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

    # FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    # FREE LLM
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
# ANSWER FUNCTION (STRICT SUMMARY)
# ======================================================
def answer_question(question: str) -> str:
    q_emb = embedder.encode([question])
    _, idx = index.search(np.array(q_emb).astype("float32"), k=3)

    # Use ONLY ONE chunk
    raw_context = texts[idx[0][0]]
    context = clean_policy_text(raw_context)

    prompt = f"""
You are an HR policy assistant.

Your task is to SUMMARIZE the information below.

STRICT RULES:
- DO NOT copy sentences from the policy
- DO NOT include numbers, clauses, or section references
- Use simple, professional language
- Limit to 2–3 short sentences
- Answer ONLY from the policy content

If the answer is not available, reply exactly:
"I checked the HR policy document, but this information is not mentioned."

Policy Content:
{context}

Question:
{question}

Summary Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=80,
        temperature=0.1,
        do_sample=False
    )[0]["generated_text"]

    # FINAL SAFETY CLEAN
    answer = re.sub(r"\b\d+(\.\d+)+\b", "", result)
    answer = re.sub(r"^\s*\d+\s*", "", answer)
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
