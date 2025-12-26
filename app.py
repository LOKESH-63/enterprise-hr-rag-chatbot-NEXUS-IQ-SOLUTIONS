import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")

st.title("🏢 NEXUS IQ SOLUTIONS")
st.caption("HR Policy Assistant")

# ---------------- PDF PATH (LOCAL) ----------------
PDF_PATH = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_PATH):
    st.error("❌ HR Policy PDF not found. Please place 'Sample_HR_Policy_Document.pdf' in the project folder.")
    st.stop()

# ---------------- LOAD PDF ----------------
@st.cache_resource
def load_pdf_text(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)

full_text = load_pdf_text(PDF_PATH)

# ---------------- EXTRACT SECTIONS ----------------
def extract_sections(text):
    pattern = re.compile(r"\n\s*(\d+)\.\s*([A-Za-z &()/-]+)\s*\n")
    matches = list(pattern.finditer(text))

    sections = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(2).strip()
        key = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
        sections[key] = {
            "title": title,
            "content": text[start:end].strip()
        }
    return sections

sections = extract_sections(full_text)

# ---------------- CLEAN CONTENT ----------------
def clean_text(text):
    text = re.sub(r"(?m)^\s*\d+\s*[•\.\-]\s*", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def split_sentences(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            parts = re.split(r'(?<=[.!?])\s+', line)
            lines.extend(parts)
    return [l for l in lines if len(l) > 6]

# ---------------- DETECT POLICY ----------------
def detect_policies(question):
    q = question.lower()
    found = []

    for key, sec in sections.items():
        if any(word in q for word in key.split()):
            found.append(key)

    if "leave" in q:
        found.append("leave policy")
    if "wfh" in q or "work from home" in q:
        found.append("work from home wfh policy")
    if "performance" in q or "kpi" in q:
        found.append("performance management policy")
    if "working hours" in q or "attendance" in q:
        found.append("working hours policy")
    if "security" in q or "it" in q:
        found.append("it and security policy")
    if "insurance" in q or "medical" in q:
        found.append("insurance and health benefits policy")
    if "conduct" in q:
        found.append("code of conduct")

    return list(dict.fromkeys(found))

# ---------------- GREETING ----------------
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

# ---------------- UI ----------------
st.write("Enter a question about the HR policies")

question = st.text_input("")

if question:
    if is_greeting(question):
        st.success("Hello 👋 How can I help you with HR policies today?")
    else:
        requested = detect_policies(question)

        shown = False
        for key in requested:
            if key in sections:
                shown = True
                sec = sections[key]
                st.markdown(f"### {sec['title']}")
                cleaned = clean_text(sec["content"])
                sentences = split_sentences(cleaned)
                for s in sentences:
                    st.markdown(s)

        if not shown:
            st.warning(
                "I checked the HR policy document, but this information is not mentioned.\n\n"
                "Please contact the HR team for further clarification."
            )
