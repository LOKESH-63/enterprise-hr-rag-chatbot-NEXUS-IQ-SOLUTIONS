# app.py
import streamlit as st
import re
import os
from langchain_community.document_loaders import PyPDFLoader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 ABC Technologies Pvt. Ltd.")
st.caption(
    "HR Policy Assistant — retrieves exact policy sections from the official HR document"
)

# ---------------- PDF PATH ----------------
PDF_PATH = "Sample_HR_Policy_Document.pdf"
if not os.path.exists(PDF_PATH):
    st.error("HR Policy PDF not found. Please upload the policy PDF.")
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
    """
    Extract numbered sections like:
    7. IT and Security Policy
    """
    pattern = re.compile(r"\n\s*(\d+)\.\s*([A-Za-z &]+)\n")
    matches = list(pattern.finditer(text))

    sections = {}
    for i, match in enumerate(matches):
        start = match.end()
        title = match.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        key = re.sub(r"[^a-z ]", "", title.lower()).strip()
        sections[key] = {
            "title": title,
            "content": content
        }

    return sections

sections = extract_sections(full_text)

# ---------------- CLEAN CONTENT ----------------
def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ---------------- SPLIT INTO POINTS ----------------
def split_points(text):
    points = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 10:
            points.append(line)
    return points

# ---------------- SECTION KEYWORDS ----------------
SECTION_KEYWORDS = {
    "working hours policy": ["working hours", "office hours", "timing"],
    "leave policy": ["leave", "paid leave", "sick leave", "casual leave"],
    "work from home policy": ["wfh", "work from home", "remote"],
    "attendance and punctuality policy": ["attendance", "punctuality", "late"],
    "code of conduct": ["conduct", "behavior", "harassment"],
    "it and security policy": ["it", "security", "password", "confidential"],
    "performance management policy": ["performance", "kpi", "appraisal"],
    "training and development policy": ["training", "learning"],
    "insurance and health benefits policy": ["insurance", "medical", "health"],
    "grievance redressal policy": ["grievance", "complaint"],
    "disciplinary action": ["disciplinary", "discipline"],
    "policy amendments": ["amendment", "modify"]
}

# ---------------- DETECT SECTION ----------------
def detect_section(question):
    q = question.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return section
    return None

# ---------------- GREETING ----------------
def is_greeting(text):
    return text.lower().strip() in {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
    }

# ---------------- UI ----------------
question = st.text_input("Enter your question")

if question:
    if is_greeting(question):
        st.success(
            "Hello 👋 I’m your HR Policy Assistant.\n\n"
            "You can ask about Leave, WFH, IT & Security, Attendance, Insurance, and more."
        )
    else:
        section_key = detect_section(question)

        if section_key and section_key in sections:
            section = sections[section_key]
            cleaned = clean_text(section["content"])
            points = split_points(cleaned)

            st.markdown(f"### {section['title']}")
            for p in points:
                st.markdown(f"- {p}")

        else:
            st.warning(
                "I checked the HR policy document, but this specific information is not mentioned.\n\n"
                "Please contact the HR team for further clarification."
            )
