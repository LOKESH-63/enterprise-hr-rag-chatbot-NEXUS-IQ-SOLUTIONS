# app.py
import streamlit as st
import re
import os
from langchain_community.document_loaders import PyPDFLoader

# ---------------- PAGE ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 NEXUS IQ SOLUTIONS")
st.caption(
    "Section-aware HR policy assistant — retrieves exact policy content from the official HR document"
)

# ---------------- PDF PATH ----------------
PDF_PATH = "Sample_HR_Policy_Document.pdf"
if not os.path.exists(PDF_PATH):
    st.error(f"HR Policy PDF not found at {PDF_PATH}. Please upload the PDF to this path.")
    st.stop()

# ---------------- LOAD PDF ----------------
@st.cache_resource
def load_pdf_text(path: str) -> str:
    loader = PyPDFLoader(path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)

full_text = load_pdf_text(PDF_PATH)

# ---------------- EXTRACT SECTIONS ----------------
def extract_sections(text: str) -> dict:
    """
    Extract sections based on headings like:
    3. Leave Policy
    """
    pattern = re.compile(r"\n\s*(\d+)\.\s*([A-Za-z0-9 &()/-]+)\s*\n")
    matches = list(pattern.finditer(text))

    sections = {}
    if not matches:
        sections["document"] = {"title": "HR Policy Document", "content": text}
        return sections

    for i, m in enumerate(matches):
        start = m.end()
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        key = re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()
        sections[key] = {"title": title, "content": content}

    return sections

sections = extract_sections(full_text)

# ---------------- CLEAN SECTION CONTENT ----------------
def clean_section_text(text: str) -> str:
    # remove bullet numbers at start of lines
    text = re.sub(r"(?m)^\s*\d+\s*[•\.\-\)]\s*", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ---------------- SPLIT INTO SENTENCES ----------------
def split_to_sentences(text: str) -> list:
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            if len(p.strip()) > 6:
                sentences.append(p.strip())
    return sentences

# ---------------- BUILD KEYWORD ALIASES ----------------
def build_section_aliases(sections_dict: dict) -> dict:
    aliases = {}
    for key, meta in sections_dict.items():
        title = meta["title"].lower()
        words = [key]

        if "leave" in title:
            words += ["leave", "leave policy", "paid leave", "casual leave", "sick leave"]
        if "work from home" in title or "wfh" in title:
            words += ["work from home", "wfh", "remote work"]
        if "working hours" in title:
            words += ["working hours", "office hours", "attendance"]
        if "insurance" in title or "health" in title:
            words += ["insurance", "medical insurance", "health insurance"]
        if "performance" in title:
            words += ["performance", "performance management", "kpi"]
        if "training" in title:
            words += ["training", "learning", "development"]
        if "it" in title or "security" in title:
            words += ["it", "security", "password", "data security"]
        if "conduct" in title:
            words += ["conduct", "behavior", "harassment"]
        if "disciplinary" in title:
            words += ["disciplinary", "discipline"]

        aliases[key] = list(set(w.lower() for w in words))
    return aliases

section_aliases = build_section_aliases(sections)

# ---------------- INTENT VALIDATION ----------------
def section_answers_question(question: str, section_text: str) -> bool:
    q = question.lower()
    text = section_text.lower()

    if any(w in q for w in ["how", "process", "avail", "enroll", "apply"]):
        return any(k in text for k in ["apply", "process", "procedure", "portal", "submit", "enroll"])

    if "claim" in q:
        return any(k in text for k in ["claim", "reimbursement", "cashless"])

    return True  # factual questions

# ---------------- DETECT SECTIONS ----------------
def detect_requested_sections(question: str) -> list:
    q = question.lower()
    found = []

    for key, kws in section_aliases.items():
        if any(k in q for k in kws):
            found.append(key)

    if not found:
        q_words = set(re.findall(r"[a-z]{3,}", q))
        for key, meta in sections.items():
            title_words = set(re.findall(r"[a-z]{3,}", meta["title"].lower()))
            if q_words & title_words:
                found.append(key)

    return list(dict.fromkeys(found))

# ---------------- GREETING ----------------
def is_greeting(text: str) -> bool:
    return text.strip().lower() in {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
    }

# ---------------- UI ----------------
question = st.text_input("Enter your question")

if question:
    if is_greeting(question):
        st.success(
            "Hello 👋 I’m your HR Policy Assistant. "
            "You can ask me about leave, working hours, WFH, insurance, IT & security, performance, and more."
        )
    else:
        requested = detect_requested_sections(question)
        shown = False

        for key in requested:
            meta = sections.get(key)
            if not meta:
                continue

            cleaned = clean_section_text(meta["content"])

            if not section_answers_question(question, cleaned):
                continue

            sentences = split_to_sentences(cleaned)
            if not sentences:
                continue

            st.markdown(f"### {meta['title']}")
            for s in sentences:
                st.markdown(s)

            shown = True

        if not shown:
            st.warning(
                "I checked the HR policy document, but this specific information is not mentioned.\n\n"
                "Please contact the HR team for further clarification."
            )   
