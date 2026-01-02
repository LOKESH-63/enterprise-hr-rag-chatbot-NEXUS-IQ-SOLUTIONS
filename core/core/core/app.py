import streamlit as st
import os

from core.pdf_loader import load_pdf_text
from core.section_extractor import extract_sections
from core.text_utils import clean_section_text, split_to_sentences
from core.section_matcher import build_section_aliases, detect_requested_sections

# ---------------- PAGE ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 NEXUS IQ SOLUTIONS")
st.caption("Section-aware HR policy assistant")

# ---------------- PDF ----------------
PDF_PATH = "data/Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_PATH):
    st.error("HR Policy PDF not found.")
    st.stop()

# ---------------- LOAD PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    text = load_pdf_text(PDF_PATH)
    sections = extract_sections(text)
    aliases = build_section_aliases(sections)
    return sections, aliases

sections, section_aliases = load_pipeline()

# ---------------- GREETING ----------------
def is_greeting(text: str) -> bool:
    return text.lower().strip() in {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
    }

# ---------------- UI ----------------
question = st.text_input("Enter your question")

if question:
    if is_greeting(question):
        st.success(
            "Hello 👋 I’m your HR Policy Assistant.\n"
            "Ask me about leave, WFH, insurance, working hours, and more."
        )
    else:
        requested = detect_requested_sections(
            question,
            sections,
            section_aliases
        )

        shown = False

        for key in requested:
            meta = sections.get(key)
            if not meta:
                continue

            cleaned = clean_section_text(meta["content"])
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
