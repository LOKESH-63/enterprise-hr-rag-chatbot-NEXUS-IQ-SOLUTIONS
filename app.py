# app.py
import streamlit as st
import re
import os
from langchain_community.document_loaders import PyPDFLoader

# ---------------- PAGE ----------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 NEXUS IQ SOLUTIONS")
st.caption("Section-aware HR policy assistant — extracts exact policy sections from the uploaded PDF")

# ---------------- PDF PATH (uploaded file) ----------------
PDF_PATH = "Sample_HR_Policy_Document.pdf"
if not os.path.exists(PDF_PATH):
    st.error(f"HR Policy PDF not found at {PDF_PATH}. Please upload the PDF to that path.")
    st.stop()

# ---------------- LOAD PDF & BUILD FULL TEXT ----------------
@st.cache_resource
def load_pdf_text(path: str) -> str:
    loader = PyPDFLoader(path)
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])
    return full_text

full_text = load_pdf_text(PDF_PATH)

# ---------------- EXTRACT SECTIONS (by numeric headings "N. Title") ----------------
def extract_sections(text: str) -> dict:
    """
    Returns dict: {normalized_title: {"title": original_title, "content": text}}
    Splits on lines like: '\n2. Working Hours Policy\n'
    """
    # find header matches
    header_pattern = re.compile(r"\n\s*(\d+)\.\s*([A-Za-z0-9 &()/-]+)\s*\n")
    matches = list(header_pattern.finditer(text))

    sections = {}
    if not matches:
        # fallback: whole document as a single section
        sections["full_document"] = {"title": "Document", "content": text}
        return sections

    for i, m in enumerate(matches):
        start = m.end()
        title_num = m.group(1).strip()
        title_raw = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Normalize key: keep only letters/numbers and spaces, lowercased
        key = re.sub(r"[^a-z0-9 ]+", "", title_raw.lower()).strip()
        sections[key] = {"title": title_raw, "content": content}

    return sections

sections = extract_sections(full_text)

# ---------------- CLEAN SECTION TEXT (remove list indices but preserve numbers in sentences) ----------------
def clean_section_text(text: str) -> str:
    # Remove bullets like '1 • ' or '1.' at the start of lines, but not numbers inside sentences
    text = re.sub(r"(?m)^\s*\d+\s*[•\.\-\)]\s*", "", text)

    # Remove stray headings like "Leave Policy" if they appear inline (we will display the heading separately)
    # but avoid removing the words 'leave' etc. inside sentences.
    text = re.sub(r"(?mi)^\s*(leave|work from home|working hours|code of conduct|performance management|training and development|insurance and health benefits|it and security|grievance redressal|disciplinary action|policy amendments)\s*\n", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.strip()
    return text

# ---------------- SPLIT INTO MEANINGFUL LINES / SENTENCES ----------------
def split_to_statements(text: str) -> list:
    # Split on sentence endings while keeping abbreviations safe (simple approach)
    # First split by newlines (many policy items are on separate lines)
    lines = []
    for part in text.split("\n"):
        part = part.strip()
        if not part:
            continue
        # If the line contains multiple sentences, split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', part)
        for s in sentences:
            s = s.strip()
            if len(s) > 6:
                lines.append(s)
    return lines

# ---------------- SECTION KEYWORD MAPPING ----------------
# Map user intents/keywords to normalized section keys present in the PDF
def build_section_aliases(sections_dict: dict) -> dict:
    # Create a reverse map of available section keys -> title
    aliases = {}
    for key, meta in sections_dict.items():
        aliases[key] = [key]  # default alias is the normalized key itself
        # also add some common phrases based on the title
        t = meta["title"].lower()
        if "leave" in t:
            aliases[key].extend(["leave", "leave policy", "paid leave", "casual leave", "sick leave"])
        if "work from home" in t or "wfh" in t:
            aliases[key].extend(["work from home", "wfh", "remote work"])
        if "working hours" in t:
            aliases[key].extend(["working hours", "attendance", "office hours"])
        if "code of conduct" in t:
            aliases[key].extend(["code of conduct", "conduct", "behavior", "harassment"])
        if "performance" in t:
            aliases[key].extend(["performance", "performance management", "kpi", "kpis"])
        if "training" in t:
            aliases[key].extend(["training", "training and development", "learning", "development"])
        if "insurance" in t or "health" in t:
            aliases[key].extend(["insurance", "medical insurance", "health insurance", "benefits"])
        if "it" in t or "security" in t:
            aliases[key].extend(["it", "security", "it security", "data security", "password"])
        if "grievance" in t:
            aliases[key].extend(["grievance", "complaint", "grievance redressal"])
        if "disciplinary" in t:
            aliases[key].extend(["disciplinary", "discipline", "disciplinary action"])
        if "policy amend" in t or "amend" in t:
            aliases[key].extend(["policy amendment", "amendment", "policy changes"])
    # normalize strings
    for k, vals in list(aliases.items()):
        aliases[k] = list({v.lower() for v in vals})
    return aliases

section_aliases = build_section_aliases(sections)

# ---------------- DETECT REQUESTED SECTIONS ----------------
def detect_requested_sections(question: str, aliases: dict) -> list:
    q = question.lower()
    requested = []
    # If user asks for multiple policies joined by 'and', split and detect separately
    # but simpler: check all aliases; include section if any alias keyword appears in q
    for key, keywords in aliases.items():
        for kw in keywords:
            if kw and kw in q:
                requested.append(key)
                break
    # If none found, attempt a looser check by word overlap
    if not requested:
        q_words = set(re.findall(r"[a-z]{3,}", q))
        for key, meta in sections.items():
            header_words = set(re.findall(r"[a-z]{3,}", meta["title"].lower()))
            if q_words & header_words:
                requested.append(key)
    # unique & preserve order
    seen = set()
    out = []
    for k in requested:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

# ---------------- GREETING & POLITE FALLBACK ----------------
def is_greeting(text: str) -> bool:
    return text.strip().lower() in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

# ---------------- UI ----------------
st.write("Enter a question about the HR policies (e.g., 'What is the leave policy?', 'What is the WFH policy?', 'What is performance management?')")

question = st.text_input("Enter your question")

if question:
    # greeting
    if is_greeting(question):
        st.success("Hello 👋 I’m your HR Policy Assistant. Ask me about leave, WFH, working hours, IT & security, insurance, performance, training, and more.")
    else:
        requested = detect_requested_sections(question, section_aliases)

        if not requested:
            st.warning("I checked the HR policy document, but this information is not mentioned. Please contact the HR team for further clarification.")
        else:
            for key in requested:
                meta = sections.get(key)
                if not meta:
                    st.warning(f"Section for '{key}' not found.")
                    continue

                # Show the title as the section heading
                st.markdown(f"### {meta['title']}")

                # Clean and split content into statements
                cleaned = clean_section_text(meta["content"])
                statements = split_to_statements(cleaned)

                if not statements:
                    st.info("This section exists but no extractable statements were found.")
                    continue

                # Display each statement as a plain sentence (no extra internal titles)
                for s in statements:
                    st.markdown(s)

