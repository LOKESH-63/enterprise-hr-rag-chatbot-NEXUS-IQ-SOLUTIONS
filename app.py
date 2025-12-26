import streamlit as st
import re
import os
from langchain_community.document_loaders import PyPDFLoader

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="NEXUS IQ HR Chatbot", page_icon="🏢")
st.markdown("## 🏢 NEXUS IQ SOLUTIONS")
st.caption("HR Policy Assistant • Accurate • Polite")
st.markdown("### 💬 Ask an HR policy question")

# ======================================================
# LOAD PDF
# ======================================================
PDF_FILE = "Sample_HR_Policy_Document.pdf"

if not os.path.exists(PDF_FILE):
    st.error("HR Policy PDF not found.")
    st.stop()

loader = PyPDFLoader(PDF_FILE)
pages = loader.load()
full_text = "\n".join([p.page_content for p in pages])

# ======================================================
# SPLIT PDF INTO POLICY SECTIONS
# ======================================================
def extract_sections(text):
    pattern = r"\n\d+\.\s+[A-Za-z ()]+\n"
    matches = re.split(pattern, text)
    headers = re.findall(pattern, text)

    sections = {}
    for i, header in enumerate(headers):
        title = re.sub(r"\d+\.|\n", "", header).strip()
        content = matches[i + 1]
        sections[title.lower()] = content.strip()

    return sections

sections = extract_sections(full_text)

# ======================================================
# CLEAN BULLET POINTS
# ======================================================
def clean_points(section_text):
    lines = section_text.split("\n")
    points = []

    for line in lines:
        line = re.sub(r"^\s*\d+\s*[•.-]?\s*", "", line)
        line = line.strip()
        if len(line) > 5:
            points.append(line)

    return points

# ======================================================
# GREETING DETECTOR
# ======================================================
def is_greeting(text):
    greetings = [
        "hi", "hello", "hey",
        "good morning", "good afternoon", "good evening"
    ]
    return text.lower().strip() in greetings

# ======================================================
# DETECT REQUESTED POLICIES
# ======================================================
def detect_requested_policies(question):
    q = question.lower()
    requested = []

    if "leave" in q:
        requested.append("leave policy")

    if "work from home" in q or "wfh" in q:
        requested.append("work from home (wfh) policy")

    if "working hours" in q:
        requested.append("working hours policy")

    if "security" in q or "it" in q:
        requested.append("it and security policy")

    if "code of conduct" in q:
        requested.append("code of conduct")

    return requested

# ======================================================
# UI
# ======================================================
question = st.text_input("Enter your question")

if question:

    # ---------- GREETING ----------
    if is_greeting(question):
        st.success(
            "Hello 👋\n\n"
            "I’m your HR Policy Assistant.\n\n"
            "You can ask me questions about company HR policies such as leave, work from home, "
            "IT and security, working hours, and code of conduct."
        )

    else:
        policies = detect_requested_policies(question)

        # ---------- NO POLICY FOUND ----------
        if not policies:
            st.warning(
                "I checked the HR policy document, but this information is not mentioned.\n\n"
                "Please contact the HR team for further clarification."
            )

        # ---------- SHOW POLICIES ----------
        else:
            for policy in policies:
                if policy in sections:
                    st.markdown(f"### {policy.title()}")
                    points = clean_points(sections[policy])
                    for p in points:
                        st.markdown(f"• {p}")
                else:
                    st.warning(
                        f"I checked the HR policy document, but details about **{policy.title()}** "
                        "are not available."
                    )
