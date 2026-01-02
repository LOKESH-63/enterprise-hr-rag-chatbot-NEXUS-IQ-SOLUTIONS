import re

def clean_section_text(text: str) -> str:
    text = re.sub(r"(?m)^\s*\d+\s*[•\.\-\)]\s*", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


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
