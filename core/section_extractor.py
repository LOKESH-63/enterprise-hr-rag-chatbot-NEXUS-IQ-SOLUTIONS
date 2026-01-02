import re

def extract_sections(text: str) -> dict:
    pattern = re.compile(r"\n\s*(\d+)\.\s*([A-Za-z0-9 &()/-]+)\s*\n")
    matches = list(pattern.finditer(text))

    sections = {}

    if not matches:
        sections["document"] = {
            "title": "HR Policy Document",
            "content": text
        }
        return sections

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        title = m.group(2).strip()
        content = text[start:end].strip()

        key = re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()
        sections[key] = {
            "title": title,
            "content": content
        }

    return sections
