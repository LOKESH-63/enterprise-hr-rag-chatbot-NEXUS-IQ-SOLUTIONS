import re

def build_section_aliases(sections: dict) -> dict:
    aliases = {}

    for key, meta in sections.items():
        title = meta["title"].lower()
        words = [key]

        if "leave" in title:
            words += ["leave", "paid leave", "casual leave", "sick leave"]
        if "work from home" in title or "wfh" in title:
            words += ["work from home", "wfh", "remote"]
        if "working hours" in title:
            words += ["working hours", "office hours", "attendance"]
        if "insurance" in title or "health" in title:
            words += ["insurance", "medical", "health"]
        if "performance" in title:
            words += ["performance", "kpi"]
        if "training" in title:
            words += ["training", "learning"]
        if "security" in title or "it" in title:
            words += ["it", "security", "password"]
        if "conduct" in title:
            words += ["conduct", "behavior", "harassment"]
        if "disciplinary" in title:
            words += ["disciplinary", "discipline"]

        aliases[key] = list(set(words))

    return aliases


def detect_requested_sections(question: str, sections: dict, aliases: dict) -> list:
    q = question.lower()
    found = []

    for key, kws in aliases.items():
        if any(k in q for k in kws):
            found.append(key)

    if not found:
        q_words = set(re.findall(r"[a-z]{3,}", q))
        for key, meta in sections.items():
            title_words = set(re.findall(r"[a-z]{3,}", meta["title"].lower()))
            if q_words & title_words:
                found.append(key)

    return list(dict.fromkeys(found))
