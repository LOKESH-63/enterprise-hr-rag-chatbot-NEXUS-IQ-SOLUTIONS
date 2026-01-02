from core.pdf_loader import load_pdf_text
from core.section_extractor import extract_sections
from core.section_matcher import build_section_aliases, detect_requested_sections

text = load_pdf_text("data/Sample_HR_Policy_Document.pdf")
sections = extract_sections(text)
aliases = build_section_aliases(sections)

print(detect_requested_sections("How many sick leaves do I get?", sections, aliases))
