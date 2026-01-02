from langchain_community.document_loaders import PyPDFLoader

def load_pdf_text(path: str) -> str:
    loader = PyPDFLoader(path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)
