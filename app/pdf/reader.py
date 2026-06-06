import fitz
from cleaner import cleaner_text
from pathlib import Path

def read_pdf(file_path) -> dict:
    if not file_path.endswith(".pdf"):
        raise ValueError("File must be a PDF")
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File {file_path} does not exist")
    if file_path is None or file_path.strip() == "":
        raise ValueError("File path cannot be empty")

    pages = []
    filename = Path(file_path).name 
    with fitz.open(file_path) as doc:
        for page_number, page in enumerate(doc, 1):
            pages.append({
                "page": page_number,
                "content": cleaner_text(page.get_text())
            })
    return {
        "filename": filename,
        "total_pages": len(pages),
        "pages": pages
    }
