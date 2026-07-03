"""
PDF loading and text extraction utilities.
"""
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF, page by page.

    Returns a list of dicts: [{"page": 1, "text": "...", "source": "file.pdf"}, ...]
    Pages with no extractable text (e.g. scanned images) are skipped.
    """
    reader = PdfReader(pdf_path)
    filename = Path(pdf_path).name
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text, "source": filename})

    return pages


def load_pdfs_from_directory(directory: str) -> List[Dict]:
    """
    Load and extract text from every PDF in a directory.
    """
    all_pages = []
    pdf_files = sorted(Path(directory).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{directory}'.")

    for pdf_file in pdf_files:
        pages = extract_text_from_pdf(str(pdf_file))
        all_pages.extend(pages)

    return all_pages
