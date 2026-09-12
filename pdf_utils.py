"""
PDF text extraction and chunking utilities.
"""

import re
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file, page by page. Raises a clear error on failure."""
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF '{file_path}': {e}")

    full_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        full_text.append(text)

    combined = "\n".join(full_text).strip()
    if not combined:
        raise ValueError(
            f"No extractable text found in '{file_path}'. "
            "This usually means it's a scanned/image-only PDF and would need OCR "
            "(not covered by this tool)."
        )
    return combined


def extract_first_page_text(file_path: str) -> str:
    """Used for guessing title/authors/year from the first page."""
    try:
        reader = PdfReader(file_path)
        if len(reader.pages) == 0:
            return ""
        return reader.pages[0].extract_text() or ""
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks, breaking on sentence boundaries where
    possible instead of cutting mid-sentence. Falls back to a hard character
    cut only if a single sentence is longer than chunk_size.
    """
    text = " ".join(text.split()).strip()
    if not text:
        return []

    # Rough sentence split (good enough without pulling in a full NLP library)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # carry overlap from the end of the previous chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = f"{overlap_text} {sentence}".strip()

            # if a single sentence itself is too long, hard-split it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]
