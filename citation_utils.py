"""
Lightweight citation metadata extraction and formatting.

Note: Automatic extraction from raw PDF text is heuristic (first line as
title, a 4-digit number as year, etc.) - it won't be perfect for every
paper's layout. The UI lets the user correct it before saving, which is
the honest way to handle this rather than pretending it's always accurate.
"""

import re


JUNK_LINE_PATTERNS = [
    r"^\d+$",                          # bare page numbers
    r"^www\.",                         # urls
    r"^http",
    r"^doi[:.]",                       # DOI lines
    r"^issn",
    r"^copyright",
    r"^©",
    r"^arxiv",
    r"^\[.*\]$",                       # bracketed headers like [DRAFT]
]


def _is_junk_line(line: str) -> bool:
    lower = line.lower().strip()
    if len(lower) < 4:
        return True
    return any(re.match(pattern, lower) for pattern in JUNK_LINE_PATTERNS)


def guess_metadata(first_page_text: str, fallback_filename: str) -> dict:
    """
    Best-effort guess at title, authors, and year from page 1 text.
    Skips obvious junk lines (page numbers, URLs, DOIs, copyright notices)
    before picking the title/author candidates - real papers often have
    1-2 header lines before the actual title.

    This is still a heuristic, not real parsing - always confirm/correct
    in the UI before indexing.
    """
    raw_lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
    clean_lines = [l for l in raw_lines if not _is_junk_line(l)]

    title = clean_lines[0] if clean_lines else fallback_filename
    authors = clean_lines[1] if len(clean_lines) > 1 else "Unknown Author"

    # Sanity check: titles are rarely a single word or extremely long
    if len(title.split()) < 2:
        title = fallback_filename

    year_match = re.search(r"(19|20)\d{2}", first_page_text)
    year = year_match.group(0) if year_match else "n.d."

    return {"title": title, "authors": authors, "year": year}


def first_author_surname(authors: str) -> str:
    """Extracts a surname to use for citation keys, e.g. 'Smith, J., Doe, A.' -> 'Smith'."""
    if not authors or authors == "Unknown Author":
        return "Unknown"
    first = authors.split(",")[0].strip()
    # handle "J. Smith" style too - take the last word as surname
    return first.split(" ")[-1]


def format_apa(meta: dict) -> str:
    """Very simplified APA-style reference. Good enough for a draft, not a final submission."""
    return f"{meta['authors']} ({meta['year']}). {meta['title']}."


def format_bibtex(meta: dict, paper_id: str) -> str:
    """Generates a BibTeX entry."""
    surname = first_author_surname(meta["authors"])
    key = f"{surname}{meta['year']}".replace(" ", "")
    return (
        f"@article{{{key},\n"
        f"  title   = {{{meta['title']}}},\n"
        f"  author  = {{{meta['authors']}}},\n"
        f"  year    = {{{meta['year']}}},\n"
        f"  note    = {{paper_id: {paper_id}}}\n"
        f"}}"
    )


def in_text_citation(meta: dict) -> str:
    """e.g. (Smith, 2023) - used when the LLM cites sources in generated text."""
    surname = first_author_surname(meta["authors"])
    return f"({surname}, {meta['year']})"
