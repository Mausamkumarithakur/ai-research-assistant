"""
AI Research Assistant - Literature Review & Citation Helper

Runs fully locally via Ollama + ChromaDB. No API keys, no cost.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3
    3. ollama pull nomic-embed-text
    4. pip install -r requirements.txt
    5. streamlit run app.py
"""

import json
import os
import uuid

import requests
import streamlit as st

from pdf_utils import extract_text_from_pdf, extract_first_page_text, chunk_text
from citation_utils import guess_metadata, format_apa, format_bibtex, in_text_citation
from vector_store import add_paper_chunks, query_chunks, delete_paper, paper_exists

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
PAPERS_FILE = "papers.json"
UPLOAD_DIR = "uploaded_pdfs"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- Ollama call ----------

def call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Could not connect to Ollama. Run 'ollama serve' in a terminal, "
            f"and make sure you've pulled both '{MODEL_NAME}' and 'nomic-embed-text'."
        )
    except Exception as e:
        return f"ERROR: {e}"


# ---------- Paper registry (metadata for all indexed papers) ----------

def load_papers() -> dict:
    if os.path.exists(PAPERS_FILE):
        with open(PAPERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_papers(papers: dict):
    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)


# ---------- UI ----------

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="centered")
st.title("📚 AI Research Assistant")
st.caption("Local RAG-based literature review & citation helper - runs via Ollama, no API keys.")

papers = load_papers()

tab_upload, tab_ask, tab_summarize, tab_review, tab_citations = st.tabs(
    ["📤 Upload Papers", "❓ Ask Across Papers", "📝 Summarize a Paper",
     "📖 Literature Review", "🔖 Citations"]
)

# ---- Tab 1: Upload & Index ----
with tab_upload:
    st.subheader("Upload and index a research paper")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        first_page = extract_first_page_text(save_path)
        guessed = guess_metadata(first_page, uploaded_file.name)

        st.write("Confirm / correct the citation info before indexing:")
        title = st.text_input("Title", value=guessed["title"])
        authors = st.text_input("Authors", value=guessed["authors"])
        year = st.text_input("Year", value=guessed["year"])

        if st.button("Index this paper", type="primary"):
            paper_id = str(uuid.uuid4())[:8]
            try:
                with st.spinner("Extracting text and generating embeddings... (can take a minute)"):
                    full_text = extract_text_from_pdf(save_path)
                    chunks = chunk_text(full_text)
                    if not chunks:
                        st.error("No usable text chunks were produced from this PDF.")
                        st.stop()

                    metadata = {"title": title, "authors": authors, "year": year}
                    add_paper_chunks(paper_id, chunks, metadata)

                    papers[paper_id] = {**metadata, "filename": uploaded_file.name, "chunk_count": len(chunks)}
                    save_papers(papers)

                st.success(f"Indexed '{title}' ({len(chunks)} chunks).")
            except ValueError as e:
                st.error(str(e))
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach Ollama while generating embeddings. "
                    "Run 'python check_setup.py' in a terminal to diagnose."
                )
            except Exception as e:
                st.error(f"Unexpected error while indexing: {e}")

    st.divider()
    st.subheader("Indexed papers")
    if not papers:
        st.info("No papers indexed yet.")
    else:
        for pid, meta in papers.items():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"**{meta['title']}** — {meta['authors']} ({meta['year']})")
            with col2:
                if st.button("Delete", key=f"del_{pid}"):
                    delete_paper(pid)
                    del papers[pid]
                    save_papers(papers)
                    st.rerun()

# ---- Tab 2: Ask across papers (RAG Q&A) ----
with tab_ask:
    st.subheader("Ask a question across all indexed papers")

    if not papers:
        st.info("Index at least one paper first.")
    else:
        question = st.text_area("Your question", placeholder="e.g. What evaluation metrics do these papers use?")

        if st.button("Ask", type="primary"):
            if not question.strip():
                st.warning("Enter a question.")
            else:
                with st.spinner("Retrieving relevant passages and generating answer..."):
                    results = query_chunks(question, n_results=6)
                    contexts = results["documents"][0]
                    metas = results["metadatas"][0]

                    context_block = ""
                    for i, (chunk, meta) in enumerate(zip(contexts, metas)):
                        cite = in_text_citation(meta)
                        context_block += f"\n[Source {cite}]: {chunk}\n"

                    prompt = f"""You are a research assistant helping answer a question using only the excerpts below.
Cite sources using the citation tags already given in parentheses, e.g. (Smith, 2023).
If the excerpts don't contain enough information, say so honestly.

Excerpts:
{context_block}

Question: {question}

Answer (with in-text citations):"""

                    answer = call_ollama(prompt)

                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Show retrieved source excerpts"):
                    for chunk, meta in zip(contexts, metas):
                        st.markdown(f"**{in_text_citation(meta)}** - {meta['title']}")
                        st.caption(chunk[:400] + "...")
                        st.divider()

# ---- Tab 3: Summarize one paper ----
with tab_summarize:
    st.subheader("Summarize a single paper")

    if not papers:
        st.info("Index at least one paper first.")
    else:
        options = {f"{m['title']} ({m['year']})": pid for pid, m in papers.items()}
        selected_label = st.selectbox("Choose a paper", list(options.keys()))
        selected_id = options[selected_label]

        if st.button("Generate summary", type="primary"):
            with st.spinner("Retrieving content and summarizing..."):
                results = query_chunks(
                    "main objective, method, findings, and limitations of this paper",
                    n_results=8,
                    paper_id=selected_id,
                )
                contexts = "\n".join(results["documents"][0])

                prompt = f"""Summarize the following research paper excerpts into a structured summary with these sections:
- Objective
- Method
- Key Findings
- Limitations

Excerpts:
{contexts}

Structured Summary:"""

                summary = call_ollama(prompt)
            st.markdown("### Summary")
            st.write(summary)

# ---- Tab 4: Literature review generator ----
with tab_review:
    st.subheader("Generate a literature review paragraph")

    if len(papers) < 2:
        st.info("Index at least 2 papers to generate a meaningful literature review.")
    else:
        topic = st.text_input("Research theme / question", placeholder="e.g. approaches to reducing hallucination in LLMs")

        if st.button("Generate Literature Review", type="primary"):
            if not topic.strip():
                st.warning("Enter a research theme.")
            else:
                with st.spinner("Synthesizing across papers..."):
                    results = query_chunks(topic, n_results=10)
                    contexts = results["documents"][0]
                    metas = results["metadatas"][0]

                    context_block = ""
                    for chunk, meta in zip(contexts, metas):
                        cite = in_text_citation(meta)
                        context_block += f"\n[{cite}]: {chunk}\n"

                    prompt = f"""You are writing an academic literature review paragraph on the theme: "{topic}"

Using ONLY the excerpts below, synthesize a coherent literature review paragraph (250-400 words) that:
- Compares and contrasts what different papers say
- Uses in-text citations in the format already given, e.g. (Smith, 2023)
- Reads like an academic literature review, not a list of summaries
- Notes agreements, disagreements, or gaps between sources if present

Excerpts:
{context_block}

Literature Review Paragraph:"""

                    review = call_ollama(prompt)
            st.markdown("### Draft Literature Review")
            st.write(review)
            st.caption("⚠️ Always verify claims and citations against the original papers before submitting.")

# ---- Tab 5: Citations ----
with tab_citations:
    st.subheader("Citation list")

    if not papers:
        st.info("No papers indexed yet.")
    else:
        for pid, meta in papers.items():
            st.markdown(f"**{meta['title']}**")
            st.code(format_apa(meta), language=None)
            st.code(format_bibtex(meta, pid), language="bibtex")
            st.divider()
