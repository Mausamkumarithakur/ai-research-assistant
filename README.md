# 📚 AI Research Assistant

A local, RAG-based research assistant that helps you read, summarize, and write literature reviews from your own research papers — with automatic citation generation. Runs **entirely on your machine** using [Ollama](https://ollama.com), no API keys, no cost, no internet required after setup.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(Llama%203)-black)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)

---



## 📌 Features

- **Upload & index PDFs** — extracts text, chunks it, and embeds it into a local vector database
- **Ask questions across all papers** — retrieval-augmented Q&A with in-text citations, e.g. `(Smith, 2023)`
- **Per-paper summaries** — structured output: Objective, Method, Findings, Limitations
- **Literature review generator** — synthesizes a properly cited literature-review paragraph across multiple papers on a given theme
- **Automatic citations** — generates APA and BibTeX entries for every indexed paper
- **Self-check script** — diagnoses your Ollama setup before you even open the app

---

## 🧠 How it works (architecture)

This is a real **RAG (Retrieval-Augmented Generation)** pipeline:

```
PDF → text extraction → sentence-aware chunking → embeddings (nomic-embed-text)
    → stored in ChromaDB (local vector database)
    → on query: retrieve top-matching chunks → feed to Llama 3 → cited answer
```

Same architecture used in production RAG tools — just running on free, local components instead of paid cloud APIs.

---

## 🗂️ Project Structure

```
ai_research_assistant/
├── app.py                # Streamlit UI - all 5 tabs/features
├── pdf_utils.py          # PDF text extraction + chunking
├── vector_store.py       # ChromaDB + Ollama embedding integration
├── citation_utils.py     # Metadata guessing + APA/BibTeX formatting
├── check_setup.py        # Verifies Ollama + models before running the app
├── requirements.txt
├── chroma_db/            # auto-created - stores indexed paper vectors
├── uploaded_pdfs/         # auto-created - stores uploaded PDF files
└── papers.json           # auto-created - metadata registry of indexed papers
```

---

## ⚙️ Setup

### 1. Install Ollama
Download from [ollama.com/download](https://ollama.com/download).

### 2. Pull the required models (one-time)
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Clone / download this project and install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify your setup
```bash
python check_setup.py
```
You should see all green checkmarks before continuing.

### 5. Run the app
```bash
streamlit run app.py
```
(If `streamlit` isn't recognized as a command, use `python -m streamlit run app.py` instead.)

The app opens automatically at `http://localhost:8501`.

---

## 🚀 Usage

1. **Upload Papers** — drop in a PDF, confirm/correct the guessed title/author/year, click *Index this paper*
2. **Ask Across Papers** — ask a question, get a cited answer pulled from your indexed papers
3. **Summarize a Paper** — pick one paper, get a structured summary
4. **Literature Review** — enter a research theme, get a synthesized, cited paragraph across multiple papers
5. **Citations** — copy ready-made APA and BibTeX entries for every paper you've indexed

---

## ⚠️ Known Limitations (honest, not hidden)

- **Citation extraction is heuristic**, not true metadata parsing — always confirm/correct title, authors, and year before indexing.
- **Scanned/image-only PDFs won't work** — text extraction requires a real text layer (no OCR built in).
- **Llama 3 (local, 8B)** is less capable than GPT-4/Claude-tier models — good for drafts and exploration, not a substitute for careful human review.
- **Always verify** any claim, citation, or summary against the original paper before using it in real academic work. Retrieval reduces hallucination but does not eliminate it.

---

## 🛠️ Tech Stack

- Python 3.10
- [Ollama](https://ollama.com) — local LLM (Llama 3) + embeddings (nomic-embed-text)
- [ChromaDB](https://www.trychroma.com) — local persistent vector database
- [pypdf](https://pypi.org/project/pypdf/) — PDF text extraction
- [Streamlit](https://streamlit.io) — UI

---

## 📄 License

Built for educational and personal research use.

---

## 🙋 Author

Built by [Mausamkumarithakur](https://github.com/Mausamkumarithakur)
