"""
Vector store + RAG engine.

Uses ChromaDB (free, local, persistent) for storage/retrieval, and Ollama's
embedding model to turn text into vectors. This is the same pattern used
in production RAG systems - just swapped to free local components.
"""

import requests
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "research_papers"


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Wraps Ollama's embedding endpoint so ChromaDB can call it directly."""

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(),
    )


def add_paper_chunks(paper_id: str, chunks: list[str], metadata: dict):
    """Adds all chunks of one paper to the vector store, tagged with its metadata."""
    collection = get_collection()
    ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"paper_id": paper_id, "title": metadata["title"],
         "authors": metadata["authors"], "year": metadata["year"]}
        for _ in chunks
    ]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)


def query_chunks(query_text: str, n_results: int = 6, paper_id: str = None) -> dict:
    """
    Retrieves the most relevant chunks for a query, optionally restricted
    to a single paper (used for per-paper summarization).
    """
    collection = get_collection()
    where_filter = {"paper_id": paper_id} if paper_id else None
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter,
    )


def paper_exists(paper_id: str) -> bool:
    collection = get_collection()
    existing = collection.get(where={"paper_id": paper_id}, limit=1)
    return len(existing["ids"]) > 0


def delete_paper(paper_id: str):
    collection = get_collection()
    collection.delete(where={"paper_id": paper_id})
