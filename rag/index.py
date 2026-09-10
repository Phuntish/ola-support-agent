"""Task 3 (part 2) - embed both chunk sets and index each one in its own Chroma collection.

Embeddings come from a local SentenceTransformers model, so no API key and no
network call is involved once the model has been cached on disk. Each chunking
strategy gets its own collection so Task 5 can score them independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings

from rag.chunking import (
    FIXED_STRATEGY,
    SENTENCE_STRATEGY,
    Chunk,
    Document,
    chunk_documents,
    load_documents,
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"

COLLECTION_NAMES = {
    FIXED_STRATEGY: "ola_kb_fixed",
    SENTENCE_STRATEGY: "ola_kb_sentence",
}

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    similarity: float


@lru_cache(maxsize=1)
def get_embedder():
    """Load the sentence encoder once per process; it is the slow part of startup."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    """Unit-normalised vectors, so Chroma's cosine distance is 1 - cosine similarity."""
    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(strategy: str) -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=COLLECTION_NAMES[strategy],
        metadata={"hnsw:space": "cosine", "strategy": strategy},
    )


def upsert_chunks(strategy: str, chunks: list[Chunk]) -> chromadb.Collection:
    collection = get_collection(strategy)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed([c.embed_text for c in chunks]),
        metadatas=[{"doc_id": c.doc_id, "strategy": c.strategy} for c in chunks],
    )
    return collection


def build_indexes(documents: list[Document] | None = None, fresh: bool = True) -> dict[str, int]:
    """(Re)build both collections from the knowledge base. Returns chunk counts.

    `fresh` drops the collections first. upsert() alone would leave behind any
    document added at runtime through POST /add-document, so a later evaluation
    run would silently be scoring against a different knowledge base than the one
    in kb/. Dropping first keeps every run reproducible from the files on disk.
    """
    docs = load_documents() if documents is None else documents
    client = get_client()

    counts: dict[str, int] = {}
    for strategy in (FIXED_STRATEGY, SENTENCE_STRATEGY):
        if fresh:
            try:
                client.delete_collection(COLLECTION_NAMES[strategy])
            except Exception:
                pass  # nothing indexed yet on a first run
        chunks = chunk_documents(docs, strategy)
        upsert_chunks(strategy, chunks)
        counts[strategy] = len(chunks)
    return counts


def index_document(document: Document) -> dict[str, int]:
    """Add or replace one document in both collections (used by POST /add-document)."""
    counts: dict[str, int] = {}
    for strategy in (FIXED_STRATEGY, SENTENCE_STRATEGY):
        chunks = chunk_documents([document], strategy)
        upsert_chunks(strategy, chunks)
        counts[strategy] = len(chunks)
    return counts


def retrieve(query: str, strategy: str = SENTENCE_STRATEGY, top_k: int = DEFAULT_TOP_K) -> list[Hit]:
    collection = get_collection(strategy)
    result = collection.query(
        query_embeddings=embed([query]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_id=str(meta["doc_id"]),
                text=text,
                # Chroma reports cosine distance; invert it back to similarity.
                similarity=round(1.0 - float(distance), 4),
            )
        )
    return hits


def main() -> None:
    documents = load_documents()
    counts = build_indexes(documents)

    print("=" * 70)
    print("TASK 3 - EMBEDDING AND INDEXING")
    print("=" * 70)
    print(f"embedding model : {EMBED_MODEL_NAME}")
    print(f"chroma path     : {CHROMA_DIR}")
    print(f"documents       : {len(documents)}")
    for strategy in (FIXED_STRATEGY, SENTENCE_STRATEGY):
        collection = get_collection(strategy)
        print(f"collection {COLLECTION_NAMES[strategy]:<18} strategy={strategy:<9}"
              f" chunks upserted={counts[strategy]:<4} count()={collection.count()}")
    print()

    sample_query = "how long do I have to wait for a reply on a P1 ticket"
    print(f"Sanity check on both collections - query: {sample_query!r}")
    for strategy in (FIXED_STRATEGY, SENTENCE_STRATEGY):
        print(f"\n  [{strategy}]")
        for rank, hit in enumerate(retrieve(sample_query, strategy, top_k=3), start=1):
            print(f"    {rank}. sim={hit.similarity:.4f}  doc={hit.doc_id}")
            print(f"       {hit.text[:110]}...")


if __name__ == "__main__":
    main()
