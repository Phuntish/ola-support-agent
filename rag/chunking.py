"""Task 3 (part 1) - load the knowledge base and cut it two different ways.

Strategy A: fixed-size character windows with a fixed overlap.
Strategy B: sentence groups with a one-sentence overlap.

Both strategies prepend the document title to the text that actually gets
embedded. The KB documents are short policy paragraphs and several of them use
the same vocabulary ("ticket", "customer", "agent"), so the title is the
cheapest way to keep a chunk anchored to its own topic. Both collections get the
same treatment, so the Task 5 comparison stays fair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "kb"

FIXED_CHUNK_CHARS = 400
FIXED_OVERLAP_CHARS = 80

SENTENCES_PER_CHUNK = 2
SENTENCE_OVERLAP = 1

FIXED_STRATEGY = "fixed"
SENTENCE_STRATEGY = "sentence"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    text: str

    @property
    def embed_text(self) -> str:
        return self.text


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    """Read every kb/*.md file. First line is '# Title', the rest is the body."""
    documents: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        documents.append(Document(doc_id=path.stem, title=title, text=body))
    if not documents:
        raise FileNotFoundError(f"no knowledge-base documents found in {kb_dir}")
    return documents


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def fixed_size_chunks(
    text: str,
    chunk_chars: int = FIXED_CHUNK_CHARS,
    overlap_chars: int = FIXED_OVERLAP_CHARS,
) -> list[str]:
    """Slide a fixed-width window over the text, snapping the cut to a word boundary."""
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap must be smaller than the chunk size")

    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_chars - overlap_chars

    while start < len(text):
        end = start + chunk_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Do not cut a word in half: walk back to the last space in the window.
        cut = text.rfind(" ", start, end)
        if cut <= start:
            cut = end
        chunks.append(text[start:cut].strip())
        start += step

    return [c for c in chunks if c]


def sentence_chunks(
    text: str,
    per_chunk: int = SENTENCES_PER_CHUNK,
    overlap: int = SENTENCE_OVERLAP,
) -> list[str]:
    """Group whole sentences, repeating the last `overlap` sentences in the next chunk."""
    if overlap >= per_chunk:
        raise ValueError("sentence overlap must be smaller than the group size")

    sentences = split_sentences(text)
    if len(sentences) <= per_chunk:
        return [" ".join(sentences)] if sentences else []

    chunks: list[str] = []
    step = per_chunk - overlap
    for start in range(0, len(sentences), step):
        group = sentences[start : start + per_chunk]
        if not group:
            break
        chunks.append(" ".join(group))
        if start + per_chunk >= len(sentences):
            break

    return chunks


def chunk_documents(documents: list[Document], strategy: str) -> list[Chunk]:
    """Apply one strategy across the whole corpus and hand back labelled chunks."""
    if strategy == FIXED_STRATEGY:
        splitter = fixed_size_chunks
    elif strategy == SENTENCE_STRATEGY:
        splitter = sentence_chunks
    else:
        raise ValueError(f"unknown chunking strategy: {strategy!r}")

    chunks: list[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(splitter(doc.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{strategy}:{doc.doc_id}:{i}",
                    doc_id=doc.doc_id,
                    strategy=strategy,
                    # The title rides along so a chunk stays attached to its topic.
                    text=f"{doc.title}. {piece}",
                )
            )
    return chunks


def summarise(documents: list[Document]) -> None:
    fixed = chunk_documents(documents, FIXED_STRATEGY)
    sentence = chunk_documents(documents, SENTENCE_STRATEGY)

    print("=" * 70)
    print("TASK 3 - CHUNKING")
    print("=" * 70)
    print(f"documents loaded : {len(documents)}")
    print(f"fixed-size       : {FIXED_CHUNK_CHARS} chars, {FIXED_OVERLAP_CHARS} char overlap"
          f" -> {len(fixed)} chunks")
    print(f"sentence-based   : {SENTENCES_PER_CHUNK} sentences, {SENTENCE_OVERLAP} sentence overlap"
          f" -> {len(sentence)} chunks")
    print()

    print(f"{'document':<34} {'sentences':>9} {'fixed':>6} {'sent':>6}")
    print("-" * 58)
    for doc in documents:
        n_fixed = sum(1 for c in fixed if c.doc_id == doc.doc_id)
        n_sent = sum(1 for c in sentence if c.doc_id == doc.doc_id)
        print(f"{doc.doc_id:<34} {len(split_sentences(doc.text)):>9} {n_fixed:>6} {n_sent:>6}")
    print()

    for label, chunks in ((FIXED_STRATEGY, fixed), (SENTENCE_STRATEGY, sentence)):
        lengths = [len(c.text) for c in chunks]
        print(f"{label:<9} chunk length: min={min(lengths)} mean={sum(lengths) // len(lengths)} max={max(lengths)}")
    print()

    print("Sample - escalation-matrix, fixed-size chunk 0:")
    print(f"  {next(c.text for c in fixed if c.doc_id == 'escalation-matrix')[:200]}...")
    print()
    print("Sample - escalation-matrix, sentence chunk 0:")
    print(f"  {next(c.text for c in sentence if c.doc_id == 'escalation-matrix')[:200]}...")


if __name__ == "__main__":
    summarise(load_documents())
