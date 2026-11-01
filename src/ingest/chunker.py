"""Split loaded Documents into chunks and write data/chunks/chunks.jsonl.

Strategies (config `chunking.strategy`):
- section (default): one chunk per section, so one chunk ≈ one fact family. A section
  longer than `chunk_size` is split on line boundaries, so "Label: Value" lines stay whole.
- recursive: split the whole document on paragraphs, then lines, sentences, words,
  ignoring section boundaries; `section` is the heading where the chunk starts.

With `fact_chunks`, a section that is a "Label: Value" list (e.g. a scheme's Key facts)
is also indexed one fact per chunk, so "Expense ratio: 1.02%" isn't outscored by
longer passages that merely mention expense ratios.

Every chunk text starts with "{title} — {section}" so four near-identical
"Exit load" blocks from different schemes stay distinguishable in search.
`chunk_size` / `chunk_overlap` apply to the body, not the prefix.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from src.config import ChunkingConfig, Settings, get_settings
from src.ingest.documents import load_documents
from src.models import Chunk, Document

SECTION_SEPARATORS = ["\n", ". ", " ", ""]
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
FACT_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9 ()/&.'-]{1,40}:\s+\S")


def chunk_documents(documents: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunking.chunk_overlap must be smaller than chunking.chunk_size")
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, cfg))
    return chunks


def chunk_document(document: Document, cfg: ChunkingConfig) -> list[Chunk]:
    if cfg.strategy == "section":
        pieces = _section_pieces(document, cfg)
    elif cfg.strategy == "recursive":
        pieces = _recursive_pieces(document, cfg)
    else:
        raise ValueError(f"Unknown chunking.strategy: {cfg.strategy!r} (use section or recursive)")

    return [
        Chunk(
            chunk_id=f"{document.source_id}::{index:03d}",
            source_id=document.source_id,
            source_url=document.url,
            section=section,
            text=f"{document.title} — {section}\n{body}",
            chunk_index=index,
        )
        for index, (section, body) in enumerate(pieces)
    ]


def _section_pieces(document: Document, cfg: ChunkingConfig) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    for section in document.sections:
        for body in split_text(section.text, cfg.chunk_size, cfg.chunk_overlap, SECTION_SEPARATORS):
            pieces.append((section.heading, body))
        if cfg.fact_chunks and is_fact_list(section.text):
            pieces.extend((section.heading, line) for line in section.text.split("\n") if FACT_LINE.match(line))
    return pieces


def is_fact_list(text: str) -> bool:
    """At least 3 lines, and at least 80% of them look like "Label: Value"."""
    lines = [line for line in text.split("\n") if line.strip()]
    return len(lines) >= 3 and sum(bool(FACT_LINE.match(line)) for line in lines) >= 0.8 * len(lines)


def _recursive_pieces(document: Document, cfg: ChunkingConfig) -> list[tuple[str, str]]:
    starts: list[tuple[int, str]] = []
    blocks: list[str] = []
    offset = 0
    for section in document.sections:
        block = f"{section.heading}\n{section.text}"
        starts.append((offset, section.heading))
        blocks.append(block)
        offset += len(block) + 2
    full_text = "\n\n".join(blocks)

    pieces: list[tuple[str, str]] = []
    cursor = 0
    for body in split_text(full_text, cfg.chunk_size, cfg.chunk_overlap, RECURSIVE_SEPARATORS):
        position = full_text.find(body, cursor)
        if position >= 0:
            cursor = position + 1
        heading = next(
            (name for start, name in reversed(starts) if start <= max(position, 0)),
            starts[0][1] if starts else "Document",
        )
        pieces.append((heading, body))
    return pieces


def split_text(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    """Split text into pieces of at most `size` chars, preferring earlier separators.

    Consecutive pieces share up to `overlap` chars of whole sub-pieces (lines, sentences…).
    Every returned piece is a contiguous substring of `text`.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    separator = next((sep for sep in separators if sep and sep in text), "")
    remaining = separators[separators.index(separator) + 1 :] if separator in separators else []
    parts = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    window: list[str] = []

    def joined() -> str:
        return separator.join(window)

    def flush() -> None:
        piece = joined().strip()
        if piece:
            chunks.append(piece)

    for part in parts:
        if len(part) > size:
            flush()
            window.clear()
            chunks.extend(split_text(part, size, overlap, remaining))
            continue
        if window and len(joined()) + len(separator) + len(part) > size:
            flush()
            while window and (
                len(joined()) > overlap or len(joined()) + len(separator) + len(part) > size
            ):
                window.pop(0)
        window.append(part)
    flush()
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


def run_chunking(settings: Settings | None = None) -> list[Chunk]:
    """Load documents from data/raw/, chunk them, write chunks.jsonl, print a summary."""
    settings = settings or get_settings()
    cfg = settings.chunking
    chunks = chunk_documents(load_documents(settings=settings), cfg)
    path = settings.resolve_path(settings.paths.chunks_file)
    write_chunks(chunks, path)

    print(f"strategy={cfg.strategy} chunk_size={cfg.chunk_size} overlap={cfg.chunk_overlap}")
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.source_id] = counts.get(chunk.source_id, 0) + 1
    for source_id, count in counts.items():
        print(f"  {source_id:40}  chunks={count}")
    if chunks:
        lengths = [len(chunk.text) for chunk in chunks]
        print(
            f"total={len(chunks)}  chars min={min(lengths)} "
            f"median={int(statistics.median(lengths))} max={max(lengths)}"
        )
    print(f"wrote {path}")
    return chunks
