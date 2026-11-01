"""Read scraped files in data/raw/ back into cleaned Documents for chunking.

Chunking always starts from disk, never from the scraper's in-memory output,
so it can be re-run without re-fetching.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.config import Settings, Source, get_settings, load_sources
from src.ingest.extract import parse_sections_file
from src.models import Document, RawMeta, Section

ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")


class DocumentLoadError(Exception):
    def __init__(self, message: str, statuses: list[SourceStatus]):
        super().__init__(message)
        self.statuses = statuses


@dataclass
class SourceStatus:
    source_id: str
    required: bool
    status: str  # ok | error
    sections: int = 0
    chars: int = 0
    detail: str = ""


@dataclass
class DocumentLoadResult:
    documents: list[Document] = field(default_factory=list)
    statuses: list[SourceStatus] = field(default_factory=list)

    @property
    def required_failures(self) -> list[SourceStatus]:
        return [item for item in self.statuses if item.required and item.status != "ok"]


def load_documents(
    sources: list[Source] | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Load every registered source; raise if a required source has no content."""
    result = read_documents(sources, settings)
    failed = result.required_failures
    if failed:
        names = ", ".join(f"{item.source_id} ({item.detail})" for item in failed)
        raise DocumentLoadError(f"Required sources not loaded: {names}", result.statuses)
    return result.documents


def read_documents(
    sources: list[Source] | None = None,
    settings: Settings | None = None,
) -> DocumentLoadResult:
    """Load what is available and report per-source status without raising."""
    settings = settings or get_settings()
    sources = sources if sources is not None else load_sources()
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    min_chars = settings.chunking.min_chunk_chars

    result = DocumentLoadResult()
    for source in sources:
        document, status = _load_one(source, raw_dir, min_chars)
        result.statuses.append(status)
        if document is not None:
            result.documents.append(document)
    return result


def _load_one(source: Source, raw_dir: Path, min_chars: int) -> tuple[Document | None, SourceStatus]:
    def fail(detail: str) -> tuple[None, SourceStatus]:
        return None, SourceStatus(source.id, source.required, "error", detail=detail)

    meta_path = raw_dir / f"{source.id}.meta.json"
    txt_path = raw_dir / f"{source.id}.txt"
    if not meta_path.exists():
        return fail("not scraped (no meta.json)")
    try:
        meta = RawMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        return fail(f"unreadable meta.json: {exc.__class__.__name__}")
    if meta.error:
        return fail(f"fetch error: {meta.error}")
    if meta.url != source.url:
        return fail("meta url differs from sources.yaml; re-run scrape")
    if not txt_path.exists():
        return fail("no extracted .txt")

    sections = clean_sections(parse_sections_file(txt_path.read_text(encoding="utf-8")), min_chars)
    if not sections:
        return fail("no usable sections after cleaning")

    document = Document(
        source_id=source.id,
        title=source.title,
        url=source.url,
        type=source.type,
        scheme=source.scheme,
        fetched_at=meta.fetched_at,
        sections=sections,
    )
    status = SourceStatus(
        source.id,
        source.required,
        "ok",
        sections=len(sections),
        chars=document.char_count,
    )
    return document, status


def clean_sections(sections: list[Section], min_chars: int) -> list[Section]:
    """Normalize text, merge sections sharing a heading, drop duplicates and fragments."""
    merged: dict[str, tuple[str, list[str]]] = {}
    for item in sections:
        heading = normalize_text(item.heading) or "Document"
        key = heading.lower()
        if key not in merged:
            merged[key] = (heading, [])
        lines = merged[key][1]
        for line in normalize_text(item.text).split("\n"):
            if line and line not in lines:
                lines.append(line)

    cleaned: list[Section] = []
    seen_texts: set[str] = set()
    for heading, lines in merged.values():
        text = "\n".join(lines)
        if len(text) < min_chars or text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())
        cleaned.append(Section(heading=heading, text=text))
    return cleaned


def normalize_text(text: str) -> str:
    """NFKC (turns non-breaking spaces into spaces; keeps ₹ and %), strip zero-width chars,
    collapse runs of spaces, and drop blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
