"""Embed a question, query Chroma, and log what came back.

Only call this after the PII filter has passed: it writes the question to
data/retrieval_logs/{query_id}.json.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import Settings, Source, get_settings, load_sources
from src.ingest.embedder import embed_texts
from src.ingest.store import open_collection
from src.models import Hit


@dataclass
class RetrievalResult:
    query_id: str
    hits: list[Hit]
    best_score: float
    below_threshold: bool


def retrieve(
    query: str,
    *,
    intent: str,
    scheme_id: str | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
    sources: list[Source] | None = None,
) -> RetrievalResult:
    settings = settings or get_settings()
    cfg = settings.retrieval
    top_k = top_k or cfg.top_k
    collection = open_collection(settings)

    # With a named scheme, over-fetch so chunks from any source about that scheme
    # (Groww page, KIM, SID, fund facts…) can be moved up front.
    boost_ids = scheme_source_ids(scheme_id, sources if sources is not None else load_sources()) if cfg.scheme_boost else set()
    n_results = min(collection.count(), top_k * 3 if boost_ids else top_k)
    embedded_query = f"{query}{cfg.query_suffix}"
    vector = embed_texts([embedded_query], settings)
    raw = collection.query(
        query_embeddings=vector,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        Hit(
            chunk_id=meta["chunk_id"],
            source_id=meta["source_id"],
            source_url=meta["source_url"],
            section=meta.get("section", ""),
            text=document,
            score=round(1.0 - distance, 4),
        )
        for document, meta, distance in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
    ]
    if boost_ids:
        hits.sort(key=lambda hit: (hit.source_id not in boost_ids, -hit.score))
    hits = hits[:top_k]

    best = max((hit.score for hit in hits), default=0.0)
    result = RetrievalResult(
        query_id=str(uuid.uuid4()),
        hits=hits,
        best_score=best,
        below_threshold=best < cfg.min_score,
    )
    write_log(
        result.query_id,
        {
            "query_id": result.query_id,
            "asked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": query,
            "embedded_query": embedded_query,
            "intent": intent,
            "scheme_id": scheme_id,
            "top_k": top_k,
            "min_score": cfg.min_score,
            "best_score": best,
            "below_threshold": result.below_threshold,
            "hits": [hit.model_dump() for hit in hits],
        },
        settings,
    )
    return result


def scheme_source_ids(scheme_id: str | None, sources: list[Source]) -> set[str]:
    """Ids of every source about the same scheme as `scheme_id` (a Groww scheme page id)."""
    by_id = {source.id: source for source in sources}
    if scheme_id not in by_id:
        return set()
    scheme = by_id[scheme_id].scheme
    if not scheme:
        return {scheme_id}
    return {source.id for source in sources if source.scheme == scheme}


def write_log(query_id: str, data: dict, settings: Settings | None = None) -> None:
    """Write or merge fields into data/retrieval_logs/{query_id}.json."""
    settings = settings or get_settings()
    log_dir = settings.resolve_path(settings.paths.retrieval_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{query_id}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(data)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
