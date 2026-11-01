"""Stage-by-stage CLI for the FAQ assistant demo."""

from __future__ import annotations

import argparse
import sys

from src.config import ROOT, get_settings, load_sources
from src.ingest.chunker import run_chunking
from src.ingest.documents import DocumentLoadError, read_documents
from src.ingest.embedder import EmbeddingsStale, run_embedding
from src.ingest.loader import load_all
from src.ingest.store import IngestRequired, build_store, open_collection, read_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scrape", help="Scrape allowlisted sources into data/raw/")
    sub.add_parser("load", help="Load data/raw/ into cleaned documents and report per source")
    sub.add_parser("chunk", help="Chunk loaded documents into data/chunks/chunks.jsonl")
    sub.add_parser("embed", help="Embed chunks.jsonl into data/embeddings/")
    sub.add_parser("store", help="Rebuild the Chroma collection and write ingest_manifest.json")
    ingest = sub.add_parser("ingest", help="Run scrape → chunk → embed → store")
    ingest.add_argument("--no-scrape", action="store_true", help="Reuse data/raw/ instead of re-fetching")
    retrieve_cmd = sub.add_parser("retrieve", help="Show top-k chunks for a question (writes a retrieval log)")
    retrieve_cmd.add_argument("question")
    ask_cmd = sub.add_parser("ask", help="Answer a question through the full guarded pipeline")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("-v", "--verbose", action="store_true", help="Also print hits and post-check flags")
    export = sub.add_parser("export-sources", help="Write Docs/sources.csv from data/raw metadata")
    export.add_argument("--out", default="Docs/sources.csv")
    sub.add_parser("status", help="Show config and artifact paths")
    args = parser.parse_args(argv)

    if args.command == "scrape":
        return load_all()
    if args.command == "load":
        return _load()
    if args.command == "chunk":
        return _chunk()
    if args.command == "embed":
        return _embed()
    if args.command == "store":
        return _store()
    if args.command == "ingest":
        return _ingest(scrape=not args.no_scrape)
    if args.command == "retrieve":
        return _retrieve(args.question)
    if args.command == "ask":
        return _ask(args.question, args.verbose)
    if args.command == "export-sources":
        return _export_sources(args.out)
    if args.command == "status":
        return _status()
    return 1


def _load() -> int:
    result = read_documents()
    print(f"{'status':6}  {'source':40}  {'sections':>8}  {'chars':>6}  detail")
    for item in result.statuses:
        flag = "ok" if item.status == "ok" else ("FAIL" if item.required else "skip")
        print(f"{flag:6}  {item.source_id:40}  {item.sections:>8}  {item.chars:>6}  {item.detail}")
    print(f"documents: {len(result.documents)} / {len(result.statuses)} sources")
    if result.required_failures:
        print("required sources missing; run `python -m src.cli scrape` and check data/raw/*.meta.json")
        return 1
    return 0


def _chunk() -> int:
    try:
        run_chunking()
    except DocumentLoadError as exc:
        print(f"{exc}; run `python -m src.cli load` for details")
        return 1
    return 0


def _embed() -> int:
    try:
        index = run_embedding()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    settings = get_settings()
    print(f"model={index['embedding_model']} normalized={index['normalized']}")
    print(f"vectors={index['count']} dimension={index['dimension']}")
    print(f"wrote {settings.resolve_path(settings.paths.embeddings_dir)}")
    return 0


def _store() -> int:
    try:
        manifest = build_store()
    except (FileNotFoundError, EmbeddingsStale) as exc:
        print(exc)
        return 1
    print(f"collection={manifest['chroma_collection']} vectors={manifest['chunk_count']} "
          f"sources={manifest['source_count']} distance={manifest['chroma_distance']}")
    print(f"last_ingest_at={manifest['last_ingest_at']} sources_fetched_at={manifest['sources_fetched_at']}")
    settings = get_settings()
    print(f"wrote {settings.resolve_path(settings.chroma.persist_dir)}")
    print(f"wrote {settings.resolve_path(settings.paths.manifest_file)}")
    return 0


def _ingest(scrape: bool) -> int:
    steps = [("scrape", load_all)] if scrape else []
    steps += [("chunk", _chunk), ("embed", _embed), ("store", _store)]
    for name, step in steps:
        print(f"== {name}")
        if step() != 0:
            print(f"ingest stopped at {name}")
            return 1
    return 0


def _retrieve(question: str) -> int:
    from src.generate.intent import classify
    from src.generate.pii import PII_MESSAGE, detect_pii
    from src.retrieve.retriever import retrieve

    if detect_pii(question):
        print(PII_MESSAGE)
        return 1
    intent = classify(question)
    try:
        result = retrieve(question, intent=intent.intent, scheme_id=intent.scheme_id)
    except IngestRequired as exc:
        print(exc)
        return 1
    print(f"intent={intent.intent} scheme={intent.scheme_id} best={result.best_score:.3f} "
          f"below_threshold={result.below_threshold}")
    for rank, hit in enumerate(result.hits, start=1):
        print(f"{rank}. {hit.score:.3f}  {hit.chunk_id:40}  {hit.section}")
        print(f"   {hit.source_url}")
    print(f"log: {result.query_id}.json")
    return 0


def _ask(question: str, verbose: bool) -> int:
    from src.generate.service import answer_question

    try:
        answer = answer_question(question)
    except IngestRequired as exc:
        print(exc)
        return 1
    print(answer.render())
    if verbose:
        print(f"\n[status={answer.status} intent={answer.intent} scheme={answer.scheme_id} query_id={answer.query_id}]")
        for hit in answer.hits:
            print(f"  {hit.score:.3f}  {hit.chunk_id:40}  {hit.section}")
        if answer.flags:
            print(f"  flags: {', '.join(answer.flags)}")
    return 0 if answer.status != "error" else 1


def _export_sources(out: str) -> int:
    import csv
    import json

    from src.ingest.chunker import read_chunks

    settings = get_settings()
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    chunks_path = settings.resolve_path(settings.paths.chunks_file)
    counts: dict[str, int] = {}
    if chunks_path.exists():
        for chunk in read_chunks(chunks_path):
            counts[chunk.source_id] = counts.get(chunk.source_id, 0) + 1

    rows = []
    for source in load_sources():
        meta_path = raw_dir / f"{source.id}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        status = "not scraped" if not meta else ("error: " + meta["error"] if meta.get("error") else "ok")
        rows.append({
            "source_id": source.id,
            "title": source.title,
            "url": source.url,
            "type": source.type,
            "scheme": source.scheme or "",
            "required": "yes" if source.required else "no",
            "pages_used": " ".join(str(page) for page in source.pages) or "all",
            "fetched_at": meta.get("fetched_at", ""),
            "http_status": meta.get("http_status", ""),
            "status": status,
            "chunks": counts.get(source.id, 0),
        })

    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    used = sum(1 for row in rows if row["status"] == "ok")
    print(f"wrote {path}: {len(rows)} sources, {used} ok")
    return 0 if used == len(rows) else 1


def _status() -> int:
    settings = get_settings()
    sources = load_sources()
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    print(f"root: {ROOT}")
    print(f"embedding: {settings.embedding.model}")
    print(f"chroma: {settings.chroma.persist_dir} / {settings.chroma.collection}")
    print(f"chunking: {settings.chunking.strategy} size={settings.chunking.chunk_size}")
    print(f"top_k: {settings.retrieval.top_k}")
    print(f"llm: {settings.llm.provider} {settings.llm.model}")
    print(f"sources: {len(sources)} ({sum(1 for item in sources if item.required)} required)")
    print(f"raw_dir: {raw_dir} exists={raw_dir.is_dir()}")
    for source in sources:
        txt = raw_dir / f"{source.id}.txt"
        print(f"  {source.id}: txt={'yes' if txt.exists() else 'no'}")
    manifest = read_manifest(settings)
    if manifest is None:
        print("manifest: none (run `python -m src.cli ingest`)")
        return 0
    print(f"manifest: last_ingest_at={manifest['last_ingest_at']} chunks={manifest['chunk_count']}")
    try:
        collection = open_collection(settings)
    except IngestRequired as exc:
        print(f"chroma: {exc}")
        return 0
    print(f"chroma: {collection.count()} vectors")
    sample = collection.get(limit=3, include=["metadatas"])
    for meta in sample["metadatas"]:
        print(f"  {meta['chunk_id']:40}  {meta['section']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
