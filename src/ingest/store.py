"""Persist chunk vectors in ChromaDB and write data/ingest_manifest.json.

The Chroma store itself is not committed: it is rebuilt from the committed
chunks.jsonl + data/embeddings/ by `ingest`/`store`, or on demand by
ensure_store() (Vercel cold start, fresh clone, or after new data lands).

Every build deletes and recreates the collection so no stale chunks survive a
re-chunk. Vectors come from data/embeddings/ (validated against the current
model and chunks.jsonl), never re-computed here.

The collection records the embedding model it was built with; opening it for
queries with a different configured model raises instead of returning
meaningless matches.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from src.config import Settings, get_settings
from src.ingest.embedder import EmbeddingsStale, read_embeddings
from src.models import RawMeta


class IngestRequired(Exception):
    """The vector store is missing, empty, or built with a different model."""


def get_client(settings: Settings):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    path = settings.resolve_path(settings.chroma.persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))


def build_store(settings: Settings | None = None, *, write_manifest: bool = True) -> dict:
    """Rebuild the collection from chunks.jsonl + data/embeddings/.

    With write_manifest (ingest), also write data/ingest_manifest.json and return it.
    Without it (runtime rebuilds), the committed manifest stays the source of truth.
    """
    from src.ingest.chunker import read_chunks

    settings = settings or get_settings()
    chunks_path = settings.resolve_path(settings.paths.chunks_file)
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks at {chunks_path}; run `python -m src.cli chunk` first")
    chunks = read_chunks(chunks_path)
    chunk_ids, vectors, index = read_embeddings(settings)
    if [chunk.chunk_id for chunk in chunks] != chunk_ids:
        raise EmbeddingsStale("chunk order differs from embeddings; re-run `python -m src.cli embed`")

    # Start from an empty directory: delete_collection() leaves the old HNSW segment
    # folders on disk, which would pile up across rebuilds (and bloat the Vercel bundle).
    persist_dir = settings.resolve_path(settings.chroma.persist_dir)
    if persist_dir.exists():
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        shutil.rmtree(persist_dir)
    client = get_client(settings)
    name = settings.chroma.collection
    collection = client.create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": settings.chroma.distance}},
        metadata={
            "embedding_model": index["embedding_model"],
            "dimension": index["dimension"],
            "chunks_sha256": index["chunks_sha256"],
        },
    )

    batch = client.get_max_batch_size()
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        collection.add(
            ids=[chunk.chunk_id for chunk in part],
            embeddings=vectors[start : start + batch],
            documents=[chunk.text for chunk in part],
            metadatas=[
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "source_url": chunk.source_url,
                    "section": chunk.section or "",
                }
                for chunk in part
            ],
        )

    count = collection.count()
    if count != len(chunks):
        raise RuntimeError(f"Chroma holds {count} vectors but there are {len(chunks)} chunks")

    source_ids = sorted({chunk.source_id for chunk in chunks})
    if not write_manifest:
        return {"chroma_collection": name, "chunk_count": count, "source_count": len(source_ids)}
    manifest = {
        "last_ingest_at": _now(),
        "sources_fetched_at": _oldest_fetch(settings, source_ids),
        "embedding_model": index["embedding_model"],
        "embedding_dimension": index["dimension"],
        "chroma_persist_dir": settings.chroma.persist_dir,
        "chroma_collection": name,
        "chroma_distance": settings.chroma.distance,
        "chunking": settings.chunking.model_dump(),
        "source_count": len(source_ids),
        "chunk_count": count,
        "sources": source_ids,
    }
    path = settings.resolve_path(settings.paths.manifest_file)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return manifest


def open_collection(settings: Settings | None = None):
    """Open the collection for querying; raise IngestRequired if it can't be trusted."""
    settings = settings or get_settings()
    client = get_client(settings)
    name = settings.chroma.collection
    if name not in [item.name for item in client.list_collections()]:
        raise IngestRequired(f"Chroma collection '{name}' not found; run `python -m src.cli ingest`")
    collection = client.get_collection(name, embedding_function=None)
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != settings.embedding.model:
        raise IngestRequired(
            f"Collection was built with {built_with} but config says "
            f"{settings.embedding.model}; re-run ingest"
        )
    if collection.count() == 0:
        raise IngestRequired(f"Chroma collection '{name}' is empty; run `python -m src.cli ingest`")
    return collection


def ensure_store(settings: Settings | None = None) -> bool:
    """Make sure the collection matches the committed embeddings; rebuild if not.

    Returns True if it (re)built. Raises if there is nothing to build from.
    """
    settings = settings or get_settings()
    _, _, index = read_embeddings(settings)
    try:
        collection = open_collection(settings)
        if (collection.metadata or {}).get("chunks_sha256") == index["chunks_sha256"]:
            return False
    except IngestRequired:
        pass
    build_store(settings, write_manifest=False)
    return True


def read_manifest(settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    path = settings.resolve_path(settings.paths.manifest_file)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _oldest_fetch(settings: Settings, source_ids: list[str]) -> str | None:
    """Oldest fetched_at among the stored sources: the facts are at least this fresh."""
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    dates = []
    for source_id in source_ids:
        meta_path = raw_dir / f"{source_id}.meta.json"
        if meta_path.exists():
            dates.append(RawMeta.model_validate_json(meta_path.read_text(encoding="utf-8")).fetched_at)
    return min(dates) if dates else None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
