import json

import numpy as np
import pytest

from src.config import ChromaConfig, EmbeddingConfig, PathsConfig, Settings
from src.ingest import embedder
from src.ingest.chunker import read_chunks, write_chunks
from src.ingest.embedder import EmbeddingsStale, run_embedding
from src.ingest.store import IngestRequired, build_store, open_collection, read_manifest
from src.models import Chunk

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = {
    "a::000": "HDFC Large Cap — Key facts\nExpense ratio: 1.02%",
    "a::001": "HDFC Large Cap — Exit load history\n2015-05-08: Exit load of 1%",
    "b::000": "HDFC Mid Cap — Key facts\nMinimum SIP: ₹100",
}


class FakeModel:
    def encode(self, texts, **_):
        rows = np.array([[len(t), t.count("a") + 1, t.count("e") + 1, 1.0] for t in texts], dtype=np.float64)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(embedder, "get_model", lambda *args, **kwargs: FakeModel())


def _settings(tmp_path, model=MODEL):
    return Settings(
        embedding=EmbeddingConfig(model=model),
        chroma=ChromaConfig(persist_dir=str(tmp_path / "chroma"), collection="test_mf"),
        paths=PathsConfig(
            raw_dir=str(tmp_path / "raw"),
            chunks_file=str(tmp_path / "chunks.jsonl"),
            embeddings_dir=str(tmp_path / "embeddings"),
            manifest_file=str(tmp_path / "ingest_manifest.json"),
        ),
    )


def _prepare(tmp_path, texts=TEXTS):
    settings = _settings(tmp_path)
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            source_id=chunk_id.split("::")[0],
            source_url=f"https://groww.in/{chunk_id.split('::')[0]}",
            section=text.split(" — ")[1].split("\n")[0],
            text=text,
            chunk_index=int(chunk_id.split("::")[1]),
        )
        for chunk_id, text in texts.items()
    ]
    write_chunks(chunks, settings.resolve_path(settings.paths.chunks_file))
    raw = settings.resolve_path(settings.paths.raw_dir)
    raw.mkdir()
    for source_id, fetched in (("a", "2026-11-01T06:00:00Z"), ("b", "2026-10-31T06:00:00Z")):
        meta = {"source_id": source_id, "title": source_id, "url": f"https://groww.in/{source_id}", "fetched_at": fetched}
        (raw / f"{source_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    run_embedding(settings)
    return settings


def test_build_store_persists_vectors_metadata_and_manifest(tmp_path):
    settings = _prepare(tmp_path)
    manifest = build_store(settings)
    assert manifest["chunk_count"] == 3 and manifest["source_count"] == 2
    assert manifest["sources_fetched_at"] == "2026-10-31T06:00:00Z"
    assert read_manifest(settings) == manifest

    collection = open_collection(settings)
    assert collection.count() == 3
    stored = collection.get(ids=["a::000"], include=["metadatas", "documents"])
    assert stored["metadatas"][0] == {
        "chunk_id": "a::000",
        "source_id": "a",
        "source_url": "https://groww.in/a",
        "section": "Key facts",
    }
    assert stored["documents"][0] == TEXTS["a::000"]


def test_rebuild_replaces_instead_of_duplicating(tmp_path):
    settings = _prepare(tmp_path)
    build_store(settings)
    build_store(settings)
    assert open_collection(settings).count() == 3

    _prepare(tmp_path / "second", {"a::000": TEXTS["a::000"]})
    smaller = _settings(tmp_path / "second")
    smaller.chroma.persist_dir = settings.chroma.persist_dir
    build_store(smaller)
    assert open_collection(settings).get()["ids"] == ["a::000"]


def test_query_returns_nearest_chunk_with_cosine(tmp_path):
    settings = _prepare(tmp_path)
    build_store(settings)
    vector = embedder.embed_texts([TEXTS["b::000"]], settings)
    result = open_collection(settings).query(query_embeddings=vector, n_results=1)
    assert result["ids"][0] == ["b::000"]
    assert result["distances"][0][0] == pytest.approx(0.0, abs=1e-5)


def test_open_collection_requires_ingest_and_matching_model(tmp_path):
    settings = _prepare(tmp_path)
    with pytest.raises(IngestRequired, match="not found"):
        open_collection(settings)
    build_store(settings)
    with pytest.raises(IngestRequired, match="re-run ingest"):
        open_collection(_settings(tmp_path, model="sentence-transformers/other-model"))


def test_build_store_refuses_stale_embeddings(tmp_path):
    settings = _prepare(tmp_path)
    write_chunks([], settings.resolve_path(settings.paths.chunks_file))
    with pytest.raises(EmbeddingsStale):
        build_store(settings)


def test_ensure_store_builds_when_missing_and_rebuilds_when_stale(tmp_path):
    from src.ingest.store import ensure_store

    settings = _prepare(tmp_path)
    assert ensure_store(settings) is True  # nothing built yet
    assert ensure_store(settings) is False  # up to date
    assert open_collection(settings).count() == 3

    # New data lands (e.g. a scheduled refresh): fewer chunks, re-embedded.
    write_chunks(
        [c for c in read_chunks(settings.resolve_path(settings.paths.chunks_file)) if c.source_id == "a"],
        settings.resolve_path(settings.paths.chunks_file),
    )
    run_embedding(settings)
    assert ensure_store(settings) is True
    assert sorted(open_collection(settings).get()["ids"]) == ["a::000", "a::001"]


def test_runtime_build_does_not_touch_the_manifest(tmp_path):
    settings = _prepare(tmp_path)
    build_store(settings, write_manifest=False)
    assert read_manifest(settings) is None
