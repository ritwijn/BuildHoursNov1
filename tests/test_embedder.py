import numpy as np
import pytest

from src.config import EmbeddingConfig, PathsConfig, Settings
from src.ingest import embedder
from src.ingest.chunker import write_chunks
from src.models import Chunk


class FakeModel:
    """Deterministic stand-in for SentenceTransformer: no download, 4-dim vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.calls.append({"batch_size": batch_size, "normalize": normalize_embeddings})
        rows = np.array([[len(text), text.count("a"), text.count("e"), 1.0] for text in texts], dtype=np.float64)
        if normalize_embeddings:
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        return rows


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embedder, "get_model", lambda *args, **kwargs: model)
    return model


def _settings(tmp_path, model="sentence-transformers/all-MiniLM-L6-v2"):
    return Settings(
        embedding=EmbeddingConfig(model=model, batch_size=8),
        paths=PathsConfig(
            chunks_file=str(tmp_path / "chunks" / "chunks.jsonl"),
            embeddings_dir=str(tmp_path / "embeddings"),
        ),
    )


def _write_chunks(settings, texts):
    chunks = [
        Chunk(chunk_id=f"s::{i:03d}", source_id="s", source_url="https://groww.in/x", section="Key facts", text=text, chunk_index=i)
        for i, text in enumerate(texts)
    ]
    write_chunks(chunks, settings.resolve_path(settings.paths.chunks_file))


def test_embed_texts_uses_config_and_returns_normalized_lists(tmp_path, fake_model):
    vectors = embedder.embed_texts(["expense ratio", "exit load"], _settings(tmp_path))
    assert len(vectors) == 2 and all(isinstance(value, float) for value in vectors[0])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert fake_model.calls == [{"batch_size": 8, "normalize": True}]


def test_run_embedding_writes_aligned_artifacts(tmp_path, fake_model):
    settings = _settings(tmp_path)
    _write_chunks(settings, ["Expense ratio: 1.02%", "Exit load: 1% within 1 year", "Minimum SIP: ₹100"])
    index = embedder.run_embedding(settings)
    assert index["count"] == 3 and index["dimension"] == 4
    assert index["chunk_ids"] == ["s::000", "s::001", "s::002"]

    chunk_ids, vectors, stored = embedder.read_embeddings(settings)
    assert chunk_ids == index["chunk_ids"]
    assert vectors.shape == (3, 4) and vectors.dtype == np.float32
    assert stored["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_read_embeddings_rejects_model_change(tmp_path, fake_model):
    settings = _settings(tmp_path)
    _write_chunks(settings, ["Expense ratio: 1.02%"])
    embedder.run_embedding(settings)
    with pytest.raises(embedder.EmbeddingsStale, match="re-run ingest"):
        embedder.read_embeddings(_settings(tmp_path, model="sentence-transformers/other-model"))


def test_read_embeddings_rejects_rechunked_corpus(tmp_path, fake_model):
    settings = _settings(tmp_path)
    _write_chunks(settings, ["Expense ratio: 1.02%"])
    embedder.run_embedding(settings)
    _write_chunks(settings, ["Expense ratio: 1.02%", "Exit load: 1%"])
    with pytest.raises(embedder.EmbeddingsStale, match="chunks.jsonl changed"):
        embedder.read_embeddings(settings)


def test_run_embedding_requires_chunks(tmp_path, fake_model):
    with pytest.raises(FileNotFoundError, match="src.cli chunk"):
        embedder.run_embedding(_settings(tmp_path))
