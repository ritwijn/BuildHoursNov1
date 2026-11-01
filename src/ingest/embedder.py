"""Embed chunk text with the configured sentence-transformers model.

The same `embed_texts` is used for chunks at ingest and for questions at query
time, so both sides always use the same model and normalization.

Artifacts in data/embeddings/:
- embeddings.npy  float32 matrix, one row per chunk, in chunks.jsonl order
- index.json      model, dimension, chunk_ids, and a hash of chunks.jsonl, so later
                  stages can tell when vectors are stale or from a different model
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import Settings, get_settings

VECTORS_FILE = "embeddings.npy"
INDEX_FILE = "index.json"


class EmbeddingsStale(Exception):
    """Stored vectors don't match the current config or chunks; re-run ingest."""


@lru_cache(maxsize=2)
def get_model(name: str, backend: str = "sentence-transformers", onnx_dir: str | None = None):
    """Return an object with sentence-transformers' encode() signature."""
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            backend = "sentence-transformers"
        except ImportError:
            backend = "onnx"
    if backend == "onnx":
        return OnnxMiniLM(Path(onnx_dir or "models/all-MiniLM-L6-v2-onnx"))
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class OnnxMiniLM:
    """all-MiniLM-L6-v2 on onnxruntime: tokenize → transformer → mean pooling → L2 normalize.

    Matches SentenceTransformer("all-MiniLM-L6-v2").encode(normalize_embeddings=True) to ~1e-7,
    without PyTorch. Files come from `python -m src.cli export-onnx`.
    """

    def __init__(self, model_dir: Path):
        import onnxruntime
        from tokenizers import Tokenizer

        if not (model_dir / "model.onnx").exists():
            raise FileNotFoundError(f"No ONNX model in {model_dir}; run `python -m src.cli export-onnx`")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts, batch_size=32, normalize_embeddings=True, **_):
        rows = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch(list(texts[start : start + batch_size]))
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            tokens = self.session.run(None, inputs)[0]
            weights = mask[..., None].astype(np.float32)
            pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            if normalize_embeddings:
                pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            rows.append(pooled.astype(np.float32))
        return np.concatenate(rows) if rows else np.zeros((0, 384), dtype=np.float32)


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    return _encode(texts, settings or get_settings()).tolist()


def _encode(texts: list[str], settings: Settings) -> np.ndarray:
    cfg = settings.embedding
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_model(cfg.model, cfg.backend, str(settings.resolve_path(cfg.onnx_dir)))
    vectors = model.encode(
        texts,
        batch_size=cfg.batch_size,
        normalize_embeddings=cfg.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def run_embedding(settings: Settings | None = None) -> dict:
    """Embed data/chunks/chunks.jsonl and write data/embeddings/; return the index."""
    from src.ingest.chunker import read_chunks

    settings = settings or get_settings()
    chunks_path = settings.resolve_path(settings.paths.chunks_file)
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks at {chunks_path}; run `python -m src.cli chunk` first")
    chunks = read_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"{chunks_path} is empty; nothing to embed")

    vectors = _encode([chunk.text for chunk in chunks], settings)
    index = {
        "embedding_model": settings.embedding.model,
        "normalized": settings.embedding.normalize,
        "dimension": int(vectors.shape[1]),
        "count": len(chunks),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chunks_sha256": _sha256(chunks_path),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }

    out_dir = settings.resolve_path(settings.paths.embeddings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_vectors = out_dir / f"{VECTORS_FILE}.tmp"
    with tmp_vectors.open("wb") as handle:
        np.save(handle, vectors)
    tmp_vectors.replace(out_dir / VECTORS_FILE)
    tmp_index = out_dir / f"{INDEX_FILE}.tmp"
    tmp_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    tmp_index.replace(out_dir / INDEX_FILE)
    return index


def read_embeddings(settings: Settings | None = None) -> tuple[list[str], np.ndarray, dict]:
    """Return (chunk_ids, vectors, index); raise EmbeddingsStale if they no longer fit."""
    settings = settings or get_settings()
    out_dir = settings.resolve_path(settings.paths.embeddings_dir)
    index_path = out_dir / INDEX_FILE
    vectors_path = out_dir / VECTORS_FILE
    if not index_path.exists() or not vectors_path.exists():
        raise EmbeddingsStale("No embeddings found; run `python -m src.cli embed`")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    if index["embedding_model"] != settings.embedding.model:
        raise EmbeddingsStale(
            f"Embeddings were built with {index['embedding_model']} but config says "
            f"{settings.embedding.model}; re-run ingest"
        )
    chunks_path = settings.resolve_path(settings.paths.chunks_file)
    if not chunks_path.exists() or _sha256(chunks_path) != index["chunks_sha256"]:
        raise EmbeddingsStale("chunks.jsonl changed since embedding; re-run `python -m src.cli embed`")
    if vectors.shape[0] != len(index["chunk_ids"]):
        raise EmbeddingsStale("embeddings.npy and index.json disagree; re-run `python -m src.cli embed`")
    return index["chunk_ids"], vectors, index


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
