"""Load YAML config and apply environment overrides. Secrets stay in .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Source(BaseModel):
    id: str
    title: str
    url: str
    type: str
    scheme: str | None = None
    aliases: list[str] = Field(default_factory=list)
    required: bool = False
    pages: list[int] = Field(default_factory=list)  # PDFs only: 1-based pages to keep; empty = all


class EmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    batch_size: int = 32


class ChromaConfig(BaseModel):
    persist_dir: str = "data/chroma"
    collection: str = "mf_faq"
    distance: str = "cosine"


class ChunkingConfig(BaseModel):
    strategy: str = "section"
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_chars: int = 40
    fact_chunks: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_score: float = 0.35
    scheme_boost: bool = True
    # Appended to the question before embedding only; every question here is about mutual funds.
    query_suffix: str = " (mutual funds)"


class LoaderConfig(BaseModel):
    timeout_s: float = 20
    user_agent: str = "MF-FAQ-Prototype/1.0 (educational demo)"
    min_text_chars: int = 500
    use_browser_fallback: bool = True
    request_delay_s: float = 1.0


class AnswerConfig(BaseModel):
    max_sentences: int = 3
    refusal_url_source_id: str = "sebi-investor-education"
    returns_source_id: str | None = "hdfc-factsheet-jun-2026"
    fallback_source_id: str = "hdfc-amc"
    last_updated_field: str = "last_ingest_at"
    dump_prompts: bool = False


class LlmConfig(BaseModel):
    provider: str = "groq"
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0
    timeout_s: float = 30
    max_tokens: int = 400


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    chunks_file: str = "data/chunks/chunks.jsonl"
    embeddings_dir: str = "data/embeddings"
    manifest_file: str = "data/ingest_manifest.json"
    retrieval_log_dir: str = "data/retrieval_logs"


class EnvOverrides(BaseSettings):
    """Optional env / .env keys. Nested YAML is the default."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    embedding_model: str | None = None
    chroma_persist_dir: str | None = None
    chroma_collection: str | None = None
    top_k: int | None = None
    min_score: float | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_timeout_s: float | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None


class Settings(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else ROOT / path


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_env(settings: Settings, env: EnvOverrides) -> Settings:
    data = settings.model_dump()
    if env.embedding_model:
        data["embedding"]["model"] = env.embedding_model
    if env.chroma_persist_dir:
        data["chroma"]["persist_dir"] = env.chroma_persist_dir
    if env.chroma_collection:
        data["chroma"]["collection"] = env.chroma_collection
    if env.top_k is not None:
        data["retrieval"]["top_k"] = env.top_k
    if env.min_score is not None:
        data["retrieval"]["min_score"] = env.min_score
    if env.llm_provider:
        data["llm"]["provider"] = env.llm_provider
    if env.llm_model:
        data["llm"]["model"] = env.llm_model
    if env.llm_temperature is not None:
        data["llm"]["temperature"] = env.llm_temperature
    if env.llm_timeout_s is not None:
        data["llm"]["timeout_s"] = env.llm_timeout_s
    data["groq_api_key"] = env.groq_api_key
    data["gemini_api_key"] = env.gemini_api_key
    data["openai_api_key"] = env.openai_api_key
    data["anthropic_api_key"] = env.anthropic_api_key
    data["mistral_api_key"] = env.mistral_api_key
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    raw = _load_yaml(ROOT / "config" / "settings.yaml")
    settings = Settings.model_validate(raw)
    return _apply_env(settings, EnvOverrides())


def load_sources() -> list[Source]:
    raw = _load_yaml(ROOT / "config" / "sources.yaml")
    return [Source.model_validate(item) for item in raw.get("sources", [])]


def load_intents() -> dict[str, list[str]]:
    raw = _load_yaml(ROOT / "config" / "intents.yaml")
    return {key: list(value or []) for key, value in raw.items()}


def allowed_source_urls() -> set[str]:
    return {source.url for source in load_sources()}
