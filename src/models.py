from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    heading: str
    text: str


class Document(BaseModel):
    """One loaded source: cleaned sections plus registry and fetch metadata."""

    source_id: str
    title: str
    url: str
    type: str
    scheme: str | None = None
    fetched_at: str
    sections: list[Section] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(item.text) for item in self.sections)


class Chunk(BaseModel):
    """One line of data/chunks/chunks.jsonl; text is what gets embedded."""

    chunk_id: str
    source_id: str
    source_url: str
    section: str
    text: str
    chunk_index: int


class Hit(BaseModel):
    chunk_id: str
    source_id: str
    source_url: str
    section: str
    text: str
    score: float


class Answer(BaseModel):
    """What the query pipeline returns to the CLI / UI.

    status: answered | not_found | refused | returns | out_of_scope | pii | empty | error
    """

    status: str
    body: str
    source_url: str | None = None
    last_updated: str | None = None
    intent: str | None = None
    scheme_id: str | None = None
    query_id: str | None = None
    hits: list[Hit] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    prompt: list[dict] | None = None

    def render(self) -> str:
        lines = [self.body]
        if self.source_url:
            lines.append(f"Source: {self.source_url}")
        if self.last_updated:
            lines.append(f"Last updated from sources: {self.last_updated}")
        return "\n".join(lines)


class RawMeta(BaseModel):
    source_id: str
    title: str
    url: str
    fetched_at: str
    http_status: int | None = None
    content_type: str | None = None
    fetch_method: str | None = None
    text_chars: int = 0
    section_count: int = 0
    error: str | None = None
