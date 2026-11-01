import pytest

from src.config import ChunkingConfig
from src.ingest.chunker import chunk_document, chunk_documents, read_chunks, split_text, write_chunks
from src.models import Document, Section

KEY_FACTS = "\n".join(
    [
        "Expense ratio: 1.02%",
        "Exit load: Exit load of 1% if redeemed within 1 year",
        "Minimum SIP: ₹100",
        "Riskometer: Very High",
        "Benchmark: NIFTY 100 Total Return Index",
    ]
)


def _doc(source_id="large-cap", title="HDFC Large Cap Fund Direct Growth", sections=None):
    return Document(
        source_id=source_id,
        title=title,
        url=f"https://groww.in/mutual-funds/{source_id}",
        type="scheme",
        fetched_at="2026-11-01T06:00:00Z",
        sections=sections
        or [
            Section(heading="Key facts", text=KEY_FACTS),
            Section(heading="About", text="The scheme invests predominantly in large-cap companies."),
        ],
    )


def test_section_strategy_one_chunk_per_section_with_title_prefix():
    chunks = chunk_document(_doc(), ChunkingConfig(strategy="section", fact_chunks=False))
    assert [chunk.chunk_id for chunk in chunks] == ["large-cap::000", "large-cap::001"]
    assert [chunk.section for chunk in chunks] == ["Key facts", "About"]
    assert chunks[0].text.startswith("HDFC Large Cap Fund Direct Growth — Key facts\n")
    assert "Expense ratio: 1.02%" in chunks[0].text
    assert chunks[0].source_url == "https://groww.in/mutual-funds/large-cap"


def test_long_section_splits_on_line_boundaries_with_overlap():
    cfg = ChunkingConfig(strategy="section", chunk_size=80, chunk_overlap=60, fact_chunks=False)
    chunks = chunk_document(_doc(sections=[Section(heading="Key facts", text=KEY_FACTS)]), cfg)
    lines = set(KEY_FACTS.split("\n"))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section == "Key facts"
        prefix, body = chunk.text.split("\n", 1)
        assert prefix == "HDFC Large Cap Fund Direct Growth — Key facts"
        assert len(body) <= 80
        assert all(line in lines for line in body.split("\n"))
    covered = {line for chunk in chunks for line in chunk.text.split("\n")[1:]}
    assert covered == lines
    assert any(
        set(a.text.split("\n")[1:]) & set(b.text.split("\n")[1:]) for a, b in zip(chunks, chunks[1:])
    )


def test_split_text_falls_back_to_words_for_a_single_long_line():
    text = " ".join(["word"] * 60)
    pieces = split_text(text, 50, 10, ["\n", ". ", " ", ""])
    assert all(len(piece) <= 50 for piece in pieces)
    assert all(piece in text for piece in pieces)
    assert " ".join(pieces).split()[:5] == ["word"] * 5


def test_recursive_strategy_tracks_heading_where_chunk_starts():
    long_about = ". ".join(f"Sentence number {i} about the fund objective" for i in range(20))
    doc = _doc(sections=[Section(heading="Key facts", text=KEY_FACTS), Section(heading="About", text=long_about)])
    chunks = chunk_document(doc, ChunkingConfig(strategy="recursive", chunk_size=300, chunk_overlap=50))
    assert chunks[0].section == "Key facts"
    assert chunks[-1].section == "About"
    assert all(len(chunk.text.split("\n", 1)[1]) <= 300 for chunk in chunks)


def test_ids_are_stable_and_round_trip_through_jsonl(tmp_path):
    docs = [_doc(), _doc("mid-cap", "HDFC Mid Cap Fund Direct Growth")]
    cfg = ChunkingConfig()
    first = chunk_documents(docs, cfg)
    assert first == chunk_documents(docs, cfg)
    path = tmp_path / "chunks" / "chunks.jsonl"
    write_chunks(first, path)
    assert read_chunks(path) == first
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(first)


@pytest.mark.parametrize(
    "cfg",
    [ChunkingConfig(strategy="sentence"), ChunkingConfig(chunk_size=100, chunk_overlap=100)],
)
def test_invalid_config_raises(cfg):
    with pytest.raises(ValueError):
        chunk_documents([_doc()], cfg)


def test_fact_list_sections_also_get_one_chunk_per_fact():
    chunks = chunk_document(_doc(), ChunkingConfig(strategy="section"))
    facts = [chunk for chunk in chunks if chunk.section == "Key facts"]
    assert facts[0].text.endswith(KEY_FACTS)
    assert [chunk.text.split("\n", 1)[1] for chunk in facts[1:]] == KEY_FACTS.split("\n")
    assert all(chunk.text.startswith("HDFC Large Cap Fund Direct Growth — Key facts\n") for chunk in facts)
    about = [chunk for chunk in chunks if chunk.section == "About"]
    assert len(about) == 1
