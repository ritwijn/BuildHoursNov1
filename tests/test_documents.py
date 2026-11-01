import json

import pytest

from src.config import PathsConfig, Settings, Source
from src.ingest.documents import (
    DocumentLoadError,
    clean_sections,
    load_documents,
    normalize_text,
    read_documents,
)
from src.models import Section

LARGE_CAP = Source(
    id="large-cap",
    title="HDFC Large Cap Fund Direct Growth",
    url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    type="scheme",
    scheme="HDFC Large Cap Fund",
    required=True,
)
AMFI = Source(
    id="amfi",
    title="AMFI Investor Corner",
    url="https://www.amfiindia.com/investor-corner",
    type="regulator",
)

KEY_FACTS = (
    "## Key facts\n\n"
    "Expense ratio: 1.02%\n"
    "Exit load: Exit load of 1% if redeemed within 1 year\n"
    "Minimum SIP: ₹100\n"
)


def _settings(tmp_path):
    return Settings(paths=PathsConfig(raw_dir=str(tmp_path)))


def _write(tmp_path, source, text=None, error=None, url=None):
    meta = {
        "source_id": source.id,
        "title": source.title,
        "url": url or source.url,
        "fetched_at": "2026-11-01T06:00:00Z",
        "http_status": None if error else 200,
        "error": error,
    }
    (tmp_path / f"{source.id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if text is not None:
        (tmp_path / f"{source.id}.txt").write_text(text, encoding="utf-8")


def test_loads_document_with_registry_and_meta_fields(tmp_path):
    _write(tmp_path, LARGE_CAP, KEY_FACTS)
    [doc] = load_documents([LARGE_CAP], _settings(tmp_path))
    assert doc.source_id == "large-cap"
    assert doc.url == LARGE_CAP.url
    assert doc.type == "scheme"
    assert doc.scheme == "HDFC Large Cap Fund"
    assert doc.fetched_at == "2026-11-01T06:00:00Z"
    assert doc.sections[0].heading == "Key facts"
    assert "Expense ratio: 1.02%" in doc.sections[0].text


def test_optional_fetch_error_is_skipped_not_fatal(tmp_path):
    _write(tmp_path, LARGE_CAP, KEY_FACTS)
    _write(tmp_path, AMFI, error="HTTP 404")
    result = read_documents([LARGE_CAP, AMFI], _settings(tmp_path))
    assert [doc.source_id for doc in result.documents] == ["large-cap"]
    amfi = next(item for item in result.statuses if item.source_id == "amfi")
    assert amfi.status == "error" and "HTTP 404" in amfi.detail
    assert not result.required_failures
    assert len(load_documents([LARGE_CAP, AMFI], _settings(tmp_path))) == 1


@pytest.mark.parametrize(
    "setup, detail",
    [
        (lambda tmp: None, "not scraped"),
        (lambda tmp: _write(tmp, LARGE_CAP, error="HTTP 500"), "fetch error"),
        (lambda tmp: _write(tmp, LARGE_CAP), "no extracted .txt"),
        (lambda tmp: _write(tmp, LARGE_CAP, "## Nav\n\nHome\n"), "no usable sections"),
        (lambda tmp: _write(tmp, LARGE_CAP, KEY_FACTS, url="https://groww.in/old"), "re-run scrape"),
    ],
)
def test_required_source_without_content_fails(tmp_path, setup, detail):
    setup(tmp_path)
    with pytest.raises(DocumentLoadError) as excinfo:
        load_documents([LARGE_CAP], _settings(tmp_path))
    assert detail in str(excinfo.value)
    assert excinfo.value.statuses[0].status == "error"


def test_normalize_text_handles_unicode_and_whitespace():
    raw = "Minimum SIP:​  ₹100 \n\n\n  Expense   ratio: 0.98%  "
    assert normalize_text(raw) == "Minimum SIP: ₹100\nExpense ratio: 0.98%"


def test_clean_sections_merges_same_heading_and_drops_duplicates_and_fragments():
    sections = [
        Section(heading="Understand terms", text="Expense ratio is a fee charged by the fund house."),
        Section(heading="Understand  terms", text="Exit load is a fee for redeeming early.\nExpense ratio is a fee charged by the fund house."),
        Section(heading="Copy", text="Expense ratio is a fee charged by the fund house.\nExit load is a fee for redeeming early."),
        Section(heading="Tiny", text="Home"),
    ]
    cleaned = clean_sections(sections, min_chars=40)
    assert [item.heading for item in cleaned] == ["Understand terms"]
    assert cleaned[0].text.split("\n") == [
        "Expense ratio is a fee charged by the fund house.",
        "Exit load is a fee for redeeming early.",
    ]
