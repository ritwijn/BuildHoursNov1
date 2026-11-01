from pathlib import Path

from src.ingest.extract import extract_sections, sections_to_text

FIXTURE = Path(__file__).parent / "fixtures" / "scheme.html"


def test_extracts_key_facts_from_next_data():
    html = FIXTURE.read_text(encoding="utf-8")
    sections = extract_sections(html)
    text = sections_to_text(sections).lower()
    assert any(item.heading == "Key facts" for item in sections)
    assert "expense ratio: 1.02%" in text
    assert "exit load of 1% if redeemed within 1 year" in text
    assert "minimum sip: ₹100" in text
    assert "lock-in: none" in text
    assert "riskometer: very high" in text
    assert "nifty 100 total return index" in text
