"""Streamlit smoke tests. They use the real ingested corpus, so they skip if ingest hasn't run.
Only template routes are exercised, so no LLM call is made."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.ingest.store import read_manifest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py")
pytestmark = pytest.mark.skipif(read_manifest() is None, reason="run `python -m src.cli ingest` first")


def run(question=None):
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    if question:
        app.chat_input[0].set_value(question).run()
    return app


def all_markdown(app):
    return "\n".join(
        [item.value for item in app.markdown]
        + [item.value for item in app.caption]
        + [item.value for item in app.info]
        + [item.value for item in app.warning]
    )


def test_landing_page_has_required_elements():
    app = run()
    assert not app.exception
    text = all_markdown(app)
    assert "Facts-only. No investment advice." in text
    assert "not a substitute for the scheme information document" in text
    assert len([b for b in app.button if "HDFC" in b.label]) == 3


def test_advice_question_shows_refusal_source_and_date():
    app = run("Should I buy HDFC Flexi Cap?")
    assert not app.exception
    text = all_markdown(app)
    assert "not buy/sell or portfolio advice" in text
    assert "**Source:** [https://investor.sebi.gov.in/iematerial.html]" in text
    assert "**Last updated from sources:**" in text


def test_pii_question_is_not_echoed():
    app = run("My PAN is ABCDE1234F, what is my SIP?")
    assert not app.exception
    rendered = all_markdown(app) + "\n".join(str(item.value) for item in app.get("markdown"))
    assert "ABCDE1234F" not in rendered
    assert "Please don't share personal details" in rendered
