import pytest

from src.config import Source, load_intents
from src.generate.intent import classify, match_scheme
from src.generate.pii import detect_pii
from src.generate.postcheck import postcheck
from src.generate.prompts import NOT_FOUND, REFUSAL
from src.models import Hit

SOURCES = [
    Source(id="hdfc-amc", title="HDFC Mutual Fund (AMC overview)", url="https://groww.in/amc", type="amc"),
    Source(id="sebi-investor-education", title="SEBI Investor Education", url="https://investor.sebi.gov.in/ie", type="regulator"),
    Source(id="hdfc-factsheet-jun-2026", title="HDFC MF Factsheet", url="https://files.hdfcfund.com/factsheet.pdf", type="factsheet"),
    Source(id="large", title="HDFC Large Cap Fund Direct Growth", url="https://groww.in/large", type="scheme",
           scheme="HDFC Large Cap Fund", aliases=["hdfc large cap", "hdfc top 100"]),
    Source(id="flexi", title="HDFC Flexi Cap Direct Plan Growth", url="https://groww.in/flexi", type="scheme",
           scheme="HDFC Flexi Cap Fund", aliases=["hdfc flexi cap", "hdfc equity fund"]),
    Source(id="elss", title="HDFC ELSS Tax Saver Fund Direct Plan Growth", url="https://groww.in/elss", type="scheme",
           scheme="HDFC ELSS Tax Saver Fund", aliases=["hdfc elss", "hdfc tax saver"]),
    Source(id="mid", title="HDFC Mid Cap Fund Direct Growth", url="https://groww.in/mid", type="scheme",
           scheme="HDFC Mid Cap Fund", aliases=["hdfc mid cap"]),
]
INTENTS = load_intents()

# --- PII -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, kind",
    [
        ("My PAN is ABCDE1234F, min SIP?", "pan"),
        ("Aadhaar 2345 6789 0123", "aadhaar"),
        ("call me on +91 98765 43210", "phone"),
        ("call 9876543210", "phone"),
        ("email me at a@b.com", "email"),
        ("OTP 123456 for my folio", "otp"),
        ("my folio number 12345678", "account"),
        ("account no: 123456789012", "account"),
    ],
)
def test_pii_detected(text, kind):
    assert kind in detect_pii(text)


@pytest.mark.parametrize(
    "text",
    [
        "What is min SIP? It is ₹500.",
        "Exit load 1% and expense 0.98%",
        "ELSS has a 3 years lock-in",
        "AUM is ₹37,293.95 Cr as of 2025",
        "How do I download my account statement?",
        "What is the benchmark NIFTY 100 TRI?",
    ],
)
def test_fund_facts_are_not_pii(text):
    assert detect_pii(text) == []


# --- Intent --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, intent, scheme",
    [
        ("What is the expense ratio of HDFC Large Cap Fund Direct Growth?", "factual", "large"),
        ("What is the ELSS lock-in for HDFC ELSS Tax Saver Fund Direct Plan Growth?", "factual", "elss"),
        ("What is the minimum SIP for HDFC Mid Cap Fund Direct Growth?", "factual", "mid"),
        ("What is the exit load of HDFC Flexi Cap Direct Plan Growth?", "factual", "flexi"),
        ("Riskometer of flexi cap?", "factual", "flexi"),
        ("How do I download a capital-gains statement?", "factual", None),
        ("Should I buy HDFC Flexi Cap?", "advice", "flexi"),
        ("Which of these funds is better for me?", "advice", None),
        ("Can I invest>", "advice", None),
        ("Is it good to invest in HDFC Mid Cap now?", "advice", "mid"),
        ("Can I buy HDFC ELSS?", "advice", "elss"),
        ("Should I invest in HDFC Flexi Cap?", "advice", "flexi"),
        ("What is the minimum SIP to invest in HDFC Mid Cap?", "factual", "mid"),
        ("How do I invest in HDFC Mid Cap on Groww?", "other", "mid"),
        ("What were the 5-year returns of HDFC Mid Cap?", "returns", "mid"),
        ("What is the expense ratio of SBI Bluechip?", "out_of_scope", None),
        ("Exit load of HDFC Small Cap Fund?", "out_of_scope", None),
        ("Who is the CEO of HDFC AMC?", "other", None),
        ("What is the expense ratio of Groww ELSS Tax Saver Fund?", "out_of_scope", None),
        ("Exit load of Groww Large Cap Fund?", "out_of_scope", None),
        ("How do I download my capital gains statement on Groww?", "factual", None),
        ("What is the exit load of HDFC Flexi Cap on Groww?", "factual", "flexi"),
        ("Is the Groww app charging an exit load on HDFC Mid Cap?", "factual", "mid"),
        ("Groww mutual fund expense ratio?", "out_of_scope", None),
    ],
)
def test_intent_routes(text, intent, scheme):
    result = classify(text, SOURCES, INTENTS)
    assert (result.intent, result.scheme_id) == (intent, scheme)


def test_keywords_match_whole_words_only():
    assert classify("Tell me about gossip", SOURCES, INTENTS).intent == "other"
    assert match_scheme("hdfc large cap fund vs hdfc mid cap", SOURCES) in {"large", "mid"}


# --- Post-check ----------------------------------------------------------------------

HITS = [
    Hit(chunk_id="large::000", source_id="large", source_url="https://groww.in/large", section="Key facts",
        text="Expense ratio: 1.02%\nMinimum SIP: ₹100\nAUM: ₹37,293.95 Cr", score=0.8),
    Hit(chunk_id="mid::000", source_id="mid", source_url="https://groww.in/mid", section="Key facts",
        text="Expense ratio: 0.75%", score=0.6),
]


def check(raw):
    return postcheck(raw, HITS, max_sentences=3, refusal_url="https://investor.sebi.gov.in/ie")


def test_valid_answer_passes_through():
    result = check("The expense ratio is 1.02%.\nSource: https://groww.in/large")
    assert (result.status, result.body, result.source_url, result.flags) == (
        "answered", "The expense ratio is 1.02%.", "https://groww.in/large", [])


def test_hallucinated_url_replaced_with_top_hit():
    result = check("The expense ratio is 1.02%.\nSource: https://evil.example/page")
    assert result.source_url == "https://groww.in/large"
    assert "citation_replaced" in result.flags


def test_two_urls_leave_one_and_body_has_none():
    result = check("Expense ratio is 0.75% (https://groww.in/large).\nSource: https://groww.in/mid")
    assert result.source_url == "https://groww.in/mid"
    assert "http" not in result.body
    assert "extra_urls_removed" in result.flags


def test_trims_to_three_sentences_without_splitting_decimals():
    raw = "It is 1.02%. Second. Third one. Fourth. Fifth.\nSource: https://groww.in/large"
    result = check(raw)
    assert result.body == "It is 1.02%. Second. Third one."
    assert "trimmed_to_3_sentences" in result.flags


@pytest.mark.parametrize("figure", ["0.98%", "₹500", "Rs. 250"])
def test_ungrounded_figures_become_not_found(figure):
    result = check(f"The value is {figure}.\nSource: https://groww.in/large")
    assert (result.status, result.body) == ("not_found", NOT_FOUND)


def test_grounded_figures_with_formatting_differences_pass():
    result = check("Minimum SIP is ₹100 and AUM is ₹37293.95 Cr.\nSource: https://groww.in/large")
    assert result.status == "answered"


def test_advice_phrasing_becomes_refusal():
    result = check("The ratio is 1.02%, so you should invest.\nSource: https://groww.in/large")
    assert (result.status, result.body, result.source_url) == ("refused", REFUSAL, "https://investor.sebi.gov.in/ie")


def test_wrapper_tags_are_removed():
    result = check("<answer> The expense ratio is 1.02%. </answer>\nSource: https://groww.in/large")
    assert result.body == "The expense ratio is 1.02%."


def test_model_not_found_and_think_blocks():
    result = check("<think>hmm 9.99%</think>I couldn't find that in my sources.")
    assert (result.status, result.body, result.source_url) == ("not_found", NOT_FOUND, "https://groww.in/large")
