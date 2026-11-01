"""Turn public HTML into labelled sections for later chunking."""

from __future__ import annotations

import io
import json
import logging
import re
from typing import Any

from bs4 import BeautifulSoup, Tag

from src.models import Section

DROP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"}
HEADINGS = {"h1", "h2", "h3", "h4"}
NOISE_HEADINGS = {
    "return calculator",
    "cookie",
    "login",
    "download app",
    "holdings",
    "page",
}
USEFUL_DOM_HEADINGS = (
    "exit load",
    "minimum investment",
    "investment objective",
    "understand terms",
    "fund house",
    "lock",
    "expense",
    "riskometer",
    "benchmark",
    "about ",
    "sip",
)

def extract_sections(html: str) -> list[Section]:
    soup = BeautifulSoup(html, "lxml")
    sections: list[Section] = []
    next_data = _parse_next_data(soup)
    if next_data:
        sections.extend(_sections_from_next_data(next_data))
    _strip_noise(soup)
    dom_sections = _sections_from_dom(soup)
    if sections:
        dom_sections = [item for item in dom_sections if _useful_dom_heading(item.heading)]
    sections.extend(dom_sections)
    return _dedupe_sections(sections)


def extract_pdf_sections(data: bytes, pages: list[int] | None = None) -> list[Section]:
    """One section per PDF page ("Page N"); `pages` keeps only those 1-based pages."""
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(io.BytesIO(data))
    wanted = set(pages or range(1, len(reader.pages) + 1))
    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        if number not in wanted:
            continue
        lines = [_clean_text(line) for line in (page.extract_text() or "").splitlines()]
        text = "\n".join(line for line in lines if line)
        if text:
            sections.append(Section(heading=f"Page {number}", text=text))
    return _dedupe_sections(sections)


def sections_to_text(sections: list[Section]) -> str:
    blocks = [f"## {item.heading}\n\n{item.text.strip()}" for item in sections if item.text.strip()]
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def parse_sections_file(text: str) -> list[Section]:
    sections: list[Section] = []
    heading = "Document"
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if lines:
                sections.append(Section(heading=heading, text="\n".join(lines).strip()))
            heading = line[3:].strip() or "Document"
            lines = []
        else:
            lines.append(line)
    if lines:
        sections.append(Section(heading=heading, text="\n".join(lines).strip()))
    return [item for item in sections if item.text]


def _parse_next_data(soup: BeautifulSoup) -> dict[str, Any] | None:
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except json.JSONDecodeError:
        return None


def _sections_from_next_data(payload: dict[str, Any]) -> list[Section]:
    page = (payload.get("props") or {}).get("pageProps") or {}
    sections: list[Section] = []
    mf = page.get("mfServerSideData")
    if isinstance(mf, dict):
        facts = _scheme_key_facts(mf)
        if facts:
            sections.append(Section(heading="Key facts", text=facts))
        about = _clean_text(str(mf.get("description") or ""))
        if about:
            name = mf.get("scheme_name") or "Scheme"
            sections.append(Section(heading=f"About {name}", text=about))
        exit_history = _exit_load_history(mf.get("historic_exit_loads"))
        if exit_history:
            sections.append(Section(heading="Exit load history", text=exit_history))
    amc_data = page.get("amcMainPageData")
    if isinstance(amc_data, list):
        sections.extend(_amc_sections(amc_data))
    help_answer = _help_answer(page)
    if help_answer:
        sections.append(help_answer)
    return sections


def _help_answer(page: dict[str, Any]) -> Section | None:
    """Groww help pages: the answer for pageProps.questionId lives in initialLandingData."""
    question_id = page.get("questionId")
    if not question_id:
        return None
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if found:
            return
        if isinstance(node, dict):
            if node.get("questionId") == question_id and node.get("answerHtml"):
                found.append(node)
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(page.get("initialLandingData"))
    if not found:
        return None
    body = _html_to_text(str(found[0]["answerHtml"]))
    title = _clean_text(str(found[0].get("questionTitle") or "Help"))
    return Section(heading=title, text=body) if body else None


def _scheme_key_facts(mf: dict[str, Any]) -> str:
    risk = None
    stats = mf.get("return_stats")
    if isinstance(stats, list) and stats and isinstance(stats[0], dict):
        risk = stats[0].get("risk")
    rows = [
        ("Scheme name", mf.get("scheme_name")),
        ("Fund house", mf.get("fund_house")),
        ("Category", mf.get("category")),
        ("Sub-category", mf.get("sub_category")),
        ("Expense ratio", _percent(mf.get("expense_ratio"))),
        ("Exit load", mf.get("exit_load")),
        ("Minimum SIP", _money(mf.get("min_sip_investment"))),
        ("Minimum lumpsum", _money(mf.get("min_investment_amount"))),
        ("Lock-in", _lock_in(mf.get("lock_in"))),
        ("Riskometer", risk or mf.get("nfo_risk")),
        ("Benchmark", mf.get("benchmark_name") or mf.get("benchmark")),
        ("AUM", _crore(mf.get("aum"))),
        ("Launch date", mf.get("launch_date")),
        ("Fund manager", _fund_manager(mf.get("fund_manager"))),
        ("SIP allowed", mf.get("sip_allowed")),
    ]
    lines = [f"{label}: {value}" for label, value in rows if _has_value(value)]
    return "\n".join(lines)


def _exit_load_history(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    lines = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        note = item.get("note") or item.get("exit_load")
        date = (item.get("as_on_date") or "")[:10]
        if note:
            lines.append(f"{date}: {note}" if date else str(note))
    return "\n".join(lines)


def _amc_sections(amc_data: list[Any]) -> list[Section]:
    sections: list[Section] = []
    for block in amc_data:
        if not isinstance(block, dict):
            continue
        info = block.get("key_information")
        if isinstance(info, dict):
            lines = []
            labels = {
                "fund_house": "Fund house",
                "asset_management_company": "Asset management company",
                "amc_setup_date": "AMC setup date",
                "sponsor_name": "Sponsor",
                "trustee_organisation": "Trustee",
                "total_aum": "Total AUM",
                "website": "Website",
                "address": "Address",
                "phone": "Phone",
                "ceo": "CEO",
                "cio": "CIO",
            }
            for key, label in labels.items():
                value = info.get(key)
                if _has_value(value):
                    if key == "total_aum":
                        value = _crore(value)
                    elif isinstance(value, list):
                        value = ", ".join(str(item) for item in value)
                    lines.append(f"{label}: {value}")
            if lines:
                sections.append(Section(heading="AMC key information", text="\n".join(lines)))
        for content in block.get("contentList") or block.get("content_list") or []:
            if not isinstance(content, dict):
                continue
            title = _clean_text(str(content.get("title") or "AMC overview"))
            body = _html_to_text(str(content.get("content") or ""))
            if body:
                sections.append(Section(heading=title, text=body))
    return sections


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()


def _sections_from_dom(soup: BeautifulSoup) -> list[Section]:
    body = soup.body or soup
    sections: list[Section] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        text = _clean_text("\n".join(buf))
        if text and heading and not _is_noise_heading(heading):
            sections.append(Section(heading=heading, text=text))
        buf.clear()

    for node in body.descendants:
        if not isinstance(node, Tag):
            continue
        name = node.name.lower() if node.name else ""
        if name in HEADINGS:
            flush()
            heading = _clean_text(node.get_text(" ", strip=True))
            continue
        if not heading:
            continue
        if name == "dl":
            buf.extend(_definition_lines(node))
            continue
        if name == "table":
            buf.extend(_table_lines(node))
            continue
        if name in {"p", "li"}:
            line = _clean_text(node.get_text(" ", strip=True))
            if line:
                buf.append(line)
    flush()
    return sections


def _is_noise_heading(heading: str) -> bool:
    lowered = heading.lower()
    if lowered in NOISE_HEADINGS:
        return True
    return "view details" in lowered or "screener" in lowered


def _useful_dom_heading(heading: str) -> bool:
    lowered = heading.lower()
    return any(token in lowered for token in USEFUL_DOM_HEADINGS)


def _definition_lines(dl: Tag) -> list[str]:
    lines: list[str] = []
    dts = dl.find_all("dt")
    for dt in dts:
        dd = dt.find_next_sibling("dd")
        if dd:
            lines.append(f"{_clean_text(dt.get_text())}: {_clean_text(dd.get_text())}")
    return lines


def _table_lines(table: Tag) -> list[str]:
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if len(cells) == 2:
            lines.append(f"{cells[0]}: {cells[1]}")
    return lines


def _html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    return _clean_text(soup.get_text("\n", strip=True))


def _dedupe_sections(sections: list[Section]) -> list[Section]:
    seen: set[tuple[str, str]] = set()
    unique: list[Section] = []
    for item in sections:
        text = _clean_text(item.text)
        if len(text) < 20:
            continue
        key = (item.heading.strip().lower(), text.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(Section(heading=item.heading.strip() or "Page", text=text))
    return unique


def _lock_in(value: Any) -> str:
    if not isinstance(value, dict):
        return "None (no lock-in stated)"
    parts = []
    for key, label in (("years", "year(s)"), ("months", "month(s)"), ("days", "day(s)")):
        amount = value.get(key)
        if isinstance(amount, (int, float)) and amount:
            parts.append(f"{amount:g} {label}")
    return " ".join(parts) if parts else "None (no lock-in stated)"


def _fund_manager(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.append(str(item.get("name") or item.get("fund_manager") or ""))
        return ", ".join(name for name in names if name) or None
    return str(value)


def _percent(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return f"{value:g}%"
    text = str(value)
    return text if text.endswith("%") else f"{text}%"


def _money(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return f"₹{value:g}"
    return str(value)


def _crore(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return f"₹{value:,.2f} Cr"
    return str(value)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
