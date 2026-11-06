"""Deterministic, multi-label classification for high-impact company events."""

from __future__ import annotations

import re
from dataclasses import dataclass

from financial_strategy.schemas import COMPANY_EVENT_TYPES, CompanyEventType


@dataclass(frozen=True)
class CompanyEventClassification:
    event_types: tuple[CompanyEventType, ...]
    matched_rules: tuple[str, ...]


_PATTERNS: dict[CompanyEventType, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "earnings": (
        (
            "earnings_results",
            re.compile(r"\b(?:earnings|financial|quarterly|annual)\s+results?\b"),
        ),
        (
            "reports_results",
            re.compile(
                r"\breports?\s+(?:first|second|third|fourth|full)[-\s]quarter\b"
            ),
        ),
    ),
    "guidance": (
        ("guidance", re.compile(r"\b(?:guidance|financial outlook)\b")),
        (
            "forecast_update",
            re.compile(
                r"\b(?:raises?|lowers?|reaffirms?|updates?)\s+"
                r"(?:its\s+)?(?:outlook|forecast|guidance)\b"
            ),
        ),
    ),
    "merger_acquisition": (
        ("merger", re.compile(r"\b(?:merger|merger agreement|business combination)\b")),
        (
            "acquisition",
            re.compile(
                r"\b(?:acquires?|acquisition|definitive agreement to acquire|"
                r"tender offer)\b"
            ),
        ),
    ),
    "buyback": (
        ("share_repurchase", re.compile(r"\b(?:share|stock)\s+repurchase\b")),
        ("buyback", re.compile(r"\b(?:buyback|repurchase authorization|repurchase program)\b")),
    ),
    "dividend": (
        ("cash_dividend", re.compile(r"\b(?:cash|quarterly|special|regular)\s+dividend\b")),
        ("declares_dividend", re.compile(r"\bdeclares?\s+(?:a\s+)?dividend\b")),
    ),
    "financing": (
        (
            "offering",
            re.compile(
                r"\b(?:public offering|private placement|notes offering|debt offering)\b"
            ),
        ),
        (
            "credit",
            re.compile(
                r"\b(?:credit agreement|term loan|debt financing|revolving credit)\b"
            ),
        ),
    ),
    "litigation_regulatory": (
        ("litigation", re.compile(r"\b(?:lawsuit|litigation|legal proceeding|class action)\b")),
        (
            "regulatory",
            re.compile(
                r"\b(?:antitrust|regulatory action|regulatory approval|consent decree|"
                r"subpoena|government investigation|sec investigation)\b"
            ),
        ),
    ),
    "management_change": (
        (
            "appointment",
            re.compile(
                r"\b(?:appoints?|names?)\s+.{0,80}\b"
                r"(?:chief executive|chief financial|president|ceo|cfo)\b"
            ),
        ),
        (
            "departure",
            re.compile(
                r"\b(?:resigns?|retires?|steps down|departure)\b.{0,80}\b"
                r"(?:chief executive|chief financial|president|ceo|cfo)\b"
            ),
        ),
    ),
    "product": (
        (
            "product_launch",
            re.compile(
                r"\b(?:launches?|introduces?|unveils?)\s+"
                r"(?:a\s+|an\s+|new\s+)?(?:product|platform|service|device|drug)\b"
            ),
        ),
        ("product_approval", re.compile(r"\b(?:fda|regulatory)\s+(?:approval|clearance)\b")),
    ),
    "cybersecurity": (
        (
            "cyber_incident",
            re.compile(
                r"\b(?:cybersecurity|cyber security|ransomware)\s+"
                r"(?:incident|attack|breach)\b"
            ),
        ),
        (
            "data_breach",
            re.compile(r"\b(?:data breach|unauthorized access|network intrusion)\b"),
        ),
    ),
}

def classify_company_event(
    *,
    title: str,
    text: str,
    filing_form: str | None,
) -> CompanyEventClassification:
    """Classify one SEC filing or exhibit without network or model calls."""

    normalized_form = (filing_form or "").upper()
    if normalized_form in {"10-Q", "10-K"}:
        return CompanyEventClassification(
            event_types=("earnings",),
            matched_rules=(f"earnings:form_{normalized_form.casefold()}",),
        )

    sample = re.sub(r"\s+", " ", f"{title}\n{text[:25_000]}").casefold()
    matches: dict[CompanyEventType, list[str]] = {}
    for event_type, rules in _PATTERNS.items():
        for rule_name, pattern in rules:
            if pattern.search(sample):
                matches.setdefault(event_type, []).append(
                    f"{event_type}:{rule_name}"
                )

    ordered = tuple(
        event_type
        for event_type in COMPANY_EVENT_TYPES
        if event_type in matches
    )
    return CompanyEventClassification(
        event_types=ordered,
        matched_rules=tuple(
            rule
            for event_type in ordered
            for rule in matches[event_type]
        ),
    )
