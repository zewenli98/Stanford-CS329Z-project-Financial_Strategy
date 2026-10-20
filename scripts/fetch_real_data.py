"""Build an immutable real-data snapshot for 2026-10-01 through 2026-10-14."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
import polars as pl
import yfinance as yf

from financial_strategy.data.calendar import TradingCalendar

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SESSIONS = (
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
    date(2026, 10, 7),
    date(2026, 10, 8),
    date(2026, 10, 9),
    date(2026, 10, 12),
    date(2026, 10, 13),
    date(2026, 10, 14),
)
TARGET_SESSION = EVALUATION_SESSIONS[-1]
EASTERN = ZoneInfo("America/New_York")
DECISION_CUTOFF = datetime(2026, 10, 14, 9, 0, tzinfo=EASTERN)
EARLIEST_DECISION_CUTOFF = datetime(2026, 10, 1, 9, 0, tzinfo=EASTERN)
NEWS_START = EARLIEST_DECISION_CUTOFF - timedelta(days=7)
LOOKBACK_SESSIONS = 60
EXPECTED_SOURCE_SESSION_COUNT = 69
SNAPSHOT_ID = "2026-10-01_2026-10-14_v1"

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "financial-strategy-research/0.1 contact=research@example.com",
)

SECURITIES = {
    "AAPL": {
        "cik": "0000320193",
        "sector": "technology",
        "venue": "XNAS",
        "company_query": '"Apple Inc"',
        "official_domains": {"apple.com"},
    },
    "MSFT": {
        "cik": "0000789019",
        "sector": "technology",
        "venue": "XNAS",
        "company_query": '"Microsoft Corp"',
        "official_domains": {"microsoft.com"},
    },
    "NVDA": {
        "cik": "0001045810",
        "sector": "technology",
        "venue": "XNAS",
        "company_query": '"NVIDIA Corp"',
        "official_domains": {"nvidia.com"},
    },
    "JPM": {
        "cik": "0000019617",
        "sector": "banks",
        "venue": "XNYS",
        "company_query": '"JPMorgan Chase"',
        "official_domains": {"jpmorganchase.com"},
    },
    "BAC": {
        "cik": "0000070858",
        "sector": "banks",
        "venue": "XNYS",
        "company_query": '"Bank of America"',
        "official_domains": {"bankofamerica.com"},
    },
    "XOM": {
        "cik": "0000034088",
        "sector": "energy",
        "venue": "XNYS",
        "company_query": '"Exxon Mobil"',
        "official_domains": {"exxonmobil.com"},
    },
    "CVX": {
        "cik": "0000093410",
        "sector": "energy",
        "venue": "XNYS",
        "company_query": '"Chevron Corp"',
        "official_domains": {"chevron.com"},
    },
    "WMT": {
        "cik": "0000104169",
        "sector": "consumer",
        "venue": "XNYS",
        "company_query": '"Walmart Inc"',
        "official_domains": {"walmart.com"},
    },
    "HD": {
        "cik": "0000354950",
        "sector": "consumer",
        "venue": "XNYS",
        "company_query": '"Home Depot"',
        "official_domains": {"homedepot.com"},
    },
    "GE": {
        "cik": "0000040545",
        "sector": "industrial",
        "venue": "XNYS",
        "company_query": '"GE Aerospace"',
        "official_domains": {"geaerospace.com", "ge.com"},
    },
    "GOOGL": {
        "cik": "0001652044",
        "sector": "communication",
        "venue": "XNAS",
        "company_query": '"Alphabet Inc"',
        "official_domains": {"abc.xyz", "google.com"},
    },
    "META": {
        "cik": "0001326801",
        "sector": "communication",
        "venue": "XNAS",
        "company_query": '"Meta Platforms"',
        "official_domains": {"about.fb.com", "meta.com"},
    },
    "AMZN": {
        "cik": "0001018724",
        "sector": "consumer",
        "venue": "XNAS",
        "company_query": '"Amazon.com Inc"',
        "official_domains": {"aboutamazon.com", "amazon.com"},
    },
    "COST": {
        "cik": "0000909832",
        "sector": "consumer",
        "venue": "XNAS",
        "company_query": '"Costco Wholesale"',
        "official_domains": {"costco.com"},
    },
    "TSLA": {
        "cik": "0001318605",
        "sector": "consumer",
        "venue": "XNAS",
        "company_query": '"Tesla Inc"',
        "official_domains": {"tesla.com"},
    },
    "UNH": {
        "cik": "0000731766",
        "sector": "healthcare",
        "venue": "XNYS",
        "company_query": '"UnitedHealth Group"',
        "official_domains": {"unitedhealthgroup.com"},
    },
    "JNJ": {
        "cik": "0000200406",
        "sector": "healthcare",
        "venue": "XNYS",
        "company_query": '"Johnson & Johnson"',
        "official_domains": {"jnj.com"},
    },
    "GS": {
        "cik": "0000886982",
        "sector": "banks",
        "venue": "XNYS",
        "company_query": '"Goldman Sachs"',
        "official_domains": {"goldmansachs.com"},
    },
    "CAT": {
        "cik": "0000018230",
        "sector": "industrial",
        "venue": "XNYS",
        "company_query": '"Caterpillar Inc"',
        "official_domains": {"caterpillar.com"},
    },
    "COP": {
        "cik": "0001163165",
        "sector": "energy",
        "venue": "XNYS",
        "company_query": '"ConocoPhillips"',
        "official_domains": {"conocophillips.com"},
    },
}

SECONDARY_NEWS_DOMAINS = {
    "reuters.com",
    "businesswire.com",
    "prnewswire.com",
    "cnbc.com",
}

OFFICIAL_NEWS = {
    "GOOGL": [
        {
            "url": (
                "https://blog.google/intl/en-mena/company-news/technology/"
                "gemini-4-argon-our-next-era-of-frontier-intelligence/"
            ),
            "published_at": "2026-10-01T00:00:00-04:00",
            "available_at": "2026-10-02T00:00:00-04:00",
        }
    ],
    "AMZN": [
        {
            "url": (
                "https://www.aboutamazon.com/news/devices/"
                "new-kindle-lineup-2026"
            ),
            "published_at": "2026-10-01T00:00:00-04:00",
            "available_at": "2026-10-02T00:00:00-04:00",
        }
    ],
    "JNJ": [
        {
            "url": (
                "https://www.jnj.com/media-center/press-releases/"
                "new-johnson-johnson-icotyde-icotrokinra-long-term-data-"
                "demonstrate-robust-2-year-results-in-the-treatment-of-adults-"
                "and-adolescents-with-plaque-psoriasis"
            ),
            "published_at": "2026-10-09T00:00:00-04:00",
            "available_at": "2026-10-10T00:00:00-04:00",
        },
        {
            "url": (
                "https://www.jnj.com/media-center/press-releases/"
                "johnson-johnson-reports-q3-2026-results-raises-2026-guidance"
            ),
            "published_at": "2026-10-13T00:00:00-04:00",
            "available_at": "2026-10-14T00:00:00-04:00",
            "event_type": "earnings",
        },
    ],
    "AAPL": [
        {
            "url": (
                "https://www.apple.com/newsroom/2026/10/"
                "apple-unveils-the-new-apple-tv-4k/"
            ),
            "published_at": "2026-10-13T00:00:00-04:00",
            "available_at": "2026-10-14T00:00:00-04:00",
        },
        {
            "url": (
                "https://www.apple.com/newsroom/2026/10/"
                "meet-apple-homeview-a-new-way-to-enjoy-entertainment-and-manage-the-home/"
            ),
            "published_at": "2026-10-13T00:00:00-04:00",
            "available_at": "2026-10-14T00:00:00-04:00",
        },
    ],
    "NVDA": [
        {
            "url": (
                "https://nvidianews.nvidia.com/news/"
                "nvidia-commits-1-billion-to-advance-us-science-over-the-next-five-years"
            ),
            "published_at": "2026-10-08T00:00:00+00:00",
            "available_at": "2026-10-09T00:00:00-04:00",
        }
    ],
    "JPM": [
        {
            "url": (
                "https://www.jpmorganchase.com/newsroom/press-releases/2026/"
                "jpmc-germany-announcement"
            ),
            "published_at": "2026-10-12T06:03:13+00:00",
            "available_at": "2026-10-12T06:03:13+00:00",
        },
        {
            "url": (
                "https://www.jpmorganchase.com/newsroom/press-releases/2026/"
                "philadelphia-office-relocation"
            ),
            "published_at": "2026-10-13T00:00:00-04:00",
            "available_at": "2026-10-14T00:00:00-04:00",
        },
    ],
    "BAC": [
        {
            "url": (
                "https://newsroom.bankofamerica.com/content/newsroom/press-releases/"
                "2026/10/bank-of-america-in-partnership-with-youth-sport-trust-"
                "bring-new-.html"
            ),
            "published_at": "2026-10-13T08:00:00-04:00",
            "available_at": "2026-10-13T08:00:00-04:00",
        }
    ],
    "WMT": [
        {
            "url": (
                "https://corporate.walmart.com/news/2026/10/"
                "walmart-scan-lauch-co-branded-medicare-plan"
            ),
            "published_at": "2026-10-07T00:00:00-04:00",
            "available_at": "2026-10-08T00:00:00-04:00",
        },
        {
            "url": (
                "https://corporate.walmart.com/news/2026/10/"
                "stockton-next-generation-fulfillment-center"
            ),
            "published_at": "2026-10-08T00:00:00-04:00",
            "available_at": "2026-10-09T00:00:00-04:00",
        },
        {
            "url": (
                "https://corporate.walmart.com/news/2026/10/"
                "keeping-affordable-healthcare-within-reach"
            ),
            "published_at": "2026-10-13T00:00:00-04:00",
            "available_at": "2026-10-14T00:00:00-04:00",
        },
    ],
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return normalize_text(" ".join(self.parts))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def content_hash(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in values)
    )


def parse_datetime(value: str, *, default_zone: ZoneInfo = EASTERN) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_zone)
    return parsed.astimezone(UTC)


def parse_gdelt_datetime(value: str) -> datetime:
    for pattern in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            pass
    return parse_datetime(value)


def domain_matches(domain: str, allowed: set[str]) -> bool:
    domain = domain.lower().removeprefix("www.")
    return any(domain == item or domain.endswith(f".{item}") for item in allowed)


def extract_published_at(raw_html: str) -> datetime | None:
    patterns = (
        r'"datePublished"\s*:\s*"([^"]+)"',
        r'property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)',
        r'name=["\']pub_date["\'][^>]*content=["\']([^"\']+)',
        r'<time[^>]*datetime=["\']([^"\']+)',
    )
    for pattern in patterns:
        match = re.search(pattern, raw_html, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return parse_datetime(match.group(1))
        except (ValueError, TypeError):
            continue
    return None


def extract_title(raw_html: str) -> str | None:
    patterns = (
        r'property=["\']og:title["\'][^>]*content=["\']([^"\']+)',
        r"<title[^>]*>(.*?)</title>",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_text(match.group(1))
    return None


def filing_excerpt(text: str) -> str:
    lowered = text.lower()
    for marker in ("item 2.02", "item 7.01", "item 8.01", "item 1.01", "item 9.01"):
        position = lowered.find(marker)
        if position >= 0:
            return text[position : position + 600]
    return text[:600]


def classify_event(title: str, text: str, form: str | None = None, items: str = "") -> str:
    sample = f"{title} {text[:4000]}".lower()
    if form in {"10-Q", "10-K"} or "2.02" in items or any(
        term in sample for term in ("earnings", "quarterly results", "financial results")
    ):
        return "earnings"
    if "guidance" in sample or "outlook" in sample:
        return "guidance"
    if any(term in sample for term in ("merger", "acquisition", "acquire")):
        return "merger_acquisition"
    if any(term in sample for term in ("regulator", "antitrust", "investigation")):
        return "major_regulatory"
    if any(term in sample for term in ("offering", "financing", "credit agreement")):
        return "financing"
    if any(term in sample for term in ("chief executive", "appoints", "resigns")):
        return "management_change"
    return "routine_product_news"


def fetch_market(output: Path, calendar: TradingCalendar, retrieved_at: datetime) -> dict:
    sessions_by_evaluation = {
        evaluation_session: calendar.completed_sessions_before(
            datetime(
                evaluation_session.year,
                evaluation_session.month,
                evaluation_session.day,
                9,
                tzinfo=EASTERN,
            ),
            LOOKBACK_SESSIONS,
        )
        for evaluation_session in EVALUATION_SESSIONS
    }
    sessions = sorted(
        {
            session
            for required_sessions in sessions_by_evaluation.values()
            for session in required_sessions
        }
    )
    if len(sessions) != EXPECTED_SOURCE_SESSION_COUNT:
        raise RuntimeError(
            "Unexpected union of required market sessions: "
            f"{len(sessions)} != {EXPECTED_SOURCE_SESSION_COUNT}"
        )
    expected = set(sessions)
    request_start = min(sessions) - timedelta(days=7)
    request_end = TARGET_SESSION + timedelta(days=1)
    bars: list[dict] = []
    labels: list[dict] = []
    actions: list[dict] = []
    repaired_rows = 0

    for symbol, metadata in SECURITIES.items():
        frame = yf.Ticker(symbol).history(
            start=request_start.isoformat(),
            end=request_end.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=True,
            keepna=True,
        )
        if frame.empty:
            raise RuntimeError(f"Yahoo returned no data for {symbol}")
        by_date = {index.date(): row for index, row in frame.iterrows()}
        missing = sorted(expected - set(by_date))
        if missing:
            raise RuntimeError(f"{symbol} is missing required sessions: {missing}")

        for session in sessions:
            row = by_date[session]
            opened = calendar.session_open(session)
            closed = calendar.session_close(session)
            values = {
                key: float(row[key]) for key in ("Open", "High", "Low", "Close", "Volume")
            }
            if any(value != value for value in values.values()):
                raise RuntimeError(f"{symbol} has NaN market data on {session}")
            if bool(row.get("Repaired?", False)):
                repaired_rows += 1
            bars.append(
                {
                    "symbol": symbol,
                    "permanent_entity_id": f"sec_cik_{metadata['cik']}",
                    "session": session,
                    "open": values["Open"],
                    "high": values["High"],
                    "low": values["Low"],
                    "close": values["Close"],
                    "volume": values["Volume"],
                    "bar_start": opened,
                    "bar_end": closed,
                    "available_at": closed + timedelta(minutes=5),
                    "feed": "yahoo_finance",
                    "timeframe": "1d",
                    "adjustment_policy": "split_adjusted",
                    "bar_status": "traded",
                    "snapshot_id": f"source_market_{SNAPSHOT_ID}",
                }
            )

        for evaluation_session in EVALUATION_SESSIONS:
            if evaluation_session not in by_date:
                raise RuntimeError(
                    f"{symbol} has no evaluation label for {evaluation_session}"
                )
            target = by_date[evaluation_session]
            labels.append(
                {
                    "session": evaluation_session,
                    "symbol": symbol,
                    "open": float(target["Open"]),
                    "close": float(target["Close"]),
                }
            )

        for index, row in frame.iterrows():
            session = index.date()
            if session > TARGET_SESSION:
                continue
            effective_at = calendar.session_open(session)
            split = float(row.get("Stock Splits", 0.0) or 0.0)
            if split:
                actions.append(
                    {
                        "action_id": f"yahoo_{symbol.lower()}_split_{session}",
                        "symbol": symbol,
                        "permanent_entity_id": f"sec_cik_{metadata['cik']}",
                        "action_type": "split" if split >= 1 else "reverse_split",
                        "ratio": split,
                        "effective_at": effective_at,
                        "available_at": effective_at,
                        "source": "yahoo_finance_effective_date",
                    }
                )
            dividend = float(row.get("Dividends", 0.0) or 0.0)
            if dividend:
                actions.append(
                    {
                        "action_id": f"yahoo_{symbol.lower()}_dividend_{session}",
                        "symbol": symbol,
                        "permanent_entity_id": f"sec_cik_{metadata['cik']}",
                        "action_type": "cash_dividend",
                        "ratio": 1.0,
                        "effective_at": effective_at,
                        "available_at": effective_at,
                        "source": "yahoo_finance_effective_date",
                    }
                )

    expected_bar_count = EXPECTED_SOURCE_SESSION_COUNT * len(SECURITIES)
    expected_label_count = len(EVALUATION_SESSIONS) * len(SECURITIES)
    if len(bars) != expected_bar_count:
        raise RuntimeError(f"Unexpected bar count: {len(bars)} != {expected_bar_count}")
    if len(labels) != expected_label_count:
        raise RuntimeError(
            f"Unexpected label count: {len(labels)} != {expected_label_count}"
        )

    market_path = output / "market/daily_bars.parquet"
    market_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(bars).sort(["symbol", "session"]).write_parquet(market_path)
    labels_path = output / "evaluation/labels.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(labels).sort(["session", "symbol"]).write_parquet(labels_path)
    write_jsonl(
        output / "market/corporate_actions.jsonl",
        sorted(actions, key=lambda item: (str(item["effective_at"]), item["action_id"])),
    )
    return {
        "provider": "Yahoo Finance via yfinance",
        "retrieved_at": retrieved_at,
        "package_version": version("yfinance"),
        "request_start": request_start,
        "request_end_exclusive": request_end,
        "auto_adjust": False,
        "repair": True,
        "adjustment_policy": "split_adjusted",
        "evaluation_sessions": list(EVALUATION_SESSIONS),
        "lookback_sessions_per_evaluation": LOOKBACK_SESSIONS,
        "source_session_count": len(sessions),
        "bar_count": len(bars),
        "label_count": len(labels),
        "corporate_action_count": len(actions),
        "repaired_row_count": repaired_rows,
        "availability_policy": "XNYS session close plus five minutes",
    }


def filing_rows(submissions: dict) -> list[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    keys = [
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "primaryDocDescription",
        "items",
    ]
    count = len(recent.get("accessionNumber", []))
    return [
        {key: recent.get(key, [""] * count)[index] for key in keys}
        for index in range(count)
    ]


def fetch_sec_and_news(
    output: Path,
    client: httpx.Client,
    retrieved_at: datetime,
) -> tuple[dict, list[dict]]:
    universe: list[dict] = []
    evidence: list[dict] = []
    source_counts = {"sec": 0, "web_news": 0}
    raw_sec = output / "raw/sec"
    raw_gdelt = output / "raw/gdelt"
    documents = output / "company/documents"
    raw_sec.mkdir(parents=True, exist_ok=True)
    raw_gdelt.mkdir(parents=True, exist_ok=True)
    documents.mkdir(parents=True, exist_ok=True)

    for symbol, metadata in SECURITIES.items():
        submissions_url = f"https://data.sec.gov/submissions/CIK{metadata['cik']}.json"
        response = client.get(submissions_url)
        response.raise_for_status()
        submissions = response.json()
        write_json(raw_sec / f"{symbol.lower()}_submissions.json", submissions)
        universe.append(
            {
                "symbol": symbol,
                "permanent_entity_id": f"sec_cik_{metadata['cik']}",
                "company_name": submissions["name"],
                "cik": metadata["cik"],
                "sic": submissions.get("sic"),
                "sic_description": submissions.get("sicDescription"),
                "sector": metadata["sector"],
                "venue": metadata["venue"],
                "country": "US",
                "security_type": "common_stock",
                "session_profile": "us_regular",
            }
        )
        for filing in filing_rows(submissions):
            if filing["form"] not in {"8-K", "10-Q", "10-K", "6-K"}:
                continue
            accepted = parse_datetime(filing["acceptanceDateTime"])
            if not NEWS_START.astimezone(UTC) <= accepted <= DECISION_CUTOFF.astimezone(UTC):
                continue
            accession = filing["accessionNumber"]
            primary = filing["primaryDocument"]
            archive_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(metadata['cik'])}/{accession.replace('-', '')}/{primary}"
            )
            document_response = client.get(archive_url)
            document_response.raise_for_status()
            extractor = TextExtractor()
            extractor.feed(document_response.text)
            text = extractor.text()
            if len(text) < 200:
                continue
            document_name = f"sec_{symbol.lower()}_{accession.replace('-', '')}.txt"
            (documents / document_name).write_text(text + "\n")
            title = filing["primaryDocDescription"] or f"{filing['form']} filing"
            evidence.append(
                {
                    "available_at": accepted,
                    "content_hash": content_hash(text),
                    "content_path": f"documents/{document_name}",
                    "event_status": "released_before_cutoff",
                    "event_time": accepted,
                    "event_type": classify_event(
                        title, text, filing["form"], filing.get("items", "")
                    ),
                    "evidence_id": f"sec_{symbol.lower()}_{accession.replace('-', '')}",
                    "excerpt": filing_excerpt(text),
                    "first_seen_at": accepted,
                    "published_at": accepted,
                    "retrieved_at": retrieved_at,
                    "scope": "company",
                    "snapshot_id": SNAPSHOT_ID,
                    "source_name": "SEC EDGAR",
                    "source_type": "regulatory_filing",
                    "source_url": archive_url,
                    "symbol": symbol,
                    "timestamp_confidence": "high",
                    "title": title,
                }
            )
            source_counts["sec"] += 1

        query = f"{metadata['company_query']} sourcelang:english"
        gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": 25,
            "format": "json",
            "sort": "datedesc",
            "startdatetime": NEWS_START.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
            "enddatetime": DECISION_CUTOFF.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
        }
        try:
            gdelt_response = client.get(gdelt_url, params=params)
            gdelt_response.raise_for_status()
            gdelt_data = gdelt_response.json()
        except (httpx.HTTPError, ValueError):
            gdelt_data = {"articles": [], "retrieval_error": True}
        write_json(raw_gdelt / f"{symbol.lower()}.json", gdelt_data)
        allowed_domains = set(metadata["official_domains"]) | SECONDARY_NEWS_DOMAINS
        accepted_news = 0
        for article in gdelt_data.get("articles", []):
            if accepted_news >= 5:
                break
            article_url = article.get("url", "")
            domain = urlparse(article_url).netloc or article.get("domain", "")
            if not article_url or not domain_matches(domain, allowed_domains):
                continue
            try:
                seen_at = parse_gdelt_datetime(article["seendate"])
                article_response = client.get(article_url)
                article_response.raise_for_status()
                published_at = extract_published_at(article_response.text)
                if published_at is None:
                    continue
                available_at = max(published_at, seen_at)
                if not (
                    NEWS_START.astimezone(UTC)
                    <= published_at
                    <= available_at
                    <= DECISION_CUTOFF.astimezone(UTC)
                ):
                    continue
                extractor = TextExtractor()
                extractor.feed(article_response.text)
                text = extractor.text()
                if len(text) < 500:
                    continue
            except (httpx.HTTPError, KeyError, ValueError):
                continue
            digest = content_hash(article_url)[:16]
            document_name = f"news_{symbol.lower()}_{digest}.txt"
            (documents / document_name).write_text(text + "\n")
            title = normalize_text(article.get("title", "")) or f"{symbol} company news"
            official = domain_matches(domain, set(metadata["official_domains"]))
            evidence.append(
                {
                    "available_at": available_at,
                    "content_hash": content_hash(text),
                    "content_path": f"documents/{document_name}",
                    "event_status": "released_before_cutoff",
                    "event_time": published_at,
                    "event_type": classify_event(title, text),
                    "evidence_id": f"news_{symbol.lower()}_{digest}",
                    "excerpt": text[:600],
                    "first_seen_at": seen_at,
                    "published_at": published_at,
                    "retrieved_at": retrieved_at,
                    "scope": "company",
                    "snapshot_id": SNAPSHOT_ID,
                    "source_name": domain.lower().removeprefix("www."),
                    "source_type": "company_release" if official else "news_article",
                    "source_url": article_url,
                    "symbol": symbol,
                    "timestamp_confidence": "high" if official else "medium",
                    "title": title,
                }
            )
            source_counts["web_news"] += 1
            accepted_news += 1

        for official_article in OFFICIAL_NEWS.get(symbol, []):
            article_url = official_article["url"]
            published_at = datetime.fromisoformat(official_article["published_at"])
            available_at = datetime.fromisoformat(official_article["available_at"])
            if not (
                NEWS_START <= published_at <= available_at <= DECISION_CUTOFF
            ):
                continue
            try:
                article_response = client.get(article_url)
                article_response.raise_for_status()
                extractor = TextExtractor()
                extractor.feed(article_response.text)
                text = extractor.text()
                if len(text) < 500:
                    continue
            except httpx.HTTPError:
                continue

            digest = content_hash(article_url)[:16]
            document_name = f"official_{symbol.lower()}_{digest}.txt"
            (documents / document_name).write_text(text + "\n")
            company_name = metadata["company_query"].strip('"')
            title = extract_title(article_response.text) or (
                f"{company_name} official announcement"
            )
            domain = urlparse(article_url).netloc.lower().removeprefix("www.")
            evidence.append(
                {
                    "available_at": available_at,
                    "content_hash": content_hash(text),
                    "content_path": f"documents/{document_name}",
                    "event_status": "released_before_cutoff",
                    "event_time": published_at,
                    "event_type": official_article.get(
                        "event_type", "routine_product_news"
                    ),
                    "evidence_id": f"official_{symbol.lower()}_{digest}",
                    "excerpt": text[:600],
                    "first_seen_at": available_at,
                    "published_at": published_at,
                    "retrieved_at": retrieved_at,
                    "scope": "company",
                    "snapshot_id": SNAPSHOT_ID,
                    "source_name": domain,
                    "source_type": "company_release",
                    "source_url": article_url,
                    "symbol": symbol,
                    "timestamp_confidence": (
                        "high" if available_at == published_at else "medium"
                    ),
                    "title": title,
                }
            )
            source_counts["web_news"] += 1

    company_release_symbols = {
        item["symbol"]
        for item in evidence
        if item["source_type"] == "company_release"
    }
    if source_counts["web_news"] < 5 or len(company_release_symbols) < 4:
        raise RuntimeError(
            "Company-news coverage is below the snapshot quality threshold"
        )

    deduplicated = {
        item["content_hash"]: item
        for item in sorted(
            evidence,
            key=lambda value: (value["available_at"], value["evidence_id"]),
        )
    }
    final_evidence = sorted(
        deduplicated.values(), key=lambda item: (item["symbol"], item["available_at"])
    )
    write_json(
        output / "universe/symbols.json",
        {"snapshot_id": f"universe_{SNAPSHOT_ID}", "securities": universe},
    )
    write_jsonl(output / "company/evidence.jsonl", final_evidence)
    return (
        {
            "provider": (
                "SEC EDGAR, verified official newsrooms, and "
                "GDELT-discovered web articles"
            ),
            "retrieved_at": retrieved_at,
            "window_start": NEWS_START,
            "window_end": DECISION_CUTOFF,
            "source_counts": source_counts,
            "evidence_count_after_deduplication": len(final_evidence),
            "gdelt_query_note": (
                "GDELT is discovery only; article timestamps and bodies are revalidated."
            ),
        },
        final_evidence,
    )


def fetch_macro(output: Path, client: httpx.Client, retrieved_at: datetime) -> dict:
    cpi_bls_urls = [
        "https://www.bls.gov/news.release/archives/cpi_10142026.htm",
        "https://www.bls.gov/news.release/cpi.nr0.htm",
    ]
    cpi_bls_response: httpx.Response | None = None
    for url in cpi_bls_urls:
        response = client.get(url)
        if response.status_code == 200:
            cpi_bls_response = response
            break
    if cpi_bls_response is None:
        raise RuntimeError("Unable to retrieve the BLS CPI release")
    extractor = TextExtractor()
    extractor.feed(cpi_bls_response.text)
    cpi_bls_text = extractor.text()
    if "3.7 percent" not in cpi_bls_text or "October 14, 2026" not in cpi_bls_text:
        raise RuntimeError("BLS CPI release content does not match the target event")

    cpi_consensus_url = (
        "https://www.cnbc.com/2026/10/13/"
        "september-consumer-price-index-inflation.html"
    )
    cpi_consensus_response = client.get(cpi_consensus_url)
    cpi_consensus_response.raise_for_status()
    consensus_extractor = TextExtractor()
    consensus_extractor.feed(cpi_consensus_response.text)
    cpi_consensus_text = consensus_extractor.text()
    if (
        "3.6%" not in cpi_consensus_text
        and "3.6 percent" not in cpi_consensus_text
    ):
        raise RuntimeError("Consensus article does not contain the expected 3.6% forecast")

    payroll_bls_urls = [
        "https://www.bls.gov/news.release/archives/empsit_10022026.htm",
        "https://www.bls.gov/news.release/empsit.nr0.htm",
    ]
    payroll_bls_response: httpx.Response | None = None
    for url in payroll_bls_urls:
        response = client.get(url)
        if response.status_code == 200:
            payroll_bls_response = response
            break
    if payroll_bls_response is None:
        raise RuntimeError("Unable to retrieve the BLS employment release")
    payroll_extractor = TextExtractor()
    payroll_extractor.feed(payroll_bls_response.text)
    payroll_bls_text = payroll_extractor.text()
    if "29,000" not in payroll_bls_text or "October 2, 2026" not in payroll_bls_text:
        raise RuntimeError("BLS employment release does not match the target event")

    payroll_consensus_url = (
        "https://www.cnbc.com/2026/10/01/"
        "the-september-jobs-report-will-be-released-friday-"
        "heres-what-to-expect.html"
    )
    payroll_consensus_response = client.get(payroll_consensus_url)
    payroll_consensus_response.raise_for_status()
    payroll_consensus_extractor = TextExtractor()
    payroll_consensus_extractor.feed(payroll_consensus_response.text)
    payroll_consensus_text = payroll_consensus_extractor.text()
    if "84,000" not in payroll_consensus_text:
        raise RuntimeError("Payroll consensus article does not contain the 84,000 forecast")

    macro_documents = output / "macro/documents"
    macro_documents.mkdir(parents=True, exist_ok=True)
    (macro_documents / "bls_cpi_2026-10-14.txt").write_text(cpi_bls_text + "\n")
    (macro_documents / "cnbc_cpi_consensus_2026-10-13.txt").write_text(
        cpi_consensus_text + "\n"
    )
    (macro_documents / "bls_payrolls_2026-10-02.txt").write_text(
        payroll_bls_text + "\n"
    )
    (macro_documents / "cnbc_payrolls_consensus_2026-10-01.txt").write_text(
        payroll_consensus_text + "\n"
    )
    cpi_event_time = datetime(2026, 10, 14, 8, 30, tzinfo=EASTERN).astimezone(UTC)
    cpi_consensus_published_at = datetime(
        2026, 10, 13, 14, 5, tzinfo=EASTERN
    ).astimezone(UTC)
    payroll_event_time = datetime(
        2026, 10, 2, 8, 30, tzinfo=EASTERN
    ).astimezone(UTC)
    payroll_consensus_published_at = datetime(
        2026, 10, 1, 14, 55, tzinfo=EASTERN
    ).astimezone(UTC)
    cpi_record = {
        "actual": 3.7,
        "affected_risk_factors": ["inflation", "interest_rates"],
        "available_at": cpi_event_time,
        "consensus": 3.6,
        "consensus_content_hash": content_hash(cpi_consensus_text),
        "consensus_published_at": cpi_consensus_published_at,
        "consensus_source_name": "Dow Jones survey via CNBC",
        "consensus_source_url": cpi_consensus_url,
        "content_hash": content_hash(cpi_bls_text),
        "content_path": "documents/bls_cpi_2026-10-14.txt",
        "event_status": "released_before_cutoff",
        "event_time": cpi_event_time,
        "event_type": "cpi_yoy",
        "evidence_id": "macro_bls_cpi_2026_09",
        "excerpt": (
            "September 2026 CPI-U increased 3.7% year over year versus "
            "a 3.6% Dow Jones consensus available before release."
        ),
        "first_seen_at": cpi_event_time,
        "published_at": cpi_event_time,
        "retrieved_at": retrieved_at,
        "scope": "macro",
        "snapshot_id": SNAPSHOT_ID,
        "source_name": "U.S. Bureau of Labor Statistics",
        "source_type": "official_release",
        "source_url": str(cpi_bls_response.url),
        "status": "released_before_cutoff",
        "symbol": None,
        "timestamp_confidence": "high",
        "title": "Consumer Price Index - September 2026",
        "unit": "percentage_points",
    }
    payroll_record = {
        "actual": 29.0,
        "affected_risk_factors": ["interest_rates", "usd"],
        "available_at": payroll_event_time,
        "consensus": 84.0,
        "consensus_content_hash": content_hash(payroll_consensus_text),
        "consensus_published_at": payroll_consensus_published_at,
        "consensus_source_name": "Dow Jones survey via CNBC",
        "consensus_source_url": payroll_consensus_url,
        "content_hash": content_hash(payroll_bls_text),
        "content_path": "documents/bls_payrolls_2026-10-02.txt",
        "event_status": "released_before_cutoff",
        "event_time": payroll_event_time,
        "event_type": "nonfarm_payrolls",
        "evidence_id": "macro_bls_payrolls_2026_09",
        "excerpt": (
            "September 2026 nonfarm payrolls increased by 29,000 versus "
            "an 84,000 Dow Jones consensus available before release."
        ),
        "first_seen_at": payroll_event_time,
        "published_at": payroll_event_time,
        "retrieved_at": retrieved_at,
        "scope": "macro",
        "snapshot_id": SNAPSHOT_ID,
        "source_name": "U.S. Bureau of Labor Statistics",
        "source_type": "official_release",
        "source_url": str(payroll_bls_response.url),
        "status": "released_before_cutoff",
        "symbol": None,
        "timestamp_confidence": "high",
        "title": "The Employment Situation - September 2026",
        "unit": "thousands_of_jobs",
    }
    records = sorted(
        [payroll_record, cpi_record],
        key=lambda item: str(item["available_at"]),
    )
    write_jsonl(output / "macro/events.jsonl", records)
    return {
        "events": [
            {
                "event_type": item["event_type"],
                "actual": item["actual"],
                "consensus": item["consensus"],
                "event_time": item["event_time"],
                "consensus_published_at": item["consensus_published_at"],
                "actual_source_url": item["source_url"],
                "consensus_source_url": item["consensus_source_url"],
            }
            for item in records
        ],
        "retrieved_at": retrieved_at,
    }


def build_snapshot(final_path: Path) -> None:
    if final_path.exists():
        raise FileExistsError(f"immutable snapshot already exists: {final_path}")
    temporary = final_path.parent / f".{final_path.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    retrieved_at = datetime.now(UTC)
    try:
        calendar = TradingCalendar("XNYS")
        market_provenance = fetch_market(temporary, calendar, retrieved_at)
        headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        }
        with httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(45.0),
        ) as client:
            company_provenance, _evidence = fetch_sec_and_news(
                temporary, client, retrieved_at
            )
            macro_provenance = fetch_macro(temporary, client, retrieved_at)
        files = {}
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(temporary))] = {
                    "sha256": content_hash(path.read_bytes()),
                    "size_bytes": path.stat().st_size,
                }
        write_json(
            temporary / "provenance.json",
            {
                "snapshot_id": SNAPSHOT_ID,
                "evaluation_sessions": EVALUATION_SESSIONS,
                "evaluation_start": EVALUATION_SESSIONS[0],
                "evaluation_end": EVALUATION_SESSIONS[-1],
                "target_session": TARGET_SESSION,
                "decision_cutoff": DECISION_CUTOFF,
                "created_at": retrieved_at,
                "market": market_provenance,
                "company": company_provenance,
                "macro": macro_provenance,
                "limitations": [
                    "Yahoo Finance is an unofficial source and may revise historical data.",
                    "Yahoo OHLC is split-adjusted; the gateway must not adjust it again.",
                    "Corporate-action availability is conservatively set to effective time.",
                    (
                        "GDELT is used only for discovery; accepted articles are "
                        "independently validated."
                    ),
                    (
                        "Official pages with date-only publication metadata are "
                        "conservatively treated as available at 00:00 ET the next day."
                    ),
                    (
                        "This ten-session snapshot is an integration example, "
                        "not a performance study."
                    ),
                ],
                "files": files,
            },
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/2026-10-01_2026-10-14",
    )
    args = parser.parse_args()
    build_snapshot(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
