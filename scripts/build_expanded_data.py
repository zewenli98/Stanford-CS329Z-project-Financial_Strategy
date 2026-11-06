"""Build a configuration-driven, immutable public-data research snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl
import yaml
import yfinance as yf

from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.data.company_events import classify_company_event
from financial_strategy.schemas import (
    CorporateActionRecord,
    EvidenceChunk,
    EvidenceRecord,
    MarketBar,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEC_USER_AGENT = (
    "financial-strategy-research/0.1 contact=research@example.com"
)
SECTOR_MAP = {
    "Communication Services": "communication",
    "Consumer Discretionary": "consumer",
    "Consumer Staples": "consumer",
    "Energy": "energy",
    "Financials": "banks",
    "Health Care": "healthcare",
    "Industrials": "industrial",
    "Information Technology": "technology",
    "Materials": "industrial",
    "Real Estate": "industrial",
    "Utilities": "industrial",
}


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    required = {
        "snapshot_id",
        "output_path",
        "evaluation_start",
        "evaluation_end",
        "download_start",
        "warmup_sessions",
        "target_universe_size",
        "candidate_universe_size",
        "minimum_session_completeness",
        "top_k",
        "deep_research_candidates",
        "development_fraction",
        "universe_source",
        "market_source",
        "company_source",
        "reproducibility",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("expansion config has missing or unknown keys")
    if payload["target_universe_size"] < 100:
        raise ValueError("expanded universe must contain at least 100 symbols")
    if payload["deep_research_candidates"] <= payload["top_k"]:
        raise ValueError("deep research candidate count must exceed top_k")
    return payload


def fetch_constituents(config: dict, client: httpx.Client) -> list[dict]:
    response = client.get(config["universe_source"]["url"])
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    result = []
    for row in rows:
        symbol = row["Symbol"].strip()
        cik = str(row.get("CIK", "")).strip().zfill(10)
        sector = SECTOR_MAP.get(row.get("GICS Sector", ""))
        if not symbol or not cik or sector is None:
            continue
        result.append(
            {
                "symbol": symbol,
                "yahoo_symbol": symbol.replace(".", "-"),
                "company_name": row.get("Security", symbol).strip(),
                "cik": cik,
                "permanent_entity_id": f"sec_cik_{cik}",
                "sector": sector,
                "gics_sector": row.get("GICS Sector"),
                "gics_sub_industry": row.get("GICS Sub-Industry"),
            }
        )
    result.sort(key=lambda item: item["symbol"])
    return result


def download_market(config: dict, securities: list[dict]) -> dict[str, pl.DataFrame]:
    start = config["download_start"]
    end = (date.fromisoformat(config["evaluation_end"]) + timedelta(days=1)).isoformat()
    chunk_size = int(config["market_source"]["chunk_size"])
    frames: dict[str, pl.DataFrame] = {}
    by_yahoo = {item["yahoo_symbol"]: item["symbol"] for item in securities}
    yahoo_symbols = sorted(by_yahoo)
    for offset in range(0, len(yahoo_symbols), chunk_size):
        chunk = yahoo_symbols[offset : offset + chunk_size]
        downloaded = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        for yahoo_symbol in chunk:
            frame = _extract_ticker_frame(downloaded, yahoo_symbol, len(chunk))
            if frame is None or frame.empty:
                continue
            frame = frame.reset_index()
            frame.columns = [str(value).lower().replace(" ", "_") for value in frame.columns]
            if "date" not in frame.columns:
                frame = frame.rename(columns={frame.columns[0]: "date"})
            records = []
            for row in frame.to_dict("records"):
                if any(
                    not math.isfinite(float(row.get(field, float("nan"))))
                    for field in ("open", "high", "low", "close")
                ):
                    continue
                records.append(
                    {
                        "session": row["date"].date(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume") or 0),
                        "stock_splits": float(row.get("stock_splits") or 0),
                    }
                )
            if records:
                frames[by_yahoo[yahoo_symbol]] = pl.DataFrame(records)
    return frames


def _extract_ticker_frame(downloaded: object, symbol: str, chunk_size: int):
    if getattr(downloaded, "empty", True):
        return None
    if chunk_size == 1:
        return downloaded
    try:
        return downloaded[symbol]
    except KeyError:
        return None


def select_universe(
    config: dict,
    securities: list[dict],
    frames: dict[str, pl.DataFrame],
    expected_sessions: list[date],
) -> tuple[list[dict], list[dict]]:
    warmup = expected_sessions[: int(config["warmup_sessions"])]
    required = set(warmup)
    scored = []
    excluded = []
    by_symbol = {item["symbol"]: item for item in securities}
    for symbol, security in by_symbol.items():
        frame = frames.get(symbol)
        if frame is None:
            excluded.append({"symbol": symbol, "reason": "NO_MARKET_DATA"})
            continue
        warmup_frame = frame.filter(pl.col("session").is_in(warmup))
        present = set(warmup_frame["session"].to_list())
        completeness = len(present & required) / len(required)
        valid = warmup_frame.filter(
            (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
            & (pl.col("volume") > 0)
        )
        if completeness < float(config["minimum_session_completeness"]):
            excluded.append(
                {
                    "symbol": symbol,
                    "reason": "INSUFFICIENT_WARMUP_COMPLETENESS",
                    "completeness": completeness,
                }
            )
            continue
        if valid.height != warmup_frame.height:
            excluded.append({"symbol": symbol, "reason": "INVALID_WARMUP_OHLCV"})
            continue
        median_dollar_volume = float(
            valid.select((pl.col("close") * pl.col("volume")).median()).item()
        )
        scored.append((median_dollar_volume, symbol, security))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [
        {**security, "warmup_median_dollar_volume": liquidity}
        for liquidity, _symbol, security in scored[: config["target_universe_size"]]
    ]
    selected_symbols = {item["symbol"] for item in selected}
    excluded.extend(
        {"symbol": symbol, "reason": "BELOW_LIQUIDITY_CUTOFF"}
        for _liquidity, symbol, _security in scored
        if symbol not in selected_symbols
    )
    if len(selected) != config["target_universe_size"]:
        raise RuntimeError(
            f"only {len(selected)} symbols passed quality gates; "
            f"{config['target_universe_size']} required"
        )
    return selected, sorted(excluded, key=lambda item: item["symbol"])


def materialize_market(
    config: dict,
    selected: list[dict],
    frames: dict[str, pl.DataFrame],
    calendar: TradingCalendar,
) -> tuple[pl.DataFrame, list[CorporateActionRecord]]:
    schedule = {
        date.fromisoformat(item["session"]): item
        for item in calendar.schedule_snapshot(
            date.fromisoformat(config["download_start"]),
            date.fromisoformat(config["evaluation_end"]),
        )
    }
    bars = []
    actions: list[CorporateActionRecord] = []
    for security in selected:
        symbol = security["symbol"]
        for row in frames[symbol].to_dicts():
            session = row["session"]
            if session not in schedule or row["volume"] <= 0:
                continue
            opened = datetime.fromisoformat(str(schedule[session]["market_open"]))
            closed = datetime.fromisoformat(str(schedule[session]["market_close"]))
            available = closed + timedelta(minutes=15)
            bar = MarketBar(
                symbol=symbol,
                permanent_entity_id=security["permanent_entity_id"],
                session=session,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                bar_start=opened,
                bar_end=closed,
                available_at=available,
                feed="yfinance_public",
                adjustment_policy="raw_unadjusted",
                snapshot_id=config["snapshot_id"],
            )
            bars.append(bar.model_dump())
            ratio = row.get("stock_splits", 0)
            if ratio and ratio > 0:
                action_type = "split" if ratio >= 1 else "reverse_split"
                actions.append(
                    CorporateActionRecord(
                        action_id=f"{symbol}_{session.isoformat()}_{action_type}",
                        symbol=symbol,
                        permanent_entity_id=security["permanent_entity_id"],
                        action_type=action_type,
                        ratio=ratio,
                        effective_at=opened,
                        available_at=opened,
                        source="yfinance_public",
                    )
                )
    return pl.DataFrame(bars).sort(["session", "symbol"]), actions


def fetch_sec_evidence(
    config: dict,
    selected: list[dict],
    client: httpx.Client,
    retrieved_at: datetime,
) -> tuple[list[EvidenceRecord], dict[str, str]]:
    start = date.fromisoformat(config["evaluation_start"]) - timedelta(days=7)
    end = date.fromisoformat(config["evaluation_end"])
    forms = set(config["company_source"]["forms"])
    delay = 1.0 / float(config["company_source"]["max_requests_per_second"])
    result: list[EvidenceRecord] = []
    documents: dict[str, str] = {}
    for security in selected:
        response = client.get(
            f"https://data.sec.gov/submissions/CIK{security['cik']}.json"
        )
        response.raise_for_status()
        payload = response.json()
        recent = payload.get("filings", {}).get("recent", {})
        columns = (
            "accessionNumber",
            "acceptanceDateTime",
            "filingDate",
            "form",
            "primaryDocument",
            "primaryDocDescription",
        )
        count = len(recent.get("accessionNumber", []))
        for index in range(count):
            item = {column: recent.get(column, [None] * count)[index] for column in columns}
            if item["form"] not in forms or not item["acceptanceDateTime"]:
                continue
            accepted = datetime.fromisoformat(
                item["acceptanceDateTime"].replace("Z", "+00:00")
            )
            if accepted.tzinfo is None:
                accepted = accepted.replace(tzinfo=ZoneInfo("America/New_York"))
            if not start <= accepted.date() <= end:
                continue
            accession = item["accessionNumber"]
            accession_compact = accession.replace("-", "")
            directory_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(security['cik'])}/"
                f"{accession_compact}"
            )
            document_names = [item["primaryDocument"]]
            try:
                index_response = client.get(f"{directory_url}/index.json")
                index_response.raise_for_status()
                names = [
                    entry.get("name", "")
                    for entry in index_response.json().get("directory", {}).get("item", [])
                ]
                document_names.extend(
                    name
                    for name in names
                    if re.search(r"(?:^|[-_])ex(?:hibit)?[-_]?99", name, re.I)
                )
            except (httpx.HTTPError, ValueError):
                # The primary filing remains authoritative when an archive index
                # is temporarily unavailable; missing exhibits are reported later.
                pass
            for document_index, document_name in enumerate(dict.fromkeys(document_names)):
                url = f"{directory_url}/{document_name}"
                document_response = client.get(url)
                document_response.raise_for_status()
                text = _normalize_document(document_response.text)
                digest = hashlib.sha256(text.encode()).hexdigest()
                suffix = "primary" if document_index == 0 else f"exhibit_{document_index}"
                evidence_id = f"sec_{security['symbol']}_{accession_compact}_{suffix}"
                relative_path = f"documents/{evidence_id}.txt"
                documents[relative_path] = text
                chunks = _document_chunks(evidence_id, text)
                excerpt = text[:1_000] or (
                    f"SEC {item['form']} accepted {accepted.isoformat()}."
                )
                title = (
                    f"{security['company_name']} {item['form']} "
                    f"{'primary filing' if document_index == 0 else 'EX-99 exhibit'}"
                )
                classification = classify_company_event(
                    title=title,
                    text=text,
                    filing_form=item["form"],
                )
                result.append(EvidenceRecord(
                    evidence_id=evidence_id,
                    scope="company",
                    symbol=security["symbol"],
                    source_type=(
                        "company_release"
                        if document_index > 0
                        else "sec_filing"
                    ),
                    source_name="SEC EDGAR submissions",
                    source_url=url,
                    title=title,
                    excerpt=excerpt,
                    event_time=accepted,
                    published_at=accepted,
                    first_seen_at=accepted,
                    retrieved_at=retrieved_at,
                    available_at=accepted,
                    content_hash=digest,
                    snapshot_id=config["snapshot_id"],
                    timestamp_confidence="high",
                    event_types=list(classification.event_types),
                    event_status="released_before_cutoff",
                    schema_version=1,
                    content_path=relative_path,
                    document_content_hash=digest,
                    accession_number=accession,
                    filing_form=item["form"],
                    exhibit_type=None if document_index == 0 else "EX-99",
                    chunks=chunks,
                ))
        time.sleep(delay)
    result.sort(key=lambda item: (item.available_at, item.evidence_id))
    return result, documents


def _normalize_document(value: str) -> str:
    """Convert SEC HTML to stable text while preserving table cell boundaries."""

    without_scripts = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
        " ",
        value,
        flags=re.I | re.S,
    )
    with_breaks = re.sub(
        r"</?(?:p|div|tr|li|br|h[1-6])\b[^>]*>",
        "\n",
        without_scripts,
        flags=re.I,
    )
    text = re.sub(r"<[^>]+>", " ", with_breaks)
    text = html.unescape(text).replace("\xa0", " ")
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


def _document_chunks(
    evidence_id: str,
    text: str,
    maximum_chars: int = 4_000,
    overlap_chars: int = 400,
) -> list[EvidenceChunk]:
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + maximum_chars)
        content = text[start:end]
        chunks.append(
            EvidenceChunk(
                chunk_id=f"{evidence_id}_chunk_{index:04}",
                start_char=start,
                end_char=end,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
            )
        )
        if end == len(text):
            break
        start = end - overlap_chars
        index += 1
    return chunks


def build_snapshot(config_path: Path) -> Path:
    config = load_config(config_path)
    final_path = ROOT / config["output_path"]
    if final_path.exists() and not config["reproducibility"]["allow_overwrite"]:
        raise FileExistsError(f"snapshot already exists: {final_path}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config['snapshot_id']}.", dir=final_path.parent)
    )
    retrieved_at = datetime.now(UTC)
    sec_user_agent = os.environ.get(
        config["company_source"]["user_agent_env"],
        DEFAULT_SEC_USER_AGENT,
    )
    headers = {"User-Agent": sec_user_agent, "Accept-Encoding": "gzip, deflate"}
    try:
        with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
            constituents = fetch_constituents(config, client)
            candidates = constituents[: int(config["candidate_universe_size"])]
            frames = download_market(config, candidates)
            calendar = TradingCalendar()
            evaluation_sessions = [
                date.fromisoformat(item["session"])
                for item in calendar.schedule_snapshot(
                    date.fromisoformat(config["evaluation_start"]),
                    date.fromisoformat(config["evaluation_end"]),
                )
            ]
            warmup_sessions = [
                date.fromisoformat(item["session"])
                for item in calendar.schedule_snapshot(
                    date.fromisoformat(config["download_start"]),
                    date.fromisoformat(config["evaluation_start"]) - timedelta(days=1),
                )
            ][-int(config["warmup_sessions"]) :]
            selected, excluded = select_universe(
                config,
                candidates,
                frames,
                warmup_sessions,
            )
            bars, actions = materialize_market(
                config,
                selected,
                frames,
                calendar,
            )
            evidence, documents = fetch_sec_evidence(
                config,
                selected,
                client,
                retrieved_at,
            )

        for directory in (
            "universe",
            "market",
            "company/documents",
            "macro/documents",
            "evaluation",
        ):
            (temporary / directory).mkdir(parents=True, exist_ok=True)
        universe = {
            "snapshot_id": config["snapshot_id"],
            "as_of": retrieved_at.isoformat(),
            "membership_policy": config["universe_source"]["membership_policy"],
            "securities": [
                {
                    "symbol": item["symbol"],
                    "company_name": item["company_name"],
                    "cik": item["cik"],
                    "permanent_entity_id": item["permanent_entity_id"],
                    "sector": item["sector"],
                    "security_type": "common_stock",
                    "country": "US",
                    "venue": "US",
                    "session_profile": "us_regular",
                }
                for item in selected
            ],
        }
        _write_json(temporary / "universe/symbols.json", universe)
        _write_json(temporary / "universe/exclusions.json", excluded)
        bars.write_parquet(temporary / "market/daily_bars.parquet")
        _write_jsonl(
            temporary / "market/corporate_actions.jsonl",
            [item.model_dump(mode="json") for item in actions],
        )
        _write_jsonl(
            temporary / "company/evidence.jsonl",
            [item.model_dump(mode="json") for item in evidence],
        )
        for relative_path, text in documents.items():
            document_path = temporary / "company" / relative_path
            document_path.parent.mkdir(parents=True, exist_ok=True)
            document_path.write_text(text)
        _write_jsonl(temporary / "macro/events.jsonl", [])
        labels = bars.filter(
            pl.col("session") >= date.fromisoformat(config["evaluation_start"])
        ).select(["session", "symbol", "open", "high", "low", "close"])
        labels.write_parquet(temporary / "evaluation/labels.parquet")
        split_at = int(len(evaluation_sessions) * float(config["development_fraction"]))
        train_sessions = evaluation_sessions[:split_at]
        validation_sessions = evaluation_sessions[split_at:]
        split = {
            "snapshot_id": config["snapshot_id"],
            "strategy": "chronological_holdout",
            "train_sessions": [item.isoformat() for item in train_sessions],
            "validation_sessions": [item.isoformat() for item in validation_sessions],
            "train_fraction": len(train_sessions) / len(evaluation_sessions),
            "validation_fraction": len(validation_sessions) / len(evaluation_sessions),
        }
        split["content_hash"] = _content_hash(split)
        _write_json(temporary / "splits.json", split)
        quality = _quality_report(config, bars, selected, evaluation_sessions, excluded)
        _write_json(temporary / "quality_report.json", quality)
        file_hashes = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "version": 1,
            "snapshot_id": config["snapshot_id"],
            "created_at": retrieved_at.isoformat(),
            "config": config,
            "source_limitations": [
                "Current-constituent universe can contain survivorship bias.",
                "yfinance historical availability is public research data, not strict PIT.",
                "SEC acceptance timestamps are authoritative; first_seen_at is reconstructed.",
            ],
            "file_sha256": file_hashes,
            "quality": quality,
        }
        _write_json(temporary / "manifest.json", manifest)
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(temporary, final_path)
        return final_path
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _quality_report(
    config: dict,
    bars: pl.DataFrame,
    selected: list[dict],
    sessions: list[date],
    excluded: list[dict],
) -> dict:
    expected = len(selected) * len(sessions)
    evaluation_rows = bars.filter(pl.col("session").is_in(sessions)).height
    duplicates = (
        bars.group_by(["session", "symbol"]).len().filter(pl.col("len") > 1).height
    )
    return {
        "selected_symbol_count": len(selected),
        "evaluation_session_count": len(sessions),
        "evaluation_bar_count": evaluation_rows,
        "expected_evaluation_bar_count": expected,
        "evaluation_completeness": evaluation_rows / expected,
        "duplicate_bar_keys": duplicates,
        "zero_volume_rows": bars.filter(pl.col("volume") == 0).height,
        "excluded_symbol_count": len(excluded),
        "target_universe_size": config["target_universe_size"],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows)
    )


def _content_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data_expansion.yaml",
    )
    args = parser.parse_args()
    print(build_snapshot(args.config))


if __name__ == "__main__":
    main()
