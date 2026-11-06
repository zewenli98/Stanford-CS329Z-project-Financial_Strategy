"""Build a replaceable historical macro fixture with explicit PIT assumptions."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from threading import local
from time import sleep
from zoneinfo import ZoneInfo

import httpx
import yaml

from financial_strategy.schemas import MacroEvidenceRecord

ROOT = Path(__file__).resolve().parents[1]
EASTERN = ZoneInfo("America/New_York")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html",
    "Accept-Language": "en-US,en;q=0.9",
}
OFFICIAL_HEADERS = {
    "User-Agent": "FinancialStrategy research contact@example.com",
    "Accept": "text/html",
}
TRADINGVIEW_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept": "application/json",
    "Origin": "https://www.tradingview.com",
}
CACHE_DIRECTORY = ROOT / "cache/macro_calendar"
_THREAD_STATE = local()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _parse_number(value: str, event_type: str) -> float | None:
    cleaned = re.sub(r"<[^>]+>", "", value or "").replace("&nbsp;", "").strip()
    if not cleaned or cleaned in {"-", "N/A"}:
        return None
    multiplier = 1.0
    if cleaned[-1:].upper() == "K":
        cleaned = cleaned[:-1]
    elif cleaned[-1:].upper() == "M":
        cleaned = cleaned[:-1]
        multiplier = 1_000.0
    cleaned = cleaned.replace(",", "").rstrip("%")
    try:
        result = float(cleaned) * multiplier
    except ValueError:
        return None
    return result * 100.0 if event_type == "fed_rate" else result


def _archive_dates(text: str, prefix: str) -> set[date]:
    return {
        datetime.strptime(value, "%m%d%Y").date()
        for value in re.findall(rf"{prefix}_(\d{{8}})\.htm", text)
    }


def _fomc_dates(text: str) -> set[date]:
    return {
        datetime.strptime(value, "%Y%m%d").date()
        for value in re.findall(
            r"/monetarypolicy/files/monetary(20\d{6})a1\.pdf", text
        )
    }


def _calendar_rows(base_url: str, requested: date) -> list[dict]:
    cache_path = CACHE_DIRECTORY / f"{requested.isoformat()}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    last_error: Exception | None = None
    client = getattr(_THREAD_STATE, "client", None)
    if client is None:
        client = httpx.Client(headers=HEADERS, timeout=20, http2=False)
        _THREAD_STATE.client = client
    for attempt in range(2):
        try:
            response = client.get(
                base_url,
                params={"date": requested.isoformat()},
            )
            response.raise_for_status()
            rows = (response.json().get("data") or {}).get("rows") or []
            CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{cache_path.name}.", dir=CACHE_DIRECTORY
            )
            with os.fdopen(descriptor, "w") as handle:
                json.dump(rows, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary).replace(cache_path)
            return rows
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            sleep(1 + attempt)
    # Individual historical dates occasionally time out at the provider. The
    # immutable audit records the resulting missing event instead of aborting
    # and discarding every successfully retrieved release.
    del last_error
    return []


def _fetch_release_rows(base_url: str, release: date) -> list[dict]:
    # Older Nasdaq pages occasionally expose a US release under the following
    # query date. Query both dates, then deduplicate the selected rows.
    rows = []
    for requested in (release, release + timedelta(days=1)):
        for item in _calendar_rows(base_url, requested):
            if item.get("country") != "United States":
                continue
            rows.append({**item, "_calendar_query_date": requested.isoformat()})
    unique = {}
    for item in rows:
        key = (
            item.get("eventName"),
            item.get("gmt"),
            item.get("actual"),
            item.get("consensus"),
        )
        unique[key] = item
    return list(unique.values())


def _tradingview_rows(base_url: str, release: date) -> list[dict]:
    cache_path = CACHE_DIRECTORY / f"{release.isoformat()}_tradingview.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    response = httpx.get(
        base_url,
        params={
            "from": f"{release.isoformat()}T00:00:00.000Z",
            "to": f"{(release + timedelta(days=1)).isoformat()}T00:00:00.000Z",
            "countries": "US",
        },
        headers=TRADINGVIEW_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("result") or []
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{cache_path.name}.", dir=CACHE_DIRECTORY
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(rows, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    Path(temporary).replace(cache_path)
    return rows


def _match_rows(rows: list[dict], event_type: str) -> list[dict]:
    names = {
        "cpi_yoy": "CPI",
        "nonfarm_payrolls": "Nonfarm Payrolls",
        "unemployment": "Unemployment Rate",
        "fed_rate": "Fed Interest Rate Decision",
    }
    matched = [item for item in rows if item.get("eventName") == names[event_type]]
    if event_type != "cpi_yoy":
        return matched[:1]
    # CPI appears as both monthly and annual changes. The annual observation is
    # the row whose absolute actual and consensus are both at least one percent.
    annual = [
        item
        for item in matched
        if abs(_parse_number(item.get("actual", ""), event_type) or 0) >= 1
        and abs(_parse_number(item.get("consensus", ""), event_type) or 0) >= 1
    ]
    return annual[:1]


def _match_tradingview_rows(rows: list[dict], event_type: str) -> list[dict]:
    tickers = {
        "cpi_yoy": "ECONOMICS:USIRYY",
        "nonfarm_payrolls": "ECONOMICS:USNFP",
        "unemployment": "ECONOMICS:USUR",
        "fed_rate": "ECONOMICS:USINTR",
    }
    names = {
        "cpi_yoy": "CPI",
        "nonfarm_payrolls": "Nonfarm Payrolls",
        "unemployment": "Unemployment Rate",
        "fed_rate": "Fed Interest Rate Decision",
    }

    def formatted(value: float) -> str:
        suffix = "K" if event_type == "nonfarm_payrolls" else "%"
        return f"{value:g}{suffix}"

    matched = [
        {
            "_calendar_query_date": str(item["date"])[:10],
            "_calendar_actual": item["actual"],
            "actual": formatted(float(item["actual"])),
            "consensus": formatted(float(item["forecast"])),
            "country": "United States",
            "description": item.get("comment", ""),
            "eventName": names[event_type],
            "gmt": "08:30" if event_type != "fed_rate" else "14:00",
            "previous": (
                formatted(float(item["previous"]))
                if item.get("previous") is not None
                else ""
            ),
        }
        for item in rows
        if item.get("ticker") == tickers[event_type]
        and item.get("actual") is not None
        and item.get("forecast") is not None
    ]
    return matched[:1]


def _parse_bls_release_actual(page: str, event_type: str) -> float:
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    patterns = {
        "cpi_yoy": (
            r"all items index (?:increased|rose) ([\d.]+) percent "
            r"before seasonal adjustment"
        ),
        "nonfarm_payrolls": (
            r"Total nonfarm payroll employment (?:increased|rose) "
            r"by ([\d,]+)"
        ),
        "unemployment": (
            r"unemployment rate .{0,40}? (?:to|at) ([\d.]+) percent"
        ),
    }
    match = re.search(patterns[event_type], text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"could not parse official {event_type} actual")
    value = float(match.group(1).replace(",", ""))
    return value / 1_000.0 if event_type == "nonfarm_payrolls" else value


def _bls_release_actual(event_type: str, release: date) -> tuple[float, str]:
    prefix = "cpi" if event_type == "cpi_yoy" else "empsit"
    url = (
        "https://www.bls.gov/news.release/archives/"
        f"{prefix}_{release.strftime('%m%d%Y')}.htm"
    )
    cache_path = CACHE_DIRECTORY / f"{prefix}_{release.isoformat()}_official.html"
    if cache_path.exists():
        page = cache_path.read_text()
    else:
        response = httpx.get(
            url,
            headers=OFFICIAL_HEADERS,
            timeout=45,
        )
        response.raise_for_status()
        page = response.text
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(page)
    return _parse_bls_release_actual(page, event_type), url


def _event_time(release: date, event_type: str) -> datetime:
    wall_time = time(14, 0) if event_type == "fed_rate" else time(8, 30)
    return datetime.combine(release, wall_time, EASTERN).astimezone(UTC)


def _record(
    *,
    release: date,
    event_type: str,
    row: dict,
    official_url: str,
    retrieved_at: datetime,
    document_path: str,
    calendar_source_name: str,
    calendar_source_url: str,
    source_type: str,
) -> dict:
    event_time = _event_time(release, event_type)
    actual = _parse_number(row.get("actual", ""), event_type)
    consensus = _parse_number(row.get("consensus", ""), event_type)
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    event_id = f"macro_reconstructed_{event_type}_{release.isoformat()}"
    factors = {
        "cpi_yoy": ["inflation", "interest_rates"],
        "nonfarm_payrolls": ["interest_rates", "usd"],
        "unemployment": ["interest_rates", "usd"],
        "fed_rate": ["interest_rates", "usd"],
    }[event_type]
    unit = {
        "cpi_yoy": "percentage_points",
        "nonfarm_payrolls": "thousands_of_jobs",
        "unemployment": "percentage_points",
        "fed_rate": "basis_points",
    }[event_type]
    return {
        "evidence_id": event_id,
        "scope": "macro",
        "symbol": None,
        "source_type": source_type,
        "source_name": f"Official release calendar and {calendar_source_name}",
        "source_url": official_url,
        "title": f"{event_type} release {release.isoformat()}",
        "excerpt": (
            f"Historical calendar row: {event_type} actual {row.get('actual')} "
            f"versus consensus {row.get('consensus')}."
        ),
        "event_time": event_time,
        "published_at": event_time,
        "first_seen_at": event_time,
        "retrieved_at": retrieved_at,
        "available_at": event_time,
        "content_hash": _hash(payload),
        "snapshot_id": "macro_pit_2023_2026",
        "timestamp_confidence": "low",
        "content_path": document_path,
        "event_type": event_type,
        "event_status": "released_before_cutoff",
        "schema_version": 1,
        "status": "released_before_cutoff",
        "actual": actual,
        "consensus": consensus,
        "consensus_source_name": calendar_source_name,
        "consensus_source_url": calendar_source_url,
        "consensus_published_at": event_time - timedelta(seconds=1),
        "consensus_content_hash": _hash(payload),
        "unit": unit,
        "affected_risk_factors": factors,
    }


def _scheduled_fomc_record(
    release: date,
    retrieved_at: datetime,
    official_url: str,
    document_path: str,
) -> dict:
    event_time = _event_time(release, "fed_rate")
    available_at = datetime.combine(
        release - timedelta(days=1), time(16, 0), EASTERN
    ).astimezone(UTC)
    payload = f"FOMC meeting scheduled for {event_time.isoformat()}"
    return {
        "evidence_id": f"macro_scheduled_fed_rate_{release.isoformat()}",
        "scope": "macro",
        "symbol": None,
        "source_type": "official_schedule_assumed_pit",
        "source_name": "Federal Reserve",
        "source_url": official_url,
        "title": f"Scheduled FOMC decision {release.isoformat()}",
        "excerpt": payload,
        "event_time": event_time,
        "published_at": available_at,
        "first_seen_at": available_at,
        "retrieved_at": retrieved_at,
        "available_at": available_at,
        "content_hash": _hash(payload),
        "snapshot_id": "macro_pit_2023_2026",
        "timestamp_confidence": "low",
        "content_path": document_path,
        "event_type": "fed_rate",
        "event_status": "scheduled_during_session",
        "schema_version": 1,
        "status": "scheduled_during_session",
        "actual": None,
        "consensus": None,
        "unit": "basis_points",
        "affected_risk_factors": ["interest_rates", "usd"],
    }


def _build_seed_fixture(config: dict, output: Path) -> Path:
    """Materialize the audited seed after an external-provider backfill failure."""

    source = ROOT / config["seed_source"]
    if not source.exists():
        raise FileNotFoundError(f"macro seed does not exist: {source}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        records = [
            MacroEvidenceRecord.model_validate_json(line)
            for line in (temporary / "events.jsonl").read_text().splitlines()
            if line
        ]
        normalized = []
        for item in records:
            if item.event_time is None:
                normalized.append(item)
                continue
            release = item.event_time.date()
            normalized_time = _event_time(release, item.event_type)
            normalized.append(
                item.model_copy(
                    update={
                        "event_time": normalized_time,
                        "published_at": normalized_time,
                        "first_seen_at": normalized_time,
                        "available_at": normalized_time,
                        "consensus_published_at": (
                            normalized_time - timedelta(seconds=1)
                            if item.consensus is not None
                            else None
                        ),
                    }
                )
            )
            if item.event_type == "fed_rate":
                relative = f"documents/fed_rate_scheduled_{release.isoformat()}.txt"
                (temporary / relative).write_text(
                    f"FOMC decision scheduled for {release.isoformat()} at 14:00 ET.\n"
                )
                normalized.append(
                    MacroEvidenceRecord.model_validate(
                        _scheduled_fomc_record(
                            release,
                            datetime.now(UTC),
                            str(item.source_url),
                            relative,
                        )
                    )
                )
        records = sorted(
            normalized,
            key=lambda item: (item.event_time or item.available_at, item.evidence_id),
        )
        (temporary / "events.jsonl").write_text(
            "".join(
                json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
                for item in records
            )
        )
        released = [
            item for item in records if item.status == "released_before_cutoff"
        ]
        counts = {
            event_type: sum(item.event_type == event_type for item in released)
            for event_type in sorted({item.event_type for item in released})
        }
        audit = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "config": config,
            "released_event_counts": counts,
            "distinct_release_sessions": len(
                {
                    item.event_time.date().isoformat()
                    for item in released
                    if item.event_time is not None
                }
            ),
            "missing_calendar_rows": [],
            "primary_eligible_count": sum(
                item.timestamp_confidence.value == "high" for item in released
            ),
            "secondary_assumed_pit_count": sum(
                item.timestamp_confidence.value != "high" for item in released
            ),
            "backfill_status": "seed_only_after_provider_timeout",
            "timestamp_normalization": (
                "BLS CPI/employment releases normalized to 08:30 America/New_York; "
                "FOMC decisions normalized to 14:00 and receive separate scheduled records."
            ),
            "limitation": (
                "The external historical calendar timed out during this run. "
                "The fixture contains the previously audited seed only and must "
                "fail the planned coverage gates."
            ),
        }
        (temporary / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def backfill(config_path: Path, *, seed_only: bool = False) -> Path:
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_path"]
    if output.exists():
        shutil.rmtree(output)
    if seed_only:
        return _build_seed_fixture(config, output)
    start = date.fromisoformat(config["start_date"])
    end = date.fromisoformat(config["end_date"])
    retrieved_at = datetime.now(UTC)

    with httpx.Client(headers=OFFICIAL_HEADERS, timeout=45) as client:
        cpi_response = client.get(config["bls_cpi_archive_url"])
        employment_response = client.get(config["bls_employment_archive_url"])
        fomc_response = client.get(config["fomc_calendar_url"])
        cpi_response.raise_for_status()
        employment_response.raise_for_status()
        fomc_response.raise_for_status()
        cpi_page = cpi_response.text
        employment_page = employment_response.text
        fomc_page = fomc_response.text
    release_dates = {
        "cpi_yoy": sorted(
            value
            for value in _archive_dates(cpi_page, "cpi")
            if start <= value <= end
        ),
        "nonfarm_payrolls": sorted(
            value
            for value in _archive_dates(employment_page, "empsit")
            if start <= value <= end
        ),
        "unemployment": sorted(
            value
            for value in _archive_dates(employment_page, "empsit")
            if start <= value <= end
        ),
        "fed_rate": sorted(
            value for value in _fomc_dates(fomc_page) if start <= value <= end
        ),
    }
    unique_dates = sorted({value for values in release_dates.values() for value in values})
    with ThreadPoolExecutor(max_workers=2) as executor:
        fetched = dict(
            zip(
                unique_dates,
                executor.map(
                    lambda value: _fetch_release_rows(
                        config["nasdaq_calendar_url"], value
                    ),
                    unique_dates,
                ),
                strict=True,
            )
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        documents = temporary / "documents"
        documents.mkdir(parents=True)
        records = []
        missing = []
        fallback_row_count = 0
        fallback_actual_override_count = 0
        for event_type, dates in release_dates.items():
            for release in dates:
                matched = _match_rows(fetched[release], event_type)
                official_release_url = (
                    config["fomc_calendar_url"]
                    if event_type == "fed_rate"
                    else config["bls_cpi_archive_url"]
                    if event_type == "cpi_yoy"
                    else config["bls_employment_archive_url"]
                )
                calendar_source_name = "Nasdaq Economic Calendar"
                calendar_source_url = (
                    f"{config['nasdaq_calendar_url']}?"
                    f"date={matched[0]['_calendar_query_date']}"
                    if matched
                    else ""
                )
                source_type = "official_date_with_reconstructed_consensus"
                if not matched:
                    matched = _match_tradingview_rows(
                        _tradingview_rows(
                            config["tradingview_calendar_url"], release
                        ),
                        event_type,
                    )
                    if matched:
                        fallback_row_count += 1
                        if event_type != "fed_rate":
                            official_actual, official_release_url = (
                                _bls_release_actual(event_type, release)
                            )
                            calendar_actual = float(
                                matched[0].pop("_calendar_actual")
                            )
                            if calendar_actual != official_actual:
                                fallback_actual_override_count += 1
                            suffix = (
                                "K"
                                if event_type == "nonfarm_payrolls"
                                else "%"
                            )
                            matched[0]["actual"] = (
                                f"{official_actual:g}{suffix}"
                            )
                        else:
                            matched[0].pop("_calendar_actual", None)
                        calendar_source_name = "TradingView Economic Calendar"
                        calendar_source_url = (
                            f"{config['tradingview_calendar_url']}?"
                            f"from={release.isoformat()}T00:00:00.000Z&"
                            f"to={(release + timedelta(days=1)).isoformat()}"
                            "T00:00:00.000Z&countries=US"
                        )
                        source_type = (
                            "official_date_with_fallback_calendar_consensus"
                        )
                if not matched:
                    missing.append(
                        {"event_type": event_type, "release_date": release.isoformat()}
                    )
                    continue
                row = matched[0]
                relative = f"documents/{event_type}_{release.isoformat()}.json"
                text = json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                (temporary / relative).write_text(text)
                records.append(
                    _record(
                        release=release,
                        event_type=event_type,
                        row=row,
                        official_url=official_release_url,
                        retrieved_at=retrieved_at,
                        document_path=relative,
                        calendar_source_name=calendar_source_name,
                        calendar_source_url=calendar_source_url,
                        source_type=source_type,
                    )
                )
                if event_type == "fed_rate":
                    scheduled_relative = (
                        f"documents/fed_rate_scheduled_{release.isoformat()}.txt"
                    )
                    (temporary / scheduled_relative).write_text(
                        f"FOMC decision scheduled for {release.isoformat()} at 14:00 ET.\n"
                    )
                    records.append(
                        _scheduled_fomc_record(
                            release,
                            retrieved_at,
                            official_release_url,
                            scheduled_relative,
                        )
                    )
        validated = [
            MacroEvidenceRecord.model_validate(item).model_dump(mode="json")
            for item in sorted(
                records,
                key=lambda item: (str(item["event_time"]), item["evidence_id"]),
            )
        ]
        (temporary / "events.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in validated)
        )
        counts = {
            event_type: sum(
                item["event_type"] == event_type
                and item["status"] == "released_before_cutoff"
                for item in validated
            )
            for event_type in release_dates
        }
        audit = {
            "version": 1,
            "created_at": retrieved_at.isoformat(),
            "config": config,
            "released_event_counts": counts,
            "distinct_release_sessions": len(
                {str(item["event_time"])[:10] for item in validated}
            ),
            "missing_calendar_rows": missing,
            "fallback_calendar_row_count": fallback_row_count,
            "fallback_actual_override_count": fallback_actual_override_count,
            "primary_eligible_count": 0,
            "secondary_assumed_pit_count": len(validated),
            "limitation": (
                "Official release dates are verified, but historical consensus "
                "availability is assumed and therefore secondary-analysis only."
            ),
        }
        (temporary / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/macro_backfill.yaml",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Materialize the audited seed when the external provider is unavailable.",
    )
    arguments = parser.parse_args()
    output = backfill(arguments.config.resolve(), seed_only=arguments.seed_only)
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
