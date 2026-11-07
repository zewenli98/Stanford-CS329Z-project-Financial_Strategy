"""Prepare a fixed, history-selected 100-stock study without changing source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.request import urlopen

import polars as pl
import yaml

from financial_strategy.schemas import EvidenceRecord, MacroEvidenceRecord

SOURCE = Path("data/2023-11-01_2026-10-30")
DEST = Path("data/research100_v1")
LIVE_DAYS = ["2026-08-03", "2026-09-01", "2026-10-01"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_study_targets(selected: list[str]) -> None:
    """Copy the chronological 676/76 split and the 100-stock label panel."""

    split = json.loads((SOURCE / "splits.json").read_text())
    (DEST / "splits.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    panel = pl.read_parquet(SOURCE / "evaluation/labels.parquet").filter(
        pl.col("symbol").is_in(selected)
    )
    prices = pl.read_parquet(DEST / "market/daily_bars.parquet").select(
        "session", "symbol", "high", "low"
    )
    labels = panel.join(prices, on=["session", "symbol"], how="left")
    (DEST / "evaluation").mkdir(exist_ok=True)
    labels.write_parquet(DEST / "evaluation/labels.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-prices", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--screening-only",
        action="store_true",
        help="Prepare screening inputs only; leave labels and configuration files untouched.",
    )
    args = parser.parse_args()
    bars = pl.read_parquet(SOURCE / "market/daily_bars.parquet")
    universe = json.loads((SOURCE / "universe/symbols.json").read_text())
    # Lock selection using only the first 60 sessions BEFORE the label interval.
    warmup = sorted(s for s in bars["session"].unique() if s < date(2023, 11, 1))[:60]
    liquidity = (
        bars.filter(pl.col("session").is_in(warmup))
        .group_by("symbol")
        .agg((pl.col("close") * pl.col("volume")).median().alias("median_dollar_volume"))
        .sort(["median_dollar_volume", "symbol"], descending=[True, False])
    )
    selected = liquidity.head(100)["symbol"].to_list()
    assert len(warmup) == 60 and len(selected) == len(set(selected)) == 100
    source_files = [
        "universe/symbols.json",
        "market/daily_bars.parquet",
        "market/corporate_actions.jsonl",
        "company/evidence.jsonl",
        "macro/events.jsonl",
    ]
    source_hashes = {name: sha(SOURCE / name) for name in source_files}
    actions = [
        json.loads(s)
        for s in (SOURCE / "market/corporate_actions.jsonl").read_text().splitlines()
        if s
    ]
    adjustment_checks = []
    for action in actions:
        day = date.fromisoformat(action["effective_at"][:10])
        history = bars.filter(pl.col("symbol") == action["symbol"]).sort("session")
        before = history.filter(pl.col("session") < day).tail(1)
        after = history.filter(pl.col("session") >= day).head(1)
        if before.is_empty() or after.is_empty():
            raise ValueError(f"Cannot establish split-price behavior: {action}")
        observed = after["close"][0] / before["close"][0]
        ratio = action["ratio"]
        continuous = abs(math.log(observed)) < abs(math.log(observed * ratio))
        adjustment_checks.append(
            {
                "symbol": action["symbol"],
                "day": str(day),
                "split_ratio": ratio,
                "close_ratio": observed,
                "already_split_adjusted": continuous,
            }
        )
    if not all(row["already_split_adjusted"] for row in adjustment_checks):
        raise ValueError(f"Mixed or unresolved price adjustment: {adjustment_checks}")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "universe").mkdir(exist_ok=True)
    (DEST / "market").mkdir(exist_ok=True)
    (DEST / "macro").mkdir(exist_ok=True)
    snapshot_id = "research100_warmup_liquidity_v1"
    derived_universe = {
        **universe,
        "snapshot_id": snapshot_id,
        "securities": sorted(
            [s for s in universe["securities"] if s["symbol"] in selected],
            key=lambda s: s["symbol"],
        ),
    }
    (DEST / "universe/symbols.json").write_text(
        json.dumps(derived_universe, indent=2), encoding="utf-8"
    )
    # Values remain exactly the vendor's values. Only misleading metadata changes.
    derived = bars.filter(pl.col("symbol").is_in(selected)).with_columns(
        pl.lit("split_adjusted").alias("adjustment_policy"),
        pl.lit(snapshot_id).alias("snapshot_id"),
    )
    derived.write_parquet(DEST / "market/daily_bars.parquet")
    if not args.screening_only:
        _write_study_targets(selected)
    (DEST / "market/corporate_actions.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in actions if a["symbol"] in selected), encoding="utf-8"
    )
    company = [
        EvidenceRecord.model_validate(json.loads(s))
        for s in (SOURCE / "company/evidence.jsonl").read_text().splitlines()
        if s
    ]
    macro = [
        MacroEvidenceRecord.model_validate(json.loads(s))
        for s in (SOURCE / "macro/events.jsonl").read_text().splitlines()
        if s
    ]
    # Retain original macro records for compatibility checks; none meet primary eligibility.
    eligible_macro = [r for r in macro if r.timestamp_confidence.value == "high"]
    (DEST / "macro/events.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in eligible_macro), encoding="utf-8"
    )
    report = {
        "source": str(SOURCE),
        "destination": str(DEST),
        "source_sha256": source_hashes,
        "selection": {
            "method": "top_100_median_close_times_volume_first_60_preevaluation_sessions",
            "first_session": str(warmup[0]),
            "last_session": str(warmup[-1]),
            "session_count": len(warmup),
            "symbols_in_liquidity_order": selected,
            "ranks": liquidity.to_dicts(),
        },
        "rows": derived.height,
        "company_records_validated": len(company),
        "macro_records_validated": len(macro),
        "macro_records_primary_eligible": len(eligible_macro),
        "adjustment_checks": adjustment_checks,
        "adjustment_decision": (
            "Retain vendor prices and volume unchanged; declare split_adjusted after "
            "checking every recorded split. Source files unchanged. "
            "Public research data, not strict PIT."
        ),
        "limitations": [
            "Current-constituent survivorship bias",
            "historical provider series may incorporate later adjustments",
            "no strict historical macro consensus availability",
            "top20 history-only research screen can miss catalysts outside shortlist",
        ],
        "live_sessions_preselected": LIVE_DAYS,
    }
    assert all(sha(SOURCE / name) == value for name, value in source_hashes.items())
    (DEST / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.screening_only:
        print(json.dumps({"prepared": str(DEST), "stocks": len(selected), "rows": derived.height}))
        return
    # Start from the maintained, repository-contained prototype configuration.
    config = yaml.safe_load(Path("configs/adaptive_range_v1.yaml").read_text())
    config["task"].update(
        universe_snapshot_id=snapshot_id,
        target_session=LIVE_DAYS[0],
        decision_cutoff=LIVE_DAYS[0] + "T09:00:00-04:00",
    )
    config["data"].update(
        universe=str(DEST / "universe/symbols.json"),
        market_bars=str(DEST / "market/daily_bars.parquet"),
        corporate_actions=str(DEST / "market/corporate_actions.jsonl"),
        company_evidence=str(SOURCE / "company/evidence.jsonl"),
        macro_events=str(SOURCE / "macro/events.jsonl"),
        macro_documents=str(SOURCE / "macro/documents"),
        input_adjustment_policy="split_adjusted",
    )
    config["agents"].update(
        mode="scripted",
        cache_path="cache/llm/research100_v1",
        max_total_tool_calls=180,
        max_total_model_calls=80,
        max_model_calls_per_symbol=16,
    )
    config["adaptive"].update(
        research_candidate_limit=20,
        daily_budget_usd=1.0,
        max_batches=3,
        max_workers=2,
        max_input_bytes=60000,
        replicate_id="research100_v1",
        memory_path=None,
    )
    prefix = "adaptive_range_100_compact" if args.compact else "adaptive_range_100"
    if args.compact:
        config["adaptive"].update(compact_research_prompts=True, max_worker_model_calls=4)
        config["agents"]["cache_path"] = "cache/llm/research100_compact_v2"
        for role in ["macro_evidence", "company_evidence"]:
            config["agents"]["roles"][role]["prompt_version"] += "_compact_v2"
    offline = Path(f"configs/{prefix}_offline.yaml")
    offline.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    deterministic_days = ["2023-11-01", "2024-06-10", *LIVE_DAYS]

    def evaluation(screening, days):
        return {
            "screening_config": str(screening),
            "evaluation": {
                "sessions": days,
                "cutoff_time": "09:00:00",
                "timezone": "America/New_York",
                "policies": ["baseline", "scenario"],
                "budgets_usd": [1.0],
                "report_top_ks": [3, 5, 10],
                "replicates": 1,
                "aggregate_budget_usd": 3.0,
                "large_move_threshold": 0.04,
            },
        }

    Path(f"configs/{prefix}_offline_evaluation.yaml").write_text(
        yaml.safe_dump(evaluation(offline, deterministic_days), sort_keys=False), encoding="utf-8"
    )
    if args.live_prices:
        with urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
            models = json.load(response)["data"]
        model = next(m for m in models if m["id"] == config["agents"]["model"])
        config["adaptive"]["prices"] = {
            model["id"]: {
                "input_per_million": float(model["pricing"]["prompt"]) * 1e6,
                "output_per_million": float(model["pricing"]["completion"]) * 1e6,
                "request_usd": float(model["pricing"].get("request", 0)),
            }
        }
        config["adaptive"]["pricing_as_of"] = datetime.now(UTC).isoformat()
        config["agents"]["mode"] = "live"
        live = Path(f"configs/{prefix}_live.yaml")
        live.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        Path(f"configs/{prefix}_live_evaluation.yaml").write_text(
            yaml.safe_dump(evaluation(live, LIVE_DAYS), sort_keys=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "live_model": model["id"],
                    "prices": config["adaptive"]["prices"],
                    "aggregate_cap": 3.0,
                }
            )
        )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "rows",
                    "company_records_validated",
                    "macro_records_validated",
                    "macro_records_primary_eligible",
                    "live_sessions_preselected",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
