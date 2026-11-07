"""Run and display the verified ranking-correction head in the one-day demo."""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import yaml

from financial_strategy.audit.writer import verify_manifest
from financial_strategy.config import MVPConfig
from financial_strategy.service import run_screening

ROOT = Path(__file__).resolve().parents[1]


def prepare_head_a(raw_b, launch, budget):
    raw = yaml.safe_load(
        (ROOT / "configs/screening_research100_llm_live.yaml").read_text(encoding="utf-8")
    )
    raw["data"] = copy.deepcopy(raw_b["data"])
    raw["task"] = copy.deepcopy(raw_b["task"])
    raw["output"]["runs_path"] = str(ROOT / "runs")
    raw["agents"].update(
        mode=raw_b["agents"]["mode"],
        cache_path=str(ROOT / "cache/demo_a" / launch.name),
        recordings_path=str(ROOT / "cache/demo_a_tape" / launch.name),
        max_cost_usd=budget,
    )
    # Full earnings claims, events and exact metric quotes need more room than
    # the old metadata-only 2048-token responses. Keep both stages bounded.
    for role in ("company_evidence", "verifier"):
        raw["agents"]["roles"][role]["max_tokens"] = max(
            4096, raw["agents"]["roles"][role]["max_tokens"]
        )
    if raw["ranking"]["market_feature_weights"] != raw_b["ranking"]["market_feature_weights"]:
        raise ValueError("Heads A and B must use identical frozen market weights for this demo.")
    config = MVPConfig.model_validate(raw)
    path = launch / "head_a.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path, config


def run_head_a(path, config, run_id, baseline_watchlist):
    """Keep a failed head explicit, while allowing B to run and report its own result."""
    started = time.monotonic()
    run_dir = Path(config.output["runs_path"]) / f"a_{run_id}"
    summary = {
        "head": "A",
        "run_id": f"a_{run_id}",
        "run_dir": str(run_dir),
        "status": "failed",
        "residual_enabled": config.ranking.residual_model.enabled,
        "score_units": "unitless ranking score in [0, 1]; not a percentage range",
        "budget_usd": config.agents.max_cost_usd,
        "cost_guard": (
            "existing Head A post-response usage guard; not a pre-dispatch dollar reservation"
        ),
        "reported_live_cost_usd": None,
        "watchlist": [],
    }
    try:
        run_dir, watchlist = run_screening(path, run_id=f"a_{run_id}")
        verify_manifest(run_dir)
        cards = json.loads((run_dir / "research_cards.json").read_text(encoding="utf-8"))
        published = json.loads(
            (run_dir / "published_research_cards.json").read_text(encoding="utf-8")
        )
        attempts = [
            json.loads(line)
            for line in (run_dir / "model_attempts.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        live = [
            row
            for row in attempts
            if row.get("status") == "completed" and row.get("response_source") == "live"
        ]
        cost = (
            sum(
                row.get("input_tokens", 0) * config.agents.input_cost_per_million
                + row.get("output_tokens", 0) * config.agents.output_cost_per_million
                for row in live
            )
            / 1_000_000
        )
        baseline_symbols = [row["symbol"] for row in baseline_watchlist]
        symbols = [row["symbol"] for row in watchlist]
        summary.update(
            status="completed" if config.agents.mode != "live" or live else "no_live_response",
            watchlist=watchlist,
            research_card_count=len(cards),
            accepted_cards=len(published),
            non_published_cards=len(cards) - len(published),
            model_attempt_count=len(attempts),
            live_provider_responses=len(live),
            failed_model_attempt_count=sum(row.get("status") == "failed" for row in attempts),
            reported_live_cost_usd=cost,
            cost_note=(
                "Successful responses with reported usage only; failed/unreported usage excluded."
            ),
            top_10_order_changed=symbols != baseline_symbols,
            entered_top_10=sorted(set(symbols) - set(baseline_symbols)),
            left_top_10=sorted(set(baseline_symbols) - set(symbols)),
            changed_top_10_scores=[
                row for row in watchlist if abs(row["score"] - row["market_score"]) > 1e-12
            ],
        )
    except Exception as exc:
        summary["error"] = str(exc)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return summary


def print_head_a(summary):
    print("\nHEAD A - RESEARCH + VERIFIED RANKING CORRECTION", flush=True)
    state = (
        "ENABLED"
        if summary["residual_enabled"]
        else "DISABLED; verified direct correction is active"
    )
    print(f"Learned correction: {state}", flush=True)
    if summary["status"] == "failed":
        print(f"Head A failed: {summary.get('error', 'See the audit output.')}", flush=True)
        print(f"Audit: {summary['run_dir']}", flush=True)
        return
    print("Scores are unitless ranking scores, not predicted percentage ranges.", flush=True)
    print(f"{'Rank':>4}  {'Stock':<7} {'Market score':>13} {'Residual':>11} {'Final score':>13}")
    for row in summary["watchlist"]:
        print(
            f"{row['rank']:>4}  {row['symbol']:<7} {row['market_score']:>13.6f} "
            f"{row['signed_residual']:>+11.6f} {row['score']:>13.6f}"
        )
    print(
        f"Research cards: {summary['research_card_count']} | "
        f"Verified/published: {summary['accepted_cards']} | "
        f"Not published: {summary['non_published_cards']}"
    )
    print(f"Changed top-10 scores: {len(summary['changed_top_10_scores'])}")
    print(
        f"Live provider responses: {summary['live_provider_responses']} | "
        f"Reported successful-call cost: ${summary['reported_live_cost_usd']:.4f}"
    )
    print(f"Head A run: {summary['run_dir']}", flush=True)


def ground_truth_after_publication(config, baseline_dir, head_a=None, head_b=None):
    """Open target-session bars only after checking the published predictions."""
    from financial_strategy.adaptive.data import derive_range_labels

    verify_manifest(baseline_dir)
    for head in (head_a, head_b):
        if head and head.get("status") == "completed":
            verify_manifest(Path(head["run_dir"]))
    predictions = json.loads(
        (baseline_dir / "range_predictions.json").read_text(encoding="utf-8")
    )
    symbols = [row["symbol"] for row in predictions]
    labels, coverage = derive_range_labels(
        config.data.market_bars, [config.task.target_session], symbols
    )
    complete = coverage["available_label_count"] == len(symbols) and bool(symbols)
    # A partial panel cannot identify the true top ten of the full prediction universe.
    ordered = sorted(labels.to_dicts(), key=lambda row: (-row["range"], row["symbol"]))
    return {
        "session": config.task.target_session.isoformat(),
        "status": "completed" if complete else "incomplete_coverage",
        "target": "(session high - session low) / session low",
        "usage": "Post-prediction evaluation only; never supplied to either head.",
        "coverage": coverage,
        "watchlist": [
            {"rank": rank, "symbol": row["symbol"], "realized_range": row["range"]}
            for rank, row in enumerate(ordered[: config.task.top_k], start=1)
        ] if complete else [],
    }


def write_comparison(launch, baseline, head_a=None, head_b=None, ground_truth=None):
    a_rows = {row["rank"]: row for row in (head_a or {}).get("watchlist", [])}
    b_rows = {row["rank"]: row for row in (head_b or {}).get("watchlist", [])}
    truth_rows = {row["rank"]: row for row in (ground_truth or {}).get("watchlist", [])}
    rows = []
    for row in baseline:
        rank = row["rank"]
        a = a_rows.get(rank, {})
        b = b_rows.get(rank, {})
        truth = truth_rows.get(rank, {})
        rows.append(
            {
                "rank": rank,
                "baseline_symbol": row["symbol"],
                "baseline_range_ratio": row.get("baseline_range", ""),
                "head_a_symbol": a.get("symbol", ""),
                "head_a_market_score": a.get("market_score", ""),
                "head_a_signed_residual": a.get("signed_residual", ""),
                "head_a_score": a.get("score", ""),
                "head_b_symbol": b.get("symbol", ""),
                "head_b_baseline_range_ratio": b.get("baseline_range", ""),
                "head_b_predicted_range_ratio": b.get("predicted_range", ""),
                "ground_truth_symbol": truth.get("symbol", ""),
                "ground_truth_realized_range_ratio": truth.get("realized_range", ""),
            }
        )
    path = launch / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if ground_truth is not None:
        (launch / "ground_truth.json").write_text(
            json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8"
        )
    print("\nTOP-10 COMPARISON (same session and stock universe)")
    print("M/A/B use the same pre-session cutoff and market weights.")
    print("Ground truth: actual session (high - low) / low; evaluation after predictions.")
    if ground_truth is not None:
        coverage = ground_truth["coverage"]
        print(
            f"Ground-truth coverage: {coverage['available_label_count']}/"
            f"{coverage['expected_label_count']} stocks"
        )
        if ground_truth["status"] != "completed":
            print("Ground-truth top ten unavailable: incomplete session high/low coverage.")
    print(
        f"{'Rank':>4}  {'Baseline M':<12} {'Head A':<12} {'Head B':<12} "
        f"{'Ground truth':<13} {'Actual range':>12}"
    )
    for row in rows:
        realized = row["ground_truth_realized_range_ratio"]
        actual = f"{realized:.2%}" if realized != "" else "--"
        print(
            f"{row['rank']:>4}  {row['baseline_symbol']:<12} "
            f"{row['head_a_symbol'] or '--':<12} {row['head_b_symbol'] or '--':<12} "
            f"{row['ground_truth_symbol'] or '--':<13} {actual:>12}"
        )
    print(f"Comparison table: {path}", flush=True)
    return path
