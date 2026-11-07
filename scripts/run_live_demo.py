"""Implementation for run_live_demo.ps1: one frozen day, fresh inference, readable output."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from datetime import time as daytime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

# Resolve this checkout's updated runtime before an editable install elsewhere.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import financial_strategy  # noqa: E402
from financial_strategy.adaptive.config import AdaptiveConfig  # noqa: E402
from financial_strategy.adaptive.service import run_adaptive  # noqa: E402
from financial_strategy.audit.writer import verify_manifest  # noqa: E402
from financial_strategy.data.calendar import TradingCalendar  # noqa: E402
from scripts.demo_head_a import (  # noqa: E402
    ground_truth_after_publication,
    prepare_head_a,
    print_head_a,
    run_head_a,
    write_comparison,
)

BASE_CONFIG = ROOT / "configs/adaptive_range_100_compact_live.yaml"


def say(message=""):
    print(message, flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@contextmanager
def elapsed_progress(label="Research"):
    """Keep the recording responsive without inventing fine-grained agent progress."""
    done = threading.Event()
    started = time.monotonic()

    def report():
        while not done.wait(15):
            say(f"  {label} running... {time.monotonic() - started:.0f}s elapsed")

    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join()


def prepare_inputs(raw):
    required_subset = [raw["data"][key] for key in ("universe", "market_bars", "corporate_actions")]
    if any(not Path(path).is_file() for path in required_subset):
        say("[1/5] Preparing the local 100-stock subset (no API calls; no labels loaded).")
        subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/prepare_research100.py", "--screening-only"],
            cwd=ROOT,
            check=True,
        )
    else:
        say("[1/5] Using the existing local 100-stock subset.")
    for key in (
        "universe",
        "market_bars",
        "corporate_actions",
        "company_evidence",
        "macro_events",
        "macro_documents",
        "sector_exposures",
    ):
        path = Path(raw["data"][key])
        if not path.exists():
            raise ValueError(f"Missing {key}: {path}. Restore the project's local archive first.")


def print_watchlist(watchlist):
    say()
    say("HEAD B - DIRECT RANGE REVISION: TOP-10 VOLATILITY WATCHLIST")
    say("Range = (session high - session low) / session low; this is movement size.")
    say(
        f"{'Rank':>4}  {'Stock':<7} {'Baseline':>10} {'Forecast':>10} "
        f"{'Change (pp)':>12} {'Cards':>6}"
    )
    for row in watchlist:
        baseline = row["baseline_range"]
        forecast = row["predicted_range"]
        say(
            f"{row['rank']:>4}  {row['symbol']:<7} {baseline:>9.2%} {forecast:>9.2%} "
            f"{100 * (forecast - baseline):>+12.3f} {len(row['supporting_card_ids']):>6}"
        )


def single_day_evidence(raw, launch, config):
    """Copy only temporally admissible documents; the normal gateway checks them again.

    The stock filter and deduplication remain in the existing pipeline, in their original
    order. Filtering availability before opening documents avoids reading the entire
    multi-year archive twice for a single-day recording.
    """
    cutoff = config.task.decision_cutoff
    earliest = cutoff - timedelta(days=config.task.news_lookback_days)
    counts = {}
    for scope, key in (("company", "company_evidence"), ("macro", "macro_events")):
        source = Path(raw["data"][key]).resolve()
        target = launch / "inputs" / scope / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        counts[scope] = 0
        with source.open(encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
            for line in reader:
                if not line.strip():
                    continue
                record = json.loads(line)
                available = datetime.fromisoformat(record["available_at"])
                if available > cutoff or (scope == "company" and available < earliest):
                    continue
                if record.get("content_path"):
                    document = (source.parent / record["content_path"]).resolve()
                    if not document.is_relative_to(source.parent):
                        raise ValueError(f"Unsafe source document path in {record['evidence_id']}.")
                    destination = (target.parent / record["content_path"]).resolve()
                    if not destination.is_relative_to(target.parent):
                        raise ValueError(f"Unsafe destination path in {record['evidence_id']}.")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(document, destination)
                writer.write(line.rstrip("\r\n") + "\n")
                counts[scope] += 1
        raw["data"][key] = str(target)
        if scope == "macro":
            documents = target.parent / "documents"
            documents.mkdir(exist_ok=True)
            raw["data"]["macro_documents"] = str(documents)
    say(f"  Date-scoped source records: {counts['company']} company, {counts['macro']} macro.")


def credential_file(explicit=None):
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f"Credential file not found: {explicit}")
        return explicit
    local = ROOT / ".env"
    if local.is_file():
        return local
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return local
    shared = Path(common).resolve().parent / ".env"
    return shared if shared.is_file() else local


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-08-27")
    parser.add_argument("--budget-usd", type=float, default=0.8)
    parser.add_argument("--head-a-budget-usd", type=float, default=0.8)
    parser.add_argument("--head", choices=["A", "B", "Both"], default="Both")
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--report-run-id",
        help="Rebuild a saved run comparison with ground truth; no provider calls.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--require-differences",
        action="store_true",
        help="Fail the demo check if M, A, and B do not have three distinct orders.",
    )
    return parser.parse_args(argv)


def comparison_report(launch, config, combined):
    baseline_dir = Path(combined["baseline_run_dir"])
    head_a = combined["heads"].get("A")
    head_b = combined["heads"].get("B")
    truth = ground_truth_after_publication(config, baseline_dir, head_a, head_b)
    write_comparison(
        launch, read_json(baseline_dir / "watchlist.json"), head_a, head_b, truth
    )
    return truth


def refresh_saved_comparison(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", run_id):
        raise ValueError("ReportRunId must be 1-24 letters, digits, underscores, or hyphens.")
    launch = ROOT / "runs/demo_launches" / run_id
    combined = read_json(launch / "summary.json")
    config = AdaptiveConfig.model_validate(
        yaml.safe_load((launch / "config.yaml").read_text(encoding="utf-8"))
    )
    say(f"SAVED-RUN COMPARISON: {run_id} | Session: {combined['session']}")
    say("No model-provider calls. Published predictions and the original summary are preserved.")
    truth = comparison_report(launch, config, combined)
    return 0 if truth["status"] == "completed" else 4


def main():
    args = parse_args()
    if args.require_differences and args.head != "Both":
        raise ValueError("RequireDifferences needs both heads: use -Head Both.")
    runtime = Path(financial_strategy.__file__).resolve()
    if not runtime.is_relative_to((ROOT / "src").resolve()):
        raise ValueError(f"Wrong code checkout loaded: {runtime}. Restart from {ROOT}.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        raise ValueError("Date must use YYYY-MM-DD.")
    session = date.fromisoformat(args.date)
    if not math.isfinite(args.budget_usd) or not 0.05 <= args.budget_usd <= 10:
        raise ValueError("Budget must be between $0.05 and $10.")
    if not math.isfinite(args.head_a_budget_usd) or not 0.05 <= args.head_a_budget_usd <= 10:
        raise ValueError("Head A budget must be between $0.05 and $10.")
    if not 1 <= args.max_batches <= 8:
        raise ValueError("MaxBatches must be between 1 and 8.")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", args.run_id):
        raise ValueError("RunId must be 1-24 letters, digits, underscores, or hyphens.")

    os.chdir(ROOT)
    if args.report_run_id:
        return refresh_saved_comparison(args.report_run_id)
    env_file = credential_file(args.env_file)
    load_dotenv(env_file, override=False)
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    calendar = TradingCalendar(raw["calendar"]["name"])
    if not calendar.is_session(session):
        raise ValueError(f"{session} is not a trading day. Try 2026-08-27.")
    cutoff = datetime.combine(session, daytime(9), ZoneInfo("America/New_York"))
    mode = "scripted" if args.offline else "live"
    # Keep paths short enough for Windows, including the run writer's temporary suffix.
    run_id = args.run_id or (
        f"{'offline' if args.offline else 'live'}_{session:%Y%m%d}_{uuid4().hex[:6]}"
    )
    launch = ROOT / "runs/demo_launches" / run_id
    if launch.exists() or (ROOT / "runs" / run_id).exists():
        raise ValueError("RunId already exists. Choose a new RunId; previous runs are immutable.")

    say("ONE-DAY RECOMMENDATION DEMO")
    say(f"Session: {session} | Decision cutoff: {cutoff.isoformat()}")
    say(f"Code checkout: {ROOT}")
    say("Data: frozen historical archive. The session date is not today's market feed.")
    say(f"Mode: {'OFFLINE REHEARSAL' if args.offline else 'LIVE MODEL CALLS'}")
    say(f"Model: {raw['agents']['model']}")
    selected_heads = ["A", "B"] if args.head == "Both" else [args.head]
    say(f"Heads: {', '.join(selected_heads)}")
    if "A" in selected_heads:
        say(f"Head A usage-based budget setting: ${args.head_a_budget_usd:.2f}")
    if "B" in selected_heads:
        say(f"Head B configured-price cap: ${args.budget_usd:.2f} | B batches: {args.max_batches}")
    say(f"Pricing reference: {raw['adaptive']['pricing_as_of']}")
    say("Dry run: no model-provider calls." if args.dry_run else "")
    key_present = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    if not args.dry_run and not args.offline and not key_present:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to this worktree's .env, set it in the "
            "environment, or pass -EnvFile with your existing .env path. "
            "Never put it in this script."
        )

    prepare_inputs(raw)
    raw["task"].update(target_session=session.isoformat(), decision_cutoff=cutoff.isoformat())
    raw["agents"].update(
        mode=mode,
        cache_path=str(ROOT / "cache/live_demo" / run_id),
        recordings_path=str(ROOT / "cache/live_demo_recordings" / run_id),
    )
    raw["adaptive"].update(
        daily_budget_usd=args.budget_usd,
        max_batches=args.max_batches,
        baseline_method="market_score",
        replicate_id="0",
        memory_path=None,
    )
    raw["output"]["runs_path"] = str(ROOT / "runs")
    config = AdaptiveConfig.model_validate(raw)
    launch.mkdir(parents=True, exist_ok=False)
    single_day_evidence(raw, launch, config)
    config = AdaptiveConfig.model_validate(raw)
    config_path = launch / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    head_a_config = (
        prepare_head_a(raw, launch, args.head_a_budget_usd) if "A" in selected_heads else None
    )
    if head_a_config:
        enabled = head_a_config[1].ranking.residual_model.enabled
        state = "ENABLED" if enabled else "DISABLED; verified direct correction active"
        say(f"Head A learned correction: {state}")

    # Run the full data path without constructing a live client or opening outcome labels.
    baseline_raw = config.model_dump(mode="python")
    baseline_raw["agents"]["mode"] = "scripted"
    baseline_raw["adaptive"]["policy"] = "baseline"
    say("[2/5] Checking point-in-time inputs and building the no-agent baseline.")
    with elapsed_progress("Input checks"):
        baseline_dir, baseline_watchlist = run_adaptive(
            AdaptiveConfig.model_validate(baseline_raw), run_id=f"baseline_{run_id}"
        )
    verify_manifest(baseline_dir)
    baseline_predictions = read_json(baseline_dir / "range_predictions.json")
    expected = len(read_json(config.data.universe)["securities"])
    if len(baseline_predictions) != expected:
        raise ValueError(
            f"Only {len(baseline_predictions)}/{expected} stocks have complete inputs "
            f"for {session}. "
            "Choose a session covered by the archive (for example 2026-08-27), "
            f"or refresh the dataset. Details: {baseline_dir / 'screening.json'}"
        )
    say(f"  Ready: {expected} eligible stocks; top {len(baseline_watchlist)} selected.")
    if args.dry_run:
        say("\nNO-AGENT BASELINE M (preflight only)")
        say(f"{'Rank':>4}  {'Stock':<7} {'Baseline range':>15}")
        for row in baseline_watchlist:
            say(f"{row['rank']:>4}  {row['symbol']:<7} {row['baseline_range']:>14.2%}")
        say()
        say(f"Preflight passed. OPENROUTER_API_KEY: {'available' if key_present else 'missing'}.")
        say("The table above is M only. Neither prediction head has run.")
        say(f"Prepared config: {config_path}")
        return 0

    combined = {
        "session": session.isoformat(),
        "cutoff": cutoff.isoformat(),
        "mode": mode,
        "selected_heads": selected_heads,
        "code_checkout": str(ROOT),
        "runtime_package": str(runtime),
        "baseline_run_dir": str(baseline_dir.resolve()),
        "heads": {},
    }
    head_a = None
    if head_a_config:
        say("[3/5] Running Head A: company/macro research, verification, and ranking.")
        with elapsed_progress("Head A"):
            head_a = run_head_a(*head_a_config, run_id, baseline_watchlist)
        print_head_a(head_a)
        combined["heads"]["A"] = head_a
        (launch / "summary.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )
        say("Replay Head A without another provider call:")
        say(
            "  .\\.venv\\Scripts\\python.exe -X utf8 "
            f".\\.venv\\Scripts\\financial-strategy.exe replay a_{run_id}"
        )
    if "B" not in selected_heads:
        combined["ground_truth"] = comparison_report(launch, config, combined)
        (launch / "summary.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )
        say(f"Demo summary: {launch / 'summary.json'}")
        return 0 if head_a["status"] == "completed" else 2

    say("[4/5] Running Head B: adaptive research, verification, and range revision.")
    started = time.monotonic()
    try:
        with elapsed_progress("Head B"):
            run_dir, watchlist = run_adaptive(config, run_id=run_id)
    except Exception as exc:
        combined["heads"]["B"] = {"status": "failed", "error": str(exc), "run_id": run_id}
        (launch / "summary.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )
        write_comparison(launch, baseline_watchlist, head_a=head_a)
        raise
    say("[5/5] Verifying the saved artifacts and comparing the recommendations.")
    verify_manifest(run_dir)
    print_watchlist(watchlist)
    predictions = read_json(run_dir / "range_predictions.json")
    cards = read_json(run_dir / "research_cards.json")
    rejected = read_json(run_dir / "rejected_cards.json")
    calls = read_rows(run_dir / "model_calls.jsonl")
    budget = read_json(run_dir / "budget.json")
    execution = read_json(run_dir / "execution.json")
    live_calls = sum(row.get("result", {}).get("source") == "live" for row in calls)
    changed = sum(
        abs(row["predicted_range"] - row["baseline_range"]) > 1e-12 for row in predictions
    )
    prior = {row["symbol"] for row in baseline_watchlist}
    selected = {row["symbol"] for row in watchlist}
    summary = {
        "head": "B",
        "status": "completed" if args.offline or live_calls else "no_live_response",
        "watchlist": watchlist,
        "session": session.isoformat(),
        "cutoff": cutoff.isoformat(),
        "mode": mode,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "baseline_run_dir": str(baseline_dir.resolve()),
        "live_provider_responses": live_calls,
        "accepted_cards": len(cards),
        "rejected_cards": len(rejected),
        "changed_forecasts": changed,
        "changed_forecast_details": [
            {
                key: row[key]
                for key in ("symbol", "baseline_range", "predicted_range", "supporting_card_ids")
            }
            for row in predictions
            if abs(row["predicted_range"] - row["baseline_range"]) > 1e-12
        ],
        "entered_top_10": sorted(selected - prior),
        "left_top_10": sorted(prior - selected),
        "metered_actual_usd": budget["actual_usd"],
        "normalized_uncached_usd": budget["normalized_uncached_usd"],
        "stop_reason": execution.get("stop_reason"),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    combined["heads"]["B"] = summary
    orders = [
        [row["symbol"] for row in baseline_watchlist],
        [row["symbol"] for row in (head_a or {}).get("watchlist", [])],
        [row["symbol"] for row in watchlist],
    ]
    combined["three_distinct_orders"] = (
        bool(head_a)
        and head_a["status"] == "completed"
        and len({tuple(order) for order in orders}) == 3
    )
    (launch / "summary.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    say()
    say(
        f"Accepted cards: {len(cards)} | Rejected cards: {len(rejected)} | "
        f"Changed forecasts: {changed}"
    )
    say(f"Entered top 10: {', '.join(summary['entered_top_10']) or 'none'}")
    say(f"Left top 10: {', '.join(summary['left_top_10']) or 'none'}")
    say(f"Live provider responses: {live_calls} | Metered spend: ${budget['actual_usd']:.4f}")
    say(f"Stop reason: {summary['stop_reason']} | Research time: {summary['elapsed_seconds']:.1f}s")
    say(f"Saved run: {run_dir.resolve()}")
    say(f"Demo summary: {launch / 'summary.json'}")
    say("Replay without another provider call:")
    say(
        "  .\\.venv\\Scripts\\python.exe -X utf8 "
        f".\\.venv\\Scripts\\financial-strategy.exe replay {run_id}"
    )
    combined["ground_truth"] = comparison_report(launch, config, combined)
    (launch / "summary.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    say(f"Three distinct M/A/B orders: {'YES' if combined['three_distinct_orders'] else 'NO'}")
    if head_a and head_a["status"] != "completed":
        say("Head A did not complete live research. Its status and audit path are in the summary.")
        return 2
    if not args.offline and not live_calls:
        say(
            "LIVE RESEARCH DID NOT COMPLETE A PROVIDER RESPONSE. "
            "This watchlist is baseline fallback."
        )
        say(f"Inspect {run_dir / 'model_calls.jsonl'} and {run_dir / 'execution.json'}.")
        return 2
    if not cards:
        say("No evidence cards were accepted. Baseline retention is the expected safe behavior.")
    if args.require_differences and not combined["three_distinct_orders"]:
        say("Live research completed, but this sample did not produce three distinct orders.")
        say("The forecasts and evidence are retained in the saved audit.")
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        say("Interrupted. Check runs/.tmp for an unfinished audit before starting another run.")
        raise SystemExit(130) from None
    except Exception as exc:
        say(f"DEMO ERROR: {exc}")
        raise SystemExit(1) from None
