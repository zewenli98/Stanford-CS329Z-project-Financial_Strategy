"""Run a bounded historical live-model study with concise per-session progress."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from financial_strategy.adaptive.evaluation import run_adaptive_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(override=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.receipt.with_suffix(".progress.jsonl")
    if args.receipt.exists() or progress_path.exists():
        parser.error("Receipt already exists; use a new path to avoid accidentally rerunning spend")

    def progress(event):
        event = {"recorded_at": datetime.now(UTC).isoformat(), **event}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
        brief = {key: event[key] for key in ("event", "session", "policy", "run_id")}
        if "integrity" in event:
            brief.update(
                {
                    key: event["integrity"].get(key)
                    for key in (
                        "actual_usd",
                        "normalized_uncached_usd",
                        "changed_forecast_count",
                        "new_unique_accepted_card_count",
                        "stop_reason",
                    )
                }
            )
        print(json.dumps(brief), flush=True)

    output, report = run_adaptive_evaluation(args.config, progress=progress)
    receipt = {"report_path": str(output.resolve()), "summary": report["summary"]}
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"completed": True, "report_path": str(output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
