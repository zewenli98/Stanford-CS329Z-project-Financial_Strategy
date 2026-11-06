"""Publish paired holdout results and reusable, unchanged experiment configurations."""

from __future__ import annotations

import difflib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree

import yaml

from financial_strategy.adaptive.config import AdaptiveConfig

ROOT = Path("runs/architecture_lab_20261106")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    state = read(ROOT / "registry.json")
    assert not any(j["status"] == "reserved" for j in state["jobs"])
    audit = read(ROOT / "audit.json")
    assert audit["completed_jobs_audited"] == sum(j["status"] == "completed" for j in state["jobs"])
    selected = state["selection"]["arm"]
    holdout = {}
    for j in state["jobs"]:
        if j["phase"] == "holdout" and j["status"] == "completed":
            holdout.setdefault(j["arm"], {})[j["session"]] = j
    summaries = {}
    for arm, by_date in holdout.items():
        jobs = list(by_date.values())
        ids = {j["id"] for j in jobs}
        checks = [a for a in audit["jobs"] if a["id"] in ids]
        outcomes = Counter()
        for j in jobs:
            outcomes.update(j["outcomes"])
        attempts = sum(j["research_attempts"] for j in jobs)
        resolved = sum(a["resolved_or_cleanly_abstained"] for a in checks)
        failures = Counter()
        for check in checks:
            failures.update(check["model_failure_reasons"])
        summaries[arm] = {
            "dates": sorted(by_date),
            "ndcg": {
                str(k): mean([j["metrics"][str(k)]["ndcg_at_k"] for j in jobs]) for k in [3, 5, 10]
            },
            "coverage_same_k": {
                str(k): mean([j["metrics"][str(k)]["top_k_overlap"] for j in jobs])
                for k in [3, 5, 10]
            },
            "watchlist10_capture_actual_top_k": {
                str(k): mean([a["watchlist10_capture_actual_top_k"][str(k)] for a in checks])
                for k in [3, 5, 10]
            },
            "research_outcomes": dict(outcomes),
            "accepted_per_question": outcomes["accepted"] / attempts if attempts else None,
            "accepted_or_clean_abstention_per_question": resolved / attempts if attempts else None,
            "research_questions": attempts,
            "tool_errors": sum(j["tool_errors"] for j in jobs),
            "tool_calls": sum(j["tool_attempts"] for j in jobs),
            "model_calls": sum(a["paid_live_responses"] for a in checks),
            "failed_model_attempts": sum(failures.values()),
            "failed_provider_or_response_attempts": sum(a["failed_model_calls"] for a in checks),
            "model_failure_reasons": dict(failures),
            "estimated_cost_attempts": sum(a["estimated_cost_calls"] for a in checks),
            "cost_usd": sum(j["charged_usd"] for j in jobs),
            "changed_forecasts": sum(j["changed_forecasts"] for j in jobs),
        }
    comparisons = {}
    for arm, jobs in holdout.items():
        if arm == selected:
            continue
        dates = sorted(set(jobs) & set(holdout[selected]))
        differences = {
            d: holdout[selected][d]["metrics"]["10"]["ndcg_at_k"]
            - jobs[d]["metrics"]["10"]["ndcg_at_k"]
            for d in dates
        }
        comparisons[arm] = {
            "paired_dates": dates,
            "candidate_minus_comparator_ndcg10": differences,
            "mean_difference": mean(list(differences.values())),
            "wins": sum(v > 1e-12 for v in differences.values()),
            "ties": sum(abs(v) <= 1e-12 for v in differences.values()),
            "losses": sum(v < -1e-12 for v in differences.values()),
        }
    spend = sum(j["charged_usd"] for j in state["jobs"])
    candidate = max(holdout[selected].values(), key=lambda j: j["session"])
    for arm in [selected, "baseline_ewma"]:
        reference = candidate if arm == selected else next(iter(holdout[arm].values()))
        config = reference["setup"]
        config["output"]["runs_path"] = "runs"
        config["adaptive"]["memory_path"] = None
        config["adaptive"]["replicate_id"] = (
            "lab_candidate_v1" if arm == selected else "lab_ewma_v1"
        )
        AdaptiveConfig.model_validate(config)
        filename = (
            "adaptive_range_lab_candidate.yaml"
            if arm == selected
            else "adaptive_range_lab_ewma.yaml"
        )
        Path("configs", filename).write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    output = {
        "finished_at": datetime.now(UTC).isoformat(),
        "selected_using_development_only": selected,
        "holdout": summaries,
        "paired_ndcg10_comparisons": comparisons,
        "new_api_spend_usd": spend,
        "unused_credit_usd": 10 - spend,
        "registered_jobs": len(state["jobs"]),
        "job_statuses": dict(Counter(j["status"] for j in state["jobs"])),
        "skipped_jobs": state.get("skipped_jobs", []),
        "recommendation": (
            "Keep the existing default and the candidate as an opt-in experiment. "
            "The original agent slightly outperformed the candidate on holdout "
            "NDCG@10 while costing roughly one sixth as much. "
            "The candidate's holdout NDCG@10 edge over the free weighted-history "
            "baseline is small, and it does not improve top-10 coverage over mean20. "
            "Preserve the source-reading and repair improvements, but require a "
            "larger untouched evaluation before paying for Terra by default."
        ),
        "interpretation": [
            "Candidate selection is frozen before holdout. Comparator results do not retune it.",
            "Use paired six-date holdout results, rather than development gains, "
            "to judge improvement.",
            "Higher NDCG and higher exact top-k overlap can disagree because "
            "they measure different qualities.",
            "A full run can complete while an individual research assignment "
            "is rejected or abstains.",
            "Six holdout sessions and one live trajectory per setup do not "
            "establish a reliable population improvement.",
            "These configurations are opt-in; existing defaults have not been replaced.",
        ],
        "next_priorities": [
            "Prevent ordinary company-name words such as target from being masked in instructions.",
            "Separate publication timestamp from a date-only business event "
            "without inventing precision.",
            "Make card identity and supersession deterministic to avoid "
            "self-supersession schema repairs.",
            "Use a larger untouched date set and live replicates before promotion.",
            "Calibrate event-driven forecast adjustments on training data; "
            "avoid treating old news as a guaranteed volatility shock.",
            "Compact accepted-card context for synthesis: the 30k input bound "
            "blocked one holdout synthesis and retained unchanged baseline forecasts.",
        ],
    }
    (ROOT / "conclusions.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    patch = []
    with zipfile.ZipFile(ROOT / "starting_source.zip") as before:
        old_names = set(before.namelist())
        with zipfile.ZipFile(ROOT / "tested_runtime.zip") as after:
            for name in after.namelist():
                old = (
                    before.read(name).decode("utf-8").splitlines(keepends=True)
                    if name in old_names
                    else []
                )
                new = after.read(name).decode("utf-8").splitlines(keepends=True)
                patch.extend(
                    difflib.unified_diff(
                        old, new, fromfile="before/" + name, tofile="after/" + name
                    )
                )
    (ROOT / "runtime_changes.patch").write_text("".join(patch), encoding="utf-8")
    changes = read(ROOT / "post_audit_source_changes.json")
    with zipfile.ZipFile(ROOT / "post_audit_fixes.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*changes, "tests/test_adaptive_end_to_end.py"]:
            archive.write(name, name)
    final_test_suite = ElementTree.parse(ROOT / "final_tests.xml").getroot().find("testsuite")
    assert final_test_suite is not None
    assert int(final_test_suite.attrib["failures"]) == int(final_test_suite.attrib["errors"]) == 0
    output["final_tests"] = final_test_suite.attrib
    (ROOT / "conclusions.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    environment = {
        "python": sys.version,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "packages": {name: version(name) for name in ["polars", "pydantic", "PyYAML", "pytest"]},
        "reproduce": [
            "Use tested_runtime.zip and the configuration for an individual job.",
            "Restore environment variables locally; credentials are never "
            "included in these artifacts.",
            "The saved task date is historical; update it deliberately before a new run.",
            "Live model sampling is not seeded; a fresh call can differ from the recorded result.",
            "The controller ledger is resumable and skips existing jobs; "
            "preserve the original ledger.",
            "python scripts/audit_architecture_lab.py --replay performs no live calls.",
            "python scripts/finalize_architecture_lab.py rebuilds conclusions and configurations.",
            "python scripts/report_architecture_lab.py rebuilds the offline report.",
        ],
    }
    (ROOT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    with zipfile.ZipFile(ROOT / "experiment_scripts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [
            "setup_architecture_lab",
            "run_architecture_lab",
            "report_architecture_lab",
            "audit_architecture_lab",
            "finalize_architecture_lab",
            "train_range_baselines",
            "train_event_baseline",
            "snapshot_architecture_runtime",
        ]:
            path = Path("scripts", name + ".py")
            archive.write(path, path.as_posix())
        for name in ["pyproject.toml", "uv.lock"]:
            archive.write(name, name)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
