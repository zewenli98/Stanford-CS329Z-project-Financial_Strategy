"""Run registered architecture comparisons with a durable ten-dollar job ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

from financial_strategy.adaptive.config import AdaptiveConfig
from financial_strategy.adaptive.data import derive_range_labels
from financial_strategy.adaptive.evaluation import (
    evaluate_range_cutoffs,
    load_published_predictions,
    summarize_run_integrity,
)
from financial_strategy.adaptive.service import run_adaptive

ROOT = Path("runs/architecture_lab_20261106")
LOCK = threading.RLock()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, body):
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def registry():
    path = ROOT / "registry.json"
    return read(path) if path.exists() else {"cap_usd": 10.0, "jobs": [], "selection": None}


def priced_config(name, phase, day, mode="live"):
    raw = yaml.safe_load(Path("configs/adaptive_range_100_compact_live.yaml").read_text())
    raw["agents"].update(
        mode=mode,
        cache_path=str(ROOT / "cache"),
        max_total_model_calls=110,
        max_total_tool_calls=220,
        max_model_calls_per_symbol=16,
    )
    raw["task"].update(
        target_session=day,
        decision_cutoff=datetime.fromisoformat(day + "T09:00:00").replace(
            tzinfo=ZoneInfo("America/New_York")
        ),
    )
    raw["adaptive"].update(
        evidence_previous_day_only=True,
        research_candidate_limit=20,
        max_batches=3,
        max_workers=2,
        max_worker_model_calls=4,
        daily_budget_usd=0.8,
        research_protocol="legacy",
        candidate_method="volatility",
        baseline_method="mean20",
        memory_path=None,
        replicate_id=f"{phase}_{name}",
        forecast_blend=1.0,
    )
    if name.startswith("baseline"):
        raw["adaptive"]["policy"] = "baseline"
        if name == "baseline_ewma":
            raw["adaptive"]["baseline_method"] = "ewma10"
    else:
        raw["adaptive"]["policy"] = "scenario"
    if name == "guided":
        raw["adaptive"]["research_protocol"] = "guided"
    if name.startswith("packet"):
        raw["adaptive"].update(research_protocol="packet", max_batches=4)
        raw["agents"]["roles"]["company_evidence"]["max_tokens"] = 8192
    if "hybrid" in name:
        raw["adaptive"]["candidate_method"] = "hybrid"
    if "ewma" in name:
        raw["adaptive"]["baseline_method"] = "ewma10"
    if name.endswith("_terra"):
        for role in ["synthesis", "verifier"]:
            raw["agents"]["roles"][role]["model"] = "openai/gpt-5.6-terra"
        raw["adaptive"].update(daily_budget_usd=3.0, max_workers=2, max_input_bytes=30000)
    if raw["adaptive"]["research_protocol"] != "legacy":
        for role in raw["agents"]["roles"]:
            raw["agents"]["roles"][role]["prompt_version"] += "_lab_v2"
    raw["output"]["runs_path"] = str(ROOT / "runs")
    models = {m["id"]: m for m in read(ROOT / "model_catalog.json")["data"]}
    used = {
        raw["agents"]["roles"][r].get("model") or raw["agents"]["model"]
        for r in raw["agents"]["roles"]
    }
    raw["adaptive"]["prices"] = {
        m: {
            "input_per_million": float(models[m]["pricing"]["prompt"]) * 1e6,
            "output_per_million": float(models[m]["pricing"]["completion"]) * 1e6,
            "request_usd": float(models[m]["pricing"].get("request", 0)),
        }
        for m in used
    }
    raw["adaptive"]["pricing_as_of"] = read(ROOT / "plan.json")["started_at"]
    return AdaptiveConfig.model_validate(raw)


def job(name, phase, day, *, mode="live"):
    ident = f"{phase}_{name}_{day.replace('-', '')}"
    config = priced_config(name, phase, day, mode)
    short = hashlib.sha256(ident.encode()).hexdigest()[:12]
    run_id = f"lab_{short}"
    config = config.model_copy(
        update={
            "adaptive": config.adaptive.model_copy(
                update={
                    "replicate_id": short,
                    "memory_path": ROOT / "m" / short,
                }
            )
        }
    )
    cap = (
        config.adaptive.daily_budget_usd
        if config.adaptive.policy != "baseline" and mode == "live"
        else 0.0
    )
    with LOCK:
        state = registry()
        if any(j["id"] == ident for j in state["jobs"]):
            return None
        committed = sum(j.get("charged_usd", j["reserved_usd"]) for j in state["jobs"])
        if committed + cap > state["cap_usd"] + 1e-12:
            state.setdefault("skipped_jobs", []).append(
                {
                    "id": ident,
                    "reason": "budget",
                    "requested_cap_usd": cap,
                    "committed_usd": committed,
                }
            )
            write(ROOT / "registry.json", state)
            return {"skipped": "budget", "id": ident}
        deadline = datetime.fromisoformat(read(ROOT / "plan.json")["deadline"])
        if (deadline - datetime.now(UTC)).total_seconds() < 180:
            state.setdefault("skipped_jobs", []).append({"id": ident, "reason": "time"})
            write(ROOT / "registry.json", state)
            return {"skipped": "time", "id": ident}
        record = {
            "id": ident,
            "arm": name,
            "phase": phase,
            "session": day,
            "mode": mode,
            "status": "reserved",
            "reserved_usd": cap,
            "started_at": datetime.now(UTC).isoformat(),
            "setup": config.model_dump(mode="json"),
        }
        state["jobs"].append(record)
        write(ROOT / "registry.json", state)
        (ROOT / "configs").mkdir(exist_ok=True)
        (ROOT / "configs" / (ident + ".yaml")).write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
    started = time.monotonic()
    charged = cap
    try:
        run, watch = run_adaptive(config, run_id=run_id)
        predictions = load_published_predictions(run)
        assert len(predictions) == 100 and len(watch) == 10
        # Only after immutable publication: open the realised target-session labels.
        labels, coverage = derive_range_labels(
            config.data.market_bars,
            [config.task.target_session],
            [p["symbol"] for p in predictions],
        )
        actual = {r["symbol"]: r["range"] for r in labels.to_dicts()}
        metrics = evaluate_range_cutoffs(predictions, actual, top_ks=[3, 5, 10])
        integrity = summarize_run_integrity(run, predictions)
        charged = integrity["actual_usd"] or 0.0
        qs = [json.loads(s) for s in (run / "research_results.jsonl").read_text().splitlines() if s]
        outcomes = dict(Counter(q["stop_reason"] for q in qs))
        tools = [t for q in qs for t in q["tool_traces"]]
        context = [
            json.loads(s)
            for s in (run / "snapshot/company_evidence.jsonl").read_text().splitlines()
            if s
        ]
        boundary = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("America/New_York"))
        assert all(
            datetime.fromisoformat(r["available_at"].replace("Z", "+00:00")) < boundary
            and datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")) < boundary
            for r in context
        )
        record.update(
            status="completed",
            run_dir=str(run),
            metrics=metrics,
            integrity=integrity,
            outcomes=outcomes,
            research_attempts=len(qs),
            accepted_assignments=outcomes.get("accepted", 0),
            tool_attempts=len(tools),
            tool_errors=sum("error_code" in t["result"] for t in tools),
            source_timing_violations=0,
            label_coverage=coverage,
            changed_forecasts=sum(p["predicted_range"] != p["baseline_range"] for p in predictions),
        )
    except Exception as exc:
        run = Path(config.output["runs_path"]) / run_id
        if (run / "budget.json").exists():
            charged = read(run / "budget.json")["actual_usd"]
        record.update(status="failed", error=str(exc), run_dir=str(run))
    if charged > cap + 1e-9:
        raise RuntimeError(f"Cost cap breach: {ident}: {charged}>{cap}")
    record.update(
        charged_usd=charged,
        elapsed_seconds=time.monotonic() - started,
        finished_at=datetime.now(UTC).isoformat(),
    )
    with LOCK:
        state = registry()
        state["jobs"][next(i for i, j in enumerate(state["jobs"]) if j["id"] == ident)] = record
        write(ROOT / "registry.json", state)
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in [
                    "id",
                    "status",
                    "charged_usd",
                    "outcomes",
                    "tool_errors",
                    "changed_forecasts",
                    "error",
                ]
            }
        ),
        flush=True,
    )
    return record


def main():
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["smoke", "dev", "holdout", "strong", "confirm"], required=True
    )
    args = parser.parse_args()
    plan = read(ROOT / "plan.json")
    if args.stage == "smoke":
        jobs = [("packet_hybrid", "smoke", plan["development_dates"][0], "scripted")]
    elif args.stage == "dev":
        jobs = [
            (arm, "development", day, "live")
            for arm in [
                "baseline_mean20",
                "baseline_ewma",
                "control",
                "guided",
                "packet_vol",
                "packet_hybrid",
                "packet_hybrid_ewma",
            ]
            for day in plan["development_dates"]
        ]
    elif args.stage == "strong":
        jobs = [
            (arm, "development", day, "live")
            for arm in ["packet_hybrid_terra", "packet_hybrid_ewma_terra"]
            for day in plan["development_dates"]
        ]
    elif args.stage == "confirm":
        jobs = [("control", "holdout", day, "live") for day in plan["holdout_dates"]]
    else:
        state = registry()
        arms = {
            j["arm"]
            for j in state["jobs"]
            if j["phase"] == "development"
            and j["status"] == "completed"
            and not j["arm"].startswith("baseline")
        }
        scores = {
            a: sum(
                j["metrics"]["10"]["ndcg_at_k"]
                for j in state["jobs"]
                if j["arm"] == a and j["phase"] == "development" and j["status"] == "completed"
            )
            / 3
            for a in arms
            if sum(
                j["arm"] == a and j["phase"] == "development" and j["status"] == "completed"
                for j in state["jobs"]
            )
            == 3
        }
        if not scores:
            raise RuntimeError("No complete development arm")
        winner = max(scores, key=scores.get)
        baseline_scores = {
            a: sum(
                j["metrics"]["10"]["ndcg_at_k"]
                for j in state["jobs"]
                if j["arm"] == a and j["phase"] == "development" and j["status"] == "completed"
            )
            / 3
            for a in ["baseline_mean20", "baseline_ewma"]
        }
        all_scores = {**scores, **baseline_scores}
        state["selection"] = {
            "arm": winner,
            "development_scores": scores,
            "frozen_at": datetime.now(UTC).isoformat(),
            "holdout_dates": plan["holdout_dates"],
            "best_overall_development_arm": max(all_scores, key=all_scores.get),
            "baseline_development_scores": baseline_scores,
        }
        write(ROOT / "registry.json", state)
        jobs = [
            (arm, "holdout", day, "live")
            for arm in ["baseline_mean20", "baseline_ewma", winner]
            for day in plan["holdout_dates"]
        ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, a, p, d, mode=m) for a, p, d, m in jobs]
        for f in as_completed(futures):
            result = f.result()
            if result and "skipped" in result:
                print(json.dumps(result), flush=True)
    state = registry()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "jobs": len(state["jobs"]),
                "charged_usd": sum(j.get("charged_usd", j["reserved_usd"]) for j in state["jobs"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
