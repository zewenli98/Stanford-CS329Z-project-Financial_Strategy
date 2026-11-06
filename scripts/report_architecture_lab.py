"""Build an offline experiment notebook from the complete registered job history."""
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("runs/architecture_lab_20261106")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    state = read(ROOT / "registry.json")
    plan = read(ROOT / "plan.json")
    groups = defaultdict(list)
    for job in state["jobs"]:
        groups[(job["phase"], job["arm"])].append(job)
    summaries = []
    for (phase, arm), jobs in groups.items():
        good = [j for j in jobs if j["status"] == "completed"]
        attempts = sum(j.get("research_attempts", 0) for j in good)
        accepted = sum(j.get("accepted_assignments", 0) for j in good)
        entry = {
            "phase": phase,
            "arm": arm,
            "completed": len(good),
            "registered": len(jobs),
            "failed": sum(j["status"] == "failed" for j in jobs),
            "cost_usd": sum(j.get("charged_usd", j["reserved_usd"]) for j in jobs),
            "ndcg": {
                str(k): mean([j["metrics"][str(k)]["ndcg_at_k"] for j in good]) for k in [3, 5, 10]
            },
            "overlap": {
                str(k): mean([j["metrics"][str(k)]["top_k_overlap"] for j in good])
                for k in [3, 5, 10]
            },
            "research_attempts": attempts,
            "accepted_assignments": accepted,
            "acceptance_rate": accepted / attempts if attempts else None,
            "tool_attempts": sum(j.get("tool_attempts", 0) for j in good),
            "tool_errors": sum(j.get("tool_errors", 0) for j in good),
            "changed_forecasts": sum(j.get("changed_forecasts", 0) for j in good),
            "model_calls": sum(
                j.get("integrity", {}).get("model_call_count", 0) or 0 for j in good
            ),
            "failed_model_attempts": sum(
                j.get("integrity", {}).get("failed_model_attempt_count", 0) or 0 for j in good
            ),
        }
        summaries.append(entry)
    costs = sum(j.get("charged_usd", j["reserved_usd"]) for j in state["jobs"])
    settled = sum(j.get("charged_usd", 0) for j in state["jobs"])
    reserved = sum(j["reserved_usd"] for j in state["jobs"] if "charged_usd" not in j)
    summary = {
        "updated_at": datetime.now(UTC).isoformat(),
        "plan": plan,
        "charged_or_reserved_usd": costs,
        "settled_spend_usd": settled,
        "open_reservations_usd": reserved,
        "cap_usd": 10,
        "selection": state["selection"],
        "experiments": summaries,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    esc = html.escape
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100-stock architecture experiments</title>
<style>body{background:#f7f4ec;color:#252a2b;font:16px/1.65 system-ui,sans-serif;margin:0}main{max-width:1250px;margin:auto;padding:36px 28px}h1{font-size:34px;line-height:1.2}h2{margin-top:40px}p{max-width:950px}.badge{display:inline-block;background:#e1ecdf;padding:8px 16px;border-radius:8px;margin-right:8px}.scroll{overflow:auto}table{border-collapse:collapse;background:#fffdf8;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #ddd9cc;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e9e5d9}summary{cursor:pointer;font-weight:600;padding:8px}details{background:#fffdf8;padding:12px;margin:12px 0;border:1px solid #ded9cb;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.5}a{color:#315950}.note{border-left:4px solid #bd8e46;background:#f2e7cf;padding:12px 18px}small{color:#62645e}@media print{details{break-inside:avoid}body{background:white}.scroll{overflow:visible}}</style><main>
<h1>100-stock architecture experiments</h1>"""
    ]
    parts.append(
        f'<p><span class="badge">${settled:.4f} settled charges</span><span class="badge">${reserved:.4f} open reservations / $10 cap</span><span class="badge">{len(state["jobs"])} registered jobs</span></p>'
    )
    if (ROOT / "conclusions.json").exists():
        final = read(ROOT / "conclusions.json")
        candidate = final["holdout"][final["selected_using_development_only"]]
        control = final["holdout"]["control"]
        parts.append(
            '<p class="note"><b>Result:</b> the new architecture collected more accepted evidence '
            'and eliminated recorded source-reading errors, but did not improve holdout '
            f'NDCG@10: {candidate["ndcg"]["10"]:.5f} versus '
            f'{control["ndcg"]["10"]:.5f} for the original agent. '
            'Both reached 35% top-10 coverage. The stronger model is not promoted to the default.</p>'
        )
    parts.append(
        "<p>This notebook includes successful, failed, and still-running experiments. Every live job reserves its full daily spending cap before starting. Missing or interrupted settlements remain conservatively charged at that cap. Costs use recorded token usage and conservative charges for failed requests; they are not a provider invoice reconciliation.</p>"
    )
    parts.append(
        "<h2>Design and information boundary</h2><p>All variants use the same fixed 100-stock universe and produce 100 forecasts plus a ten-stock watchlist. Company evidence must have both publication and availability timestamps strictly before New York midnight at the start of the target date. Low-confidence macro records are excluded. Target-day prices are used only after forecast publication.</p>"
    )
    parts.append(
        "<p><b>Metric definitions:</b> the realized range is (daily high − daily low) / daily low. "
        "At each rank, NDCG uses that realized range divided by log2(rank + 1), adds the first k "
        "terms, and divides by the same sum for the ideal ranking. It uses linear range relevance, "
        "not exponential relevance. Coverage@k is the fraction of the actual top-k names also in "
        "the predicted top-k. Capture by the ten-name watchlist is reported separately below. "
        "All displayed aggregate metrics average sessions equally.</p>"
    )
    parts.append(
        f"<p><b>Development dates:</b> {esc(', '.join(plan['development_dates']))}.<br><b>Preselected holdout dates:</b> {esc(', '.join(plan['holdout_dates']))}.</p>"
    )
    parts.append(
        '<p class="note">Development results are used to choose a candidate. Holdout results are reported separately and do not establish statistical significance. The current-constituent universe has survivorship bias, and vendor history can incorporate later adjustments. This remains a public-data research backtest.</p>'
    )
    parts.append(
        "<details><summary>Read the preregistered plan</summary><pre>"
        + esc(json.dumps(plan, indent=2))
        + "</pre></details>"
    )
    parts.append(
        "<h2>Architecture key</h2><ul>"
        "<li><b>control:</b> existing compact planner and tool-using researchers.</li>"
        "<li><b>guided:</b> safer identity masking, explicit tool contracts, substantive default "
        "reads, a reserved repair attempt, and verifier access to the question and baseline.</li>"
        "<li><b>packet_vol:</b> deterministic, dated document passages supplied before model "
        "extraction; no model planner; exact source spans remain mandatory.</li>"
        "<li><b>hybrid:</b> ten volatility candidates plus ten recent-event candidates; unused "
        "event slots are filled from volatility ranks.</li>"
        "<li><b>ewma:</b> training-selected weighted history: each completed range receives "
        "weight 2/11 and the previous estimate receives 9/11, initialized at the oldest of "
        "the 60 available history sessions.</li>"
        "<li><b>terra:</b> Luna extracts evidence; Terra verifies and forecasts. Up to eight "
        "assignments, matching the corresponding Luna packet arm.</li></ul>"
        "<p>Live generation uses provider defaults, without independently seeded sampling "
        "replicates. Exact replay reuses recorded responses; it is not a new live replication.</p>"
    )
    train = ROOT / "deterministic_training.json"
    if train.exists():
        training = read(train)
        parts.append(
            f'<h2>Historical baseline selection</h2><p>Five baselines compared on {training["session_count"]} training dates, {training["start"]} through {training["end"]}. Selected by mean training NDCG@10: <b>{training["selected_method"]}</b>. This selection precedes the holdout.</p><div class="scroll"><table><tr><th>Method</th><th>NDCG@3</th><th>NDCG@5</th><th>NDCG@10</th><th>Coverage@10</th></tr>'
        )
        for method, values in training["summary"].items():
            parts.append(
                f"<tr><td>{method}</td>"
                + "".join(f"<td>{values[str(k)]['ndcg_at_k']:.5f}</td>" for k in [3, 5, 10])
                + f"<td>{values['10']['top_k_overlap']:.1%}</td></tr>"
            )
        parts.append("</table></div>")
    event_path = ROOT / "event_baseline_training.json"
    if event_path.exists():
        event = read(event_path)
        parts.append(
            "<details><summary>Additional training-only event heuristic</summary><p>A small "
            "grid tested a dated-event uplift to the weighted-history forecast. It was "
            "recorded as an exploratory training experiment and was not promoted into "
            "the live holdout candidate.</p><pre>"
            + esc(json.dumps({k: v for k, v in event.items() if k != "daily"}, indent=2))
            + "</pre></details>"
        )
    for phase in ["development", "holdout", "smoke"]:
        entries = [s for s in summaries if s["phase"] == phase]
        if not entries:
            continue
        parts.append(
            f'<h2>{phase.capitalize()} comparisons</h2><div class="scroll"><table><tr><th>Arm</th><th>Done/jobs</th><th>NDCG@3</th><th>NDCG@5</th><th>NDCG@10</th><th>Cover@3</th><th>Cover@5</th><th>Cover@10</th><th>Accepted/questions</th><th>Tool errors</th><th>Model attempts</th><th>Cost</th></tr>'
        )
        for s in entries:
            cells = (
                [f"{s['completed']}/{s['registered']}"]
                + [
                    f"{s['ndcg'][str(k)]:.5f}" if s["ndcg"][str(k)] is not None else "—"
                    for k in [3, 5, 10]
                ]
                + [
                    f"{s['overlap'][str(k)]:.1%}" if s["overlap"][str(k)] is not None else "—"
                    for k in [3, 5, 10]
                ]
                + [
                    f"{s['accepted_assignments']}/{s['research_attempts']}",
                    f"{s['tool_errors']}/{s['tool_attempts']}",
                    str(s["model_calls"]),
                    f"${s['cost_usd']:.4f}",
                ]
            )
            parts.append(
                "<tr><td>"
                + esc(s["arm"])
                + "</td>"
                + "".join("<td>" + c + "</td>" for c in cells)
                + "</tr>"
            )
        parts.append("</table></div>")
    if state["selection"]:
        parts.append(
            "<h2>Frozen selection</h2><pre>"
            + esc(json.dumps(state["selection"], indent=2))
            + "</pre>"
        )
    if (ROOT / "audit.json").exists():
        audit = read(ROOT / "audit.json")
        replays = [j for j in audit["jobs"] if j["replay"] != "not_requested"]
        parts.append(
            "<h2>Independent audit</h2><p>"
            + str(audit["completed_jobs_audited"])
            + " completed jobs checked: forecast counts, full-universe ranking metrics, "
            "source timestamps, exact cited spans, artifact integrity, and cost settlements. "
            "Original source files are unchanged. Recorded replay checks: "
            + str(len(replays))
            + " exact matches with live-client construction disabled.</p>"
        )
        parts.append(
            '<p>The replay audit discovered and corrected a daylight-saving-time issue. '
            'The saved live results and forecast logic are unchanged. '
            '<a href="replay_audit_discovery.json">Discovery and regression history</a> · '
            '<a href="post_audit_source_changes.json">Exact post-audit changes</a> · '
            '<a href="post_audit_fixes.zip">Replay fix and regression source</a>.</p>'
        )
    if (ROOT / "conclusions.json").exists():
        conclusions = read(ROOT / "conclusions.json")
        selected = conclusions["selected_using_development_only"]
        parts.append(
            "<h2>Final assessment</h2><p>The frozen candidate is <b>"
            + esc(selected)
            + "</b>. All configurations remain opt-in. The following comparisons use only dates "
            "shared with the candidate. Positive NDCG differences favor the candidate.</p>"
            '<div class="scroll"><table><tr><th>Comparator</th><th>Paired dates</th>'
            "<th>Candidate Δ NDCG@10</th><th>Wins / ties / losses</th></tr>"
        )
        for arm, result in conclusions["paired_ndcg10_comparisons"].items():
            parts.append(
                f"<tr><td>{esc(arm)}</td><td>{len(result['paired_dates'])}</td>"
                f"<td>{result['mean_difference']:+.5f}</td>"
                f"<td>{result['wins']} / {result['ties']} / {result['losses']}</td></tr>"
            )
        parts.append("</table></div>")
        parts.append('<p class="note">' + esc(conclusions["recommendation"]) + "</p>")
        suite = conclusions.get("final_tests", {})
        if suite:
            parts.append(
                f'<p><b>Final full test suite:</b> {suite["tests"]} tests, '
                f'{suite["failures"]} failures, {suite["errors"]} errors, '
                f'{suite["skipped"]} skipped. <a href="final_tests.xml">Test record</a>.</p>'
            )
        parts.append(
            '<h2>Ten-name watchlist capture and research completion</h2><div class="scroll">'
            "<table><tr><th>Arm</th><th>Actual top-3 captured</th><th>Actual top-5 captured</th>"
            "<th>Actual top-10 captured</th><th>Accepted / questions</th>"
            "<th>Accepted or clean abstention</th><th>Failed model attempts</th></tr>"
        )
        for arm, values in conclusions["holdout"].items():
            rate = values["accepted_or_clean_abstention_per_question"]
            rate_text = f"{rate:.1%}" if rate is not None else "—"
            parts.append(
                "<tr><td>"
                + esc(arm)
                + "</td>"
                + "".join(
                    f"<td>{values['watchlist10_capture_actual_top_k'][str(k)]:.1%}</td>"
                    for k in [3, 5, 10]
                )
                + f"<td>{values['research_outcomes'].get('accepted', 0)} / {values['research_questions']}</td>"
                + f"<td>{rate_text}</td><td>{values['failed_model_attempts']}</td></tr>"
            )
        parts.append(
            "</table></div><p>Clean abstention means the archive did not support a material card "
            "and no tool error occurred. It is a completed research decision, but contributes no "
            "accepted evidence. A rejected repair is unresolved research even when the daily run "
            "successfully publishes its baseline-supported forecasts.</p>"
        )
        parts.append(
            "<details><summary>Complete assessment, daily paired differences, and next priorities</summary><pre>"
            + esc(json.dumps(conclusions, indent=2))
            + "</pre></details>"
        )
    if (ROOT / "manual_process_review.json").exists():
        review = read(ROOT / "manual_process_review.json")
        parts.append("<h2>Manual process review</h2><p>" + esc(review["method"]) + "</p>")
        for case in review["cases"]:
            parts.append(
                "<details open><summary>"
                + esc(case["company"] + " · " + case["outcome"])
                + "</summary><p>"
                + esc(case["finding"])
                + "</p><p><b>Assessment:</b> "
                + esc(case["assessment"])
                + "</p></details>"
            )
        parts.append("<p><b>Remaining process issues:</b></p><ul>")
        for issue in review["remaining_issues"]:
            parts.append("<li>" + esc(issue["finding"] + " " + issue["implication"]) + "</li>")
        parts.append('</ul><p class="note">' + esc(review["validity_scope"]) + "</p>")
    parts.append(
        "<h2>Every setup and result</h2><p>Expand a row to inspect the complete configuration, cost settlement, metrics, and failure details. Full source reads, model responses, evidence cards, and forecasts are stored in the corresponding run directory.</p>"
    )
    parts.append(
        '<p>Download records: <a href="registry.json">complete experiment ledger</a> · '
        '<a href="summary.json">summary data</a> · <a href="audit.json">independent audit</a> · '
        '<a href="manual_process_review.json">manual review</a> · '
        '<a href="starting_source.zip">starting source</a> · '
        '<a href="tested_runtime.zip">tested runtime</a> · '
        '<a href="experiment_scripts.zip">experiment scripts and dependencies</a> · '
        '<a href="environment.json">environment and reproduction notes</a> · '
        '<a href="runtime_changes.patch">runtime changes</a> · '
        '<a href="model_catalog.json">dated model prices</a> · '
        '<a href="tests.xml">test results</a>.</p>'
    )
    if (ROOT / "conclusions.json").exists():
        parts.append(
            '<p>Reusable configurations: <a href="../../configs/adaptive_range_lab_candidate.yaml">'
            'frozen agent candidate</a> · <a href="../../configs/adaptive_range_lab_ewma.yaml">'
            'free weighted-history baseline</a>. Update the saved historical task date deliberately '
            'before a new run.</p>'
        )
    for j in state["jobs"]:
        label = f"{j['phase']} · {j['arm']} · {j['session']} · {j['status']} · ${j.get('charged_usd', j['reserved_usd']):.4f}"
        links = '<p><a href="configs/' + esc(j["id"]) + '.yaml">Complete configuration</a>'
        if j.get("run_dir"):
            relative = Path(j["run_dir"]).relative_to(ROOT).as_posix()
            for artifact, title in [
                ("range_predictions.json", "All forecasts"),
                ("research_results.jsonl", "Research outcomes"),
                ("research_cards.json", "Accepted evidence"),
                ("model_calls.jsonl", "Model requests and responses"),
            ]:
                links += ' · <a href="' + esc(relative + "/" + artifact) + '">' + title + "</a>"
        links += "</p>"
        parts.append(
            "<details><summary>"
            + esc(label)
            + "</summary>"
            + links
            + "<pre>"
            + esc(json.dumps(j, indent=2, default=str))
            + "</pre></details>"
        )
    parts.append(
        f"<p><small>Updated {summary['updated_at']}. Sources and configurations are archived alongside this report. NDCG weights rank position; coverage is the overlap between the predicted and actual top-k sets divided by k.</small></p></main></html>"
    )
    (ROOT / "report.html").write_text("".join(parts), encoding="utf-8")
    print(json.dumps({"charged_or_reserved_usd": costs, "experiments": summaries}, indent=2))


if __name__ == "__main__":
    main()
