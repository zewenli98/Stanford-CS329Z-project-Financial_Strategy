"""Build an offline results brief and reconcile costs across all study attempts."""
# ruff: noqa: E501 -- literal HTML rows and embedded styles stay together.

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from io import StringIO
from pathlib import Path

from financial_strategy.audit.writer import verify_manifest


def csv_text(rows):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def table(headers, rows):
    return (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def percent(value):
    return "Unavailable" if value is None else f"{100 * value:.3f}%"


def points(value):
    return "Unavailable" if value is None else f"{100 * value:.3f} pp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--prior-receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("docs/live-five-day-results.html"))
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    report_path = Path(receipt["report_path"])
    report = json.loads(report_path.read_text())
    screening_config = json.loads((report_path.parent / "screening_config.json").read_text())
    verify_manifest(report_path.parent)
    all_records = []
    for source in [*args.prior_receipt, args.receipt]:
        other_path = Path(json.loads(source.read_text())["report_path"])
        verify_manifest(other_path.parent)
        all_records.extend(json.loads(other_path.read_text())["sessions"])
    baseline = {r["session"]: r for r in report["sessions"] if r["policy"] == "baseline"}
    live = [r for r in report["sessions"] if r["policy"] == "scenario"]
    assert len(live) == 5 and all(r["status"] == "completed" for r in live)
    costs = defaultdict(
        lambda: {
            "token_priced_usd": 0.0,
            "uncertain_charge_bound_usd": 0.0,
            "accounted_total_usd": 0.0,
            "normalized_uncached_usd": 0.0,
            "attempted_runs": 0,
        }
    )
    for record in all_records:
        if record["policy"] != "scenario":
            continue
        run_dir = report_path.parent.parent / record["run_id"]
        verify_manifest(run_dir)
        bucket = costs[record["session"]]
        bucket["attempted_runs"] += 1
        budget = json.loads((run_dir / "budget.json").read_text())
        bucket["accounted_total_usd"] += budget["actual_usd"]
        bucket["normalized_uncached_usd"] += budget["normalized_uncached_usd"]
        for line in (run_dir / "model_calls.jsonl").read_text().splitlines():
            call = json.loads(line)
            category = (
                "uncertain_charge_bound_usd" if call.get("usage_estimated") else "token_priced_usd"
            )
            bucket[category] += call.get("actual_usd", 0.0)
    daily_costs = []
    daily_metrics = []
    watchlists = []
    for row in live:
        session = row["session"]
        cost = costs[session]
        assert (
            abs(
                cost["accounted_total_usd"]
                - cost["token_priced_usd"]
                - cost["uncertain_charge_bound_usd"]
            )
            < 1e-8
        )
        assert cost["accounted_total_usd"] <= 2 + 1e-8
        assert cost["normalized_uncached_usd"] <= 2 + 1e-8
        daily_costs.append(
            {"session": session, **cost, "live_run_id": row["run_id"], "daily_cap_usd": 2.0}
        )
        for k in (3, 5, 10):
            metric = row["metrics_by_k"][str(k)]
            reference = baseline[session]["metrics_by_k"][str(k)]
            daily_metrics.append(
                {
                    "session": session,
                    "top_k": k,
                    **metric,
                    "baseline_top_k_regret": reference["top_k_regret"],
                    "regret_improvement": reference["top_k_regret"] - metric["top_k_regret"],
                }
            )
        watchlists.extend({"session": session, **stock} for stock in row["watchlist"])
    total_cost = sum(row["accounted_total_usd"] for row in daily_costs)
    assert total_cost <= 10 + 1e-8
    aggregate = []
    for k in (3, 5, 10):
        rows = [r for r in daily_metrics if r["top_k"] == k]
        fields = (
            "top_k_overlap",
            "mean_selected_range",
            "oracle_mean_range",
            "top_k_regret",
            "baseline_top_k_regret",
            "regret_improvement",
            "ndcg_at_k",
        )
        aggregate.append(
            {"top_k": k, **{name: sum(r[name] for r in rows) / len(rows) for name in fields}}
        )
    assets = args.output.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    consolidated = {
        "source_report": str(report_path),
        "model": screening_config["agents"]["model"],
        "pricing_as_of": screening_config["adaptive"]["pricing_as_of"],
        "prices": screening_config["adaptive"]["prices"],
        "dates": [r["session"] for r in live],
        "aggregate": aggregate,
        "daily_costs": daily_costs,
        "daily_metrics": daily_metrics,
        "watchlists": watchlists,
        "daily_integrity": [{"session": row["session"], **row["integrity"]} for row in live],
        "total_accounted_usd": total_cost,
        "full_universe_mae": {
            "live": sum(r["metrics"]["mae"] for r in live) / len(live),
            "baseline": sum(baseline[r["session"]]["metrics"]["mae"] for r in live) / len(live),
        },
        "cost_note": "Token-priced usage plus conservative bounds for failed/usage-missing calls; not a reconciled invoice.",
        "scope": "Historical live-model pilot, one replicate, frozen retrospective evidence, masked direct identifiers.",
    }
    (assets / "live-five-day-results.json").write_text(json.dumps(consolidated, indent=2))
    for name, rows in [
        ("costs", daily_costs),
        ("metrics", daily_metrics),
        ("watchlists", watchlists),
    ]:
        (assets / f"live-five-day-{name}.csv").write_text(csv_text(rows), encoding="utf-8")
    model = screening_config["agents"]["model"]
    body = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Five-day live-model range study</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f3ed;color:#18343c;font:16px/1.65 Segoe UI,Arial,sans-serif}
main{max-width:1200px;margin:auto;padding:40px 28px 80px}h1{font:46px/1.15 Georgia,serif}h2{margin-top:44px}
.notice{padding:18px 22px;border-left:4px solid #ac7b24;background:#fbf0d9;border-radius:8px}
.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px;margin:18px 0}
th,td{text-align:left;padding:12px 14px;border-bottom:1px solid #d6ded8;white-space:nowrap}th{background:#e7f1ed}
a{color:#096f63}details{padding:16px 20px;background:white;border:1px solid #d6ded8;border-radius:12px;margin:14px 0}
summary{cursor:pointer;font-weight:600}.stat{font-size:23px;color:#096f63}.muted{color:#4c656b}
@media(max-width:600px){main{padding:24px 16px}h1{font-size:34px}}@media print{body{background:white}main{padding:0}details{break-inside:avoid}}
</style></head><body><main><p class='muted'>FINANCIAL STRATEGY · adaptive_range_v1</p>
<h1>Five days with live model inference.</h1>
<p>The target is regular-session (high − low) / low, forecast at 09:00 America/New_York.
The watchlist has ten stocks; top-3 and top-5 use prefixes of the same predicted order.</p>
<div class='notice'>This is a historical pilot on five dates with one model replicate. It is not a prospective trading test.
Frozen evidence was collected retrospectively; direct identifiers are masked, but contextual identity and historical-knowledge risks remain.
All results are descriptive. These dates are now development data.</div>"""
    dates = ", ".join(row["session"] for row in live)
    universe_sizes = ", ".join(
        str(n) for n in sorted({row["metrics"]["forecast_count"] for row in live})
    )
    body += f"<p>Model: <strong>{html.escape(model)}</strong>. Universe sizes: {universe_sizes} stocks. Dates: {dates}.</p>"
    body += f"<p class='stat'>Conservatively accounted model spend: ${total_cost:.6f} / $10 cap</p>"
    body += "<h2>Average watchlist quality</h2><p>Overlap compares the selected first k stocks with the realized best k in the full universe. Lower regret is better; positive improvement means lower regret than the historical baseline. pp means percentage points.</p>"
    body += table(
        [
            "Cutoff",
            "Overlap",
            "Selected range",
            "Oracle range",
            "Live regret",
            "Baseline regret",
            "Improvement",
            "NDCG",
        ],
        [
            [
                f"Top {r['top_k']}",
                percent(r["top_k_overlap"]),
                percent(r["mean_selected_range"]),
                percent(r["oracle_mean_range"]),
                points(r["top_k_regret"]),
                points(r["baseline_top_k_regret"]),
                points(r["regret_improvement"]),
                f"{r['ndcg_at_k']:.4f}",
            ]
            for r in aggregate
        ],
    )
    body += (
        f"<p>Full-universe mean absolute forecast error: live "
        f"<strong>{points(consolidated['full_universe_mae']['live'])}</strong>; "
        f"historical baseline <strong>{points(consolidated['full_universe_mae']['baseline'])}</strong>. "
        "Each day and each of its twenty stocks have equal weight.</p>"
    )
    body += "<h2>Daily costs, including earlier failed attempts</h2><p>Token-priced cost uses reported token counts and frozen prices. The uncertain-charge column reserves worst-case charges for failed or usage-missing calls; it may exceed what the provider actually billed. Baseline inference cost is zero.</p>"
    body += table(
        ["Session", "Token-priced", "Uncertain bound", "Accounted total", "Run attempts"],
        [
            [
                r["session"],
                f"${r['token_priced_usd']:.6f}",
                f"${r['uncertain_charge_bound_usd']:.6f}",
                f"${r['accounted_total_usd']:.6f}",
                r["attempted_runs"],
            ]
            for r in daily_costs
        ],
    )
    body += "<h2>Research execution</h2><p>New cards exclude carried-forward memory. Numerical changes count revisions and may affect the same stock more than once. The semantic verifier has not been independently audited for accuracy.</p>"
    body += table(
        [
            "Session",
            "Assignments",
            "New cards",
            "Model attempts",
            "Failed calls",
            "Changes",
            "Stop",
        ],
        [
            [
                row["session"],
                row["integrity"]["research_question_attempt_count"],
                row["integrity"]["new_unique_accepted_card_count"],
                row["integrity"]["model_attempt_count"],
                row["integrity"]["failed_model_attempt_count"],
                row["integrity"]["changed_forecast_count"],
                row["integrity"]["stop_reason"],
            ]
            for row in live
        ],
    )
    body += "<h2>Results and watchlists by day</h2>"
    for row in live:
        session = row["session"]
        integrity = row["integrity"]
        body += f"<details open><summary>{session}</summary><p>Stop: {html.escape(str(integrity['stop_reason']))}. New accepted cards: {integrity['new_unique_accepted_card_count']}. Numerical forecast changes across revisions: {integrity['changed_forecast_count']}. Full-universe mean absolute forecast error: {points(row['metrics']['mae'])}.</p>"
        body += table(
            ["Cutoff", "Overlap", "Selected range", "Live regret", "Baseline regret", "NDCG"],
            [
                [
                    f"Top {r['top_k']}",
                    percent(r["top_k_overlap"]),
                    percent(r["mean_selected_range"]),
                    points(r["top_k_regret"]),
                    points(r["baseline_top_k_regret"]),
                    f"{r['ndcg_at_k']:.4f}",
                ]
                for r in daily_metrics
                if r["session"] == session
            ],
        )
        body += (
            table(
                ["Predicted rank", "Stock", "Predicted range", "Realized range"],
                [
                    [
                        r["rank"],
                        r["symbol"],
                        percent(r["predicted_range"]),
                        percent(r["realized_range"]),
                    ]
                    for r in row["watchlist"]
                ],
            )
            + "</details>"
        )
    body += "<h2>Files and interpretation</h2><p><a href='assets/live-five-day-costs.csv'>Daily costs CSV</a> · <a href='assets/live-five-day-metrics.csv'>Daily top-3/5/10 metrics CSV</a> · <a href='assets/live-five-day-watchlists.csv'>Watchlists CSV</a> · <a href='assets/live-five-day-results.json'>Consolidated JSON</a></p><p>MAE is calculated across all twenty forecasts and is reported once per day. Research cards and numerical changes describe execution, not correctness. Verification rejection rates are proxies, not independently measured factual-error rates.</p>"
    body += f"<p class='muted'>Original immutable evaluation: {html.escape(str(report_path))}. All monetary values are USD.</p></main></body></html>"
    args.output.write_text(body, encoding="utf-8")
    print(
        json.dumps(
            {
                "html": str(args.output.resolve()),
                "aggregate": aggregate,
                "total_accounted_usd": total_cost,
                "daily_costs": daily_costs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
