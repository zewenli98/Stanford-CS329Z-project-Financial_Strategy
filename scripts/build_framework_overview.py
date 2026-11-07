"""Build the offline team brief from saved engineering artifacts, without model calls."""
# ruff: noqa: E501 -- inline SVG markup keeps literal attributes together.

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs/adaptive_demo_v1"
COMPARISON = ROOT / "runs/adaptive_evaluation_2013f87de535/report.json"
OUTPUT = ROOT / "docs/adaptive-range-v1-overview.html"
ASSETS = ROOT / "docs/assets"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def facts():
    calls = read_rows(DEMO / "model_calls.jsonl")
    decisions = read_rows(DEMO / "allocation_decisions.jsonl")
    cards = read_json(DEMO / "research_cards.json")
    forecasts = read_json(DEMO / "range_predictions.json")
    report = read_json(COMPARISON)
    budget = read_json(DEMO / "budget.json")
    coverage = read_json(ROOT / "runs/range_labels_smoke_v1/coverage.json")
    return {
        "brief_date": "2026-11-04",
        "demo_id": DEMO.name,
        "target_session": read_json(DEMO / "task.json")["target_session"],
        "status": read_json(DEMO / "status.json")["status"],
        "forecasts": len(forecasts),
        "watchlist_size": len(read_json(DEMO / "watchlist.json")),
        "batches": len({row["batch"] for row in decisions}),
        "assignments": len(read_rows(DEMO / "research_results.jsonl")),
        "accepted_cards": len(cards),
        "rejected_cards": len(read_json(DEMO / "rejected_cards.json")),
        "model_invocations": len(calls),
        "operations": dict(Counter(row["payload"]["op"] for row in calls)),
        "explored_dispatches": sum(row["explored"] for row in decisions),
        "changed_forecasts": sum(
            row["baseline_range"] != row["predicted_range"] for row in forecasts
        ),
        "budget": budget,
        "execution": read_json(DEMO / "execution.json"),
        "example_watchlist": [
            {key: row[key] for key in ("rank", "symbol", "baseline_range", "predicted_range")}
            for row in read_json(DEMO / "watchlist.json")[:3]
        ],
        "first_dispatches": [
            {
                key: row[key]
                for key in (
                    "question_id",
                    "batch",
                    "priority",
                    "propensity",
                    "explored",
                    "exploration_probability",
                )
            }
            for row in decisions[:2]
        ],
        "coverage": coverage,
        "comparison": {
            "id": report["evaluation_id"],
            "run_count": len(report["sessions"]),
            "policies": sorted({row["policy"] for row in report["sessions"]}),
            "dates": sorted({row["session"] for row in report["sessions"]}),
            "caps": sorted({row["budget_usd"] for row in report["sessions"]}),
            "failed_runs": sum(row["status"] == "failed" for row in report["sessions"]),
            "scored_runs": sum(row["status"] == "completed" for row in report["sessions"]),
            "missing_label_runs": sum(
                row["status"] == "missing_range_labels" for row in report["sessions"]
            ),
            "actual_usd": sum(row["total_actual_usd"] for row in report["summary"].values()),
            "reference_usd": sum(
                row["total_normalized_uncached_usd"] for row in report["summary"].values()
            ),
        },
        "validation": {
            "tests_passed": 132,
            "receipt": "132 passed in 20.85s",
            "date": "2026-11-04",
            "note": "Last completed implementation validation, recorded before this brief.",
        },
    }


def portable_svg(markup: str) -> str:
    """Inline drawing attributes so exports survive viewers that discard CSS."""
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    root = ET.fromstring(markup)
    root.set("width", "1208")
    root.set("height", "732")
    root.set("version", "1.1")
    root.set("preserveAspectRatio", "xMidYMid meet")
    styles = {
        "node-bg": {"fill": "#f8fbfa", "stroke": "#cadbd7", "stroke-width": "1.5"},
        "n": {"font-size": "15", "font-weight": "700", "fill": "#276b61"},
        "kind": {"font-size": "12", "letter-spacing": ".6"},
        "title": {"font-size": "18", "font-weight": "700"},
        "sub": {"font-size": "13.7", "fill": "#405d63"},
        "line": {
            "fill": "none",
            "stroke": "#829b9c",
            "stroke-width": "2",
            "marker-end": "url(#arrow)",
        },
        "feedback": {"stroke": "#176e64", "marker-end": "url(#arrow-green)"},
        "repair": {"stroke": "#b97724", "marker-end": "url(#arrow-amber)"},
        "label": {"font-size": "14", "fill": "#405d63"},
        "green": {"fill": "#176e64"},
        "amber": {"fill": "#8a571a"},
        "aux": {"fill": "#fffdf8", "stroke": "#d9d5c9", "stroke-width": "1.2"},
    }

    def visit(element, agent=False):
        classes = element.get("class", "").split()
        agent = agent or "agent" in classes
        if element.tag == f"{{{namespace}}}text":
            element.set("font-family", "Segoe UI, Arial, sans-serif")
            element.set("fill", "#17333b")
        for name in classes:
            for attribute, value in styles.get(name, {}).items():
                element.set(attribute, value)
        if agent and "node-bg" in classes:
            element.set("fill", "#eef3fa")
            element.set("stroke", "#cad6e5")
        if element.tag == f"{{{namespace}}}marker":
            element.set("orient", "auto")
        for child in element:
            visit(child, agent)

    visit(root)
    # Only interaction styling remains optional; the drawing needs no stylesheet.
    root.find(f"{{{namespace}}}style").text = """
      #workflow-svg .node{cursor:pointer;outline:none}
      #workflow-svg .node:hover .node-bg,#workflow-svg .node:focus .node-bg,
      #workflow-svg .node.selected .node-bg{stroke:#176e64;stroke-width:3}
      """
    return ET.tostring(root, encoding="unicode", short_empty_elements=False)


def workflow_svg():
    nodes = [
        (
            "freeze",
            32,
            70,
            "01",
            "Freeze eligible inputs",
            "CODE",
            [
                "Cutoff + universe + source copies",
                "Completed bars; no target outcomes",
            ],
        ),
        (
            "baseline",
            326,
            70,
            "02",
            "Initialize range forecasts",
            "CODE",
            [
                "20-session mean range",
                "Shallow coverage of every company",
            ],
        ),
        (
            "plan",
            620,
            70,
            "03",
            "Propose factual questions",
            "LLM",
            [
                "Answers + scenario assumptions",
                "Questions concern available evidence",
            ],
        ),
        (
            "allocate",
            914,
            70,
            "04",
            "Allocate research effort",
            "CODE",
            [
                "Answer sensitivity × resolution / cost",
                "Exploration + exact propensity log",
            ],
        ),
        (
            "publish",
            32,
            286,
            "08",
            "Publish the watchlist",
            "CODE",
            [
                "Full ranking + configurable top-k",
                "Freeze predictions and audit hashes",
            ],
        ),
        (
            "forecast",
            326,
            286,
            "07",
            "Update range forecasts",
            "LLM",
            [
                "Use accepted new cards or corrections",
                "No new evidence → retain forecasts",
            ],
        ),
        (
            "verify",
            620,
            286,
            "06",
            "Verify information cards",
            "CODE + LLM",
            [
                "References, spans, identity, timing",
                "Assess factual support; flag inference",
            ],
        ),
        (
            "research",
            914,
            286,
            "05",
            "Run general researchers",
            "LLM + TOOLS",
            [
                "Up to 2 concurrent ReAct workers",
                "Search → read → card / unresolved",
            ],
        ),
    ]
    content = []
    for key, x, y, number, title, kind, lines in nodes:
        category = "agent" if "LLM" in kind else "code"
        content.append(f'''<g class="node {category}" data-stage="{key}" tabindex="0"
          role="button" aria-label="{number}. {title}">
          <rect class="node-bg" x="{x}" y="{y}" width="260" height="124" rx="13"/>
          <text class="n" x="{x + 17}" y="{y + 27}">{number}</text>
          <text class="kind" x="{x + 243}" y="{y + 27}" text-anchor="end">{kind}</text>
          <text class="title" x="{x + 17}" y="{y + 56}">{title}</text>
          <text class="sub" x="{x + 17}" y="{y + 84}">{lines[0]}</text>
          <text class="sub" x="{x + 17}" y="{y + 106}">{lines[1]}</text></g>''')
    return portable_svg(
        """<svg xmlns="http://www.w3.org/2000/svg" id="workflow-svg" viewBox="0 0 1208 732"
      role="img" aria-labelledby="workflow-title workflow-desc">
      <title id="workflow-title">Adaptive range research workflow</title>
      <desc id="workflow-desc">Eligible inputs produce baseline range forecasts. The planner proposes
      questions, code allocates a bounded batch, general researchers inspect local evidence,
      and verification admits or repairs cards. Accepted evidence informs forecast updates.
      Updated forecasts return to planning. The run publishes an immutable watchlist and audit;
      only then are labels loaded for evaluation. Active memory contains accepted cards only.
      Dollar reservations and model and tool limits apply throughout.</desc>
      <style>
      text{font-family:Segoe UI,Arial,sans-serif;fill:#17333b}
      .node-bg{fill:#f8fbfa;stroke:#cadbd7;stroke-width:1.5}
      .agent .node-bg{fill:#eef3fa;stroke:#cad6e5}
      .node{cursor:pointer;outline:none}.node:hover .node-bg,.node:focus .node-bg,
      .node.selected .node-bg{stroke:#176e64;stroke-width:3}
      .n{font-size:15px;font-weight:700;fill:#276b61}.kind{font-size:12px;letter-spacing:.6px}
      .title{font-size:18px;font-weight:650}.sub{font-size:13.7px;fill:#405d63}
      .line{fill:none;stroke:#829b9c;stroke-width:2;marker-end:url(#arrow)}
      .feedback{stroke:#176e64;marker-end:url(#arrow-green)}
      .repair{stroke:#b97724;marker-end:url(#arrow-amber)}
      .label{font-size:14px;fill:#405d63}.label.green{fill:#176e64}.label.amber{fill:#8a571a}
      .aux{fill:#fffdf8;stroke:#d9d5c9;stroke-width:1.2}
      </style>
      <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#829b9c"/></marker>
      <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#176e64"/></marker>
      <marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#b97724"/></marker>
      </defs>
      <rect width="1208" height="732" rx="18" fill="#ffffff"/>
      <text x="32" y="35" font-size="15" font-weight="600">Logical stages · CODE / LLM labels show who controls each decision</text>
      <path class="line" d="M292 132H326"/><path class="line" d="M586 132H620"/>
      <path class="line" d="M880 132H914"/><path class="line" d="M1044 194V286"/>
      <text class="label" x="1058" y="246">bounded batch</text>
      <path class="line" d="M914 348H880"/><path class="line" d="M620 348H586"/>
      <path class="line" d="M326 348H292"/>
      <text class="label" x="162" y="270" text-anchor="middle">on a stopping condition</text>
      <path class="line feedback" d="M456 286V232H750V194"/>
      <text class="label green" x="603" y="220" text-anchor="middle">new forecasts + research history</text>
      <path class="line repair" d="M835 410V462H1044V410"/>
      <text class="label amber" x="940" y="448" text-anchor="middle">one targeted repair, within limits</text>
      <path class="line" d="M162 410V493"/>
      <text class="label" x="176" y="455">after publication</text>
      <path class="line feedback" d="M750 410V493"/>
      <path class="line feedback" d="M456 493V410"/>
      <rect class="aux" x="32" y="493" width="260" height="104" rx="12"/>
      <text x="49" y="524" font-size="18" font-weight="600">Evaluate + replay</text>
      <text class="sub" x="49" y="549">Load labels after forecasts freeze</text>
      <text class="sub" x="49" y="574">Regret, coverage, cost, integrity</text>
      <rect class="aux" x="326" y="493" width="554" height="104" rx="12"/>
      <text x="344" y="524" font-size="18" font-weight="600">Accepted-card memory</text>
      <text class="sub" x="344" y="549">7-day active view · isolated by policy / budget / replicate</text>
      <text class="sub" x="344" y="574">Expiry affects retrieval; source and card history stay auditable</text>
      <text class="label" x="930" y="533">Rejected / unresolved?</text>
      <text class="label" x="930" y="558">No evidence-based update.</text>
      <text class="label" x="930" y="583">Keep the latest valid forecast.</text>
      <rect x="32" y="633" width="1142" height="69" rx="12" fill="#eaf2ef"/>
      <text x="51" y="660" font-size="17" font-weight="600">GLOBAL CONTROLS · $2 default daily model cap · finalization funds reserved first</text>
      <text class="sub" x="51" y="684">Atomic reservations · every retry counted · local tool allowlist · model / tool / batch limits · immutable run artifacts</text>
      """
        + "".join(content)
        + "</svg>"
    )


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    diagram = workflow_svg()
    (ASSETS / "adaptive-range-v1-workflow.svg").write_text(diagram, encoding="utf-8")
    data = facts()
    template = (ROOT / "scripts/framework_overview_template.html").read_text(encoding="utf-8")
    html = template.replace("{{WORKFLOW_SVG}}", diagram).replace(
        "{{FACTS_JSON}}", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    )
    OUTPUT.write_text(html, encoding="utf-8")
    (ASSETS / "framework-overview-facts.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "html": str(OUTPUT),
                "workflow_svg": str(ASSETS / "adaptive-range-v1-workflow.svg"),
                "bytes": OUTPUT.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
