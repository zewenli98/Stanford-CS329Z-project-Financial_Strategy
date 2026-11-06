"""History-only baselines and deterministic evidence selection for lab variants."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from financial_strategy.adaptive.models import ResearchQuestion

EVENT_WEIGHTS = {
    "earnings": 1.0,
    "guidance": 1.0,
    "merger_acquisition": 0.9,
    "litigation_regulatory": 0.8,
    "cybersecurity": 0.8,
    "management_change": 0.6,
    "financing": 0.6,
    "dividend": 0.15,
}


def event_priority(records, cutoff: datetime):
    scores = defaultdict(float)
    for record in records:
        if record.available_at > cutoff:
            continue
        age = max(0.0, (cutoff - record.available_at).total_seconds() / 86400)
        types = record.event_types or [record.event_type]
        weight = max((EVENT_WEIGHTS.get(t, 0.2) for t in types), default=0.2)
        scores[record.symbol] = max(scores[record.symbol], weight * math.exp(-age / 2))
    return dict(scores)


def history_baselines(features, bars, method):
    if method == "mean20":
        return {s: f.mean_range_20d for s, f in features.items()}
    if method == "median20":
        return {s: f.median_range_20d for s, f in features.items()}
    histories = defaultdict(list)
    for row in bars.sort(["symbol", "session"]).to_dicts():
        if row["symbol"] in features:
            histories[row["symbol"]].append((row["high"] - row["low"]) / row["low"])
    output = {}
    for symbol, values in histories.items():
        if method in {"mean5", "mean10"}:
            n = 5 if method == "mean5" else 10
            output[symbol] = sum(values[-n:]) / n
        elif method == "ewma10":
            mean = values[0]
            for value in values[1:]:
                mean = (2 / 11) * value + (9 / 11) * mean
            output[symbol] = mean
        else:
            raise ValueError(method)
    return output


def select_candidates(baselines, records, cutoff, limit, method):
    ordered = sorted(baselines, key=lambda s: (-baselines[s], s))
    if limit is None:
        return ordered
    if method == "volatility":
        return ordered[:limit]
    keep = ordered[: max(1, limit // 2)]
    scores = event_priority(records, cutoff)
    for symbol in sorted(scores, key=lambda s: (-scores[s], -baselines.get(s, 0), s)):
        if symbol in baselines and symbol not in keep:
            keep.append(symbol)
        if len(keep) >= limit:
            return keep[:limit]
    return (keep + [s for s in ordered if s not in keep])[:limit]


def packet_questions(view, forecasts, inspected, reserved_cost):
    scores = event_priority(view.snapshot.company_evidence, view.cutoff)
    by_alias = {f.symbol: f for f in forecasts}
    symbols = [
        s for s in view.aliases if scores.get(s, 0) > 0 and not inspected.get(view.aliases[s], 0)
    ]
    symbols.sort(key=lambda s: (-scores[s], -by_alias[view.aliases[s]].predicted_range, s))
    questions = []
    for symbol in symbols:
        alias = view.aliases[symbol]
        leads = [r for r in view.lead_index() if r["symbol"] == alias]
        questions.append(
            ResearchQuestion(
                question_id=f"packet_{alias}",
                text=(
                    "Extract a material new catalyst and counterevidence from dated passages. "
                    "Distinguish announcement time from future effective dates. Assess whether any "
                    "uncertainty remains for the target date; abstain on metadata-only material."
                ),
                symbols=[alias],
                evidence_ids=[r["evidence_id"] for r in leads],
                event_ids=list(dict.fromkeys(r["event_id"] for r in leads)),
                resolution_probability=min(1.0, scores[symbol]),
                discovery=True,
                allowed_tools=["read"],
                reserved_cost_usd=reserved_cost,
            )
        )
    return questions
