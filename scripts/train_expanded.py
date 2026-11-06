"""Train an expanded-universe deterministic ranker without running LLM workers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from financial_strategy.config import load_mvp_config
from financial_strategy.evaluation.labels import RankLabelMode, load_rank_labels
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.features.engine import compute_features
from financial_strategy.ranking.ranker import (
    MARKET_FEATURE_FIELDS,
    average_percentiles,
    market_feature_percentiles,
)
from financial_strategy.schemas import CorporateActionRecord
from financial_strategy.tuning.learning import (
    fit_nonnegative_linear_ranker,
    fit_pairwise_ranker,
    transform_rank_target,
)
from financial_strategy.tuning.runner import _block_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
LARGE_MOVE_THRESHOLD = 0.05


def build_feature_cache(
    dataset: Path,
    output: Path,
    label_mode: RankLabelMode = "open_to_close",
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]], dict]:
    bars = pl.read_parquet(dataset / "market/daily_bars.parquet")
    universe = json.loads((dataset / "universe/symbols.json").read_text())
    split = json.loads((dataset / "splits.json").read_text())
    actions = [
        CorporateActionRecord.model_validate(json.loads(line))
        for line in (dataset / "market/corporate_actions.jsonl").read_text().splitlines()
        if line
    ]
    sectors = {
        item["symbol"]: item["sector"] for item in universe["securities"]
    }
    labels = load_rank_labels(dataset, label_mode)
    source_sessions = sorted(bars["session"].unique().to_list())
    session_position = {session: index for index, session in enumerate(source_sessions)}
    target_sessions = [
        date.fromisoformat(value)
        for value in [*split["train_sessions"], *split["validation_sessions"]]
    ]
    action_by_symbol: dict[str, list[CorporateActionRecord]] = {}
    for action in actions:
        action_by_symbol.setdefault(action.symbol, []).append(action)

    vectors: dict[str, dict[str, np.ndarray]] = {}
    cache_rows: list[dict] = []
    rejected_counts: dict[str, int] = {}
    for target in target_sessions:
        position = session_position[target]
        history_sessions = source_sessions[max(0, position - 61) : position]
        frame = bars.filter(pl.col("session").is_in(history_sessions))
        adjusted = _adjust_window(frame, action_by_symbol, target)
        features, rejected = compute_features(
            adjusted,
            history_sessions[-21:],
            required_sessions=21,
            symbol_sectors=sectors,
        )
        for reason in rejected.values():
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
        percentile_maps = market_feature_percentiles(features)
        session_text = target.isoformat()
        vectors[session_text] = {
            symbol: np.asarray(
                [percentile_maps[field][symbol] for field in MARKET_FEATURE_FIELDS],
                dtype=float,
            )
            for symbol in features
        }
        for symbol, feature in features.items():
            cache_rows.append(
                {
                    "target_session": target,
                    "symbol": symbol,
                    **{
                        field: getattr(feature, field)
                        for field in MARKET_FEATURE_FIELDS
                    },
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cache_rows).write_parquet(output)
    diagnostics = {
        "target_session_count": len(target_sessions),
        "feature_row_count": len(cache_rows),
        "rejected_feature_windows": rejected_counts,
        "minimum_symbols_per_session": min(map(len, vectors.values())),
        "maximum_symbols_per_session": max(map(len, vectors.values())),
    }
    return vectors, labels, diagnostics


def _adjust_window(
    frame: pl.DataFrame,
    actions: dict[str, list[CorporateActionRecord]],
    cutoff_session: date,
) -> pl.DataFrame:
    rows = frame.to_dicts()
    for row in rows:
        ratio = 1.0
        for action in actions.get(row["symbol"], []):
            effective_date = action.effective_at.date()
            if effective_date <= cutoff_session and row["session"] < effective_date:
                ratio *= action.ratio
        if ratio != 1.0:
            for field in ("open", "high", "low", "close"):
                row[field] /= ratio
            row["volume"] *= ratio
        row["adjustment_policy"] = "split_adjusted"
        row["snapshot_id"] = f"expanded_features_{cutoff_session.isoformat()}"
    return pl.DataFrame(rows)


def _load_feature_cache(
    dataset: Path,
    cache_path: Path,
    label_mode: RankLabelMode = "open_to_close",
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]], dict]:
    cache = pl.read_parquet(cache_path)
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for session in sorted(cache["target_session"].unique().to_list()):
        frame = cache.filter(pl.col("target_session") == session)
        symbols = frame["symbol"].to_list()
        percentile_maps: dict[str, dict[str, float]] = {}
        for field in MARKET_FEATURE_FIELDS:
            values = dict(zip(symbols, frame[field].to_list(), strict=True))
            percentile_maps[field] = (
                average_percentiles(values)
                if all(value is not None for value in values.values())
                else {symbol: 0.5 for symbol in symbols}
            )
        vectors[session.isoformat()] = {
            symbol: np.asarray(
                [percentile_maps[field][symbol] for field in MARKET_FEATURE_FIELDS],
                dtype=float,
            )
            for symbol in symbols
        }
    labels = load_rank_labels(dataset, label_mode)
    diagnostics = {
        "target_session_count": len(vectors),
        "feature_row_count": cache.height,
        "rejected_feature_windows": {},
        "minimum_symbols_per_session": min(map(len, vectors.values())),
        "maximum_symbols_per_session": max(map(len, vectors.values())),
        "loaded_from_cache": True,
    }
    return vectors, labels, diagnostics


def train_expanded(
    config_path: Path,
    *,
    label_mode: RankLabelMode = "open_to_close",
    output_dir: Path | None = None,
    feature_cache_path: Path | None = None,
    candidate_path: Path | None = None,
    use_train_winner: bool = False,
) -> tuple[Path, dict]:
    expansion = yaml.safe_load(config_path.read_text())
    dataset = ROOT / expansion["output_path"]
    split = json.loads((dataset / "splits.json").read_text())
    train_sessions = list(split["train_sessions"])
    output_dir = (output_dir or ROOT / "runs/expanded_training").resolve()
    feature_cache = (
        feature_cache_path or output_dir / "feature_cache.parquet"
    ).resolve()
    candidate_path = (
        candidate_path or ROOT / "configs/screening_expanded_candidate.yaml"
    ).resolve()
    if feature_cache.exists():
        vectors, labels, feature_diagnostics = _load_feature_cache(
            dataset,
            feature_cache,
            label_mode,
        )
    else:
        vectors, labels, feature_diagnostics = build_feature_cache(
            dataset,
            feature_cache,
            label_mode,
        )
    base = load_mvp_config(ROOT / "configs/screening_train_validation.yaml")
    anchor = _anchor_weights(base)
    baseline = _baseline_metrics(
        train_sessions,
        vectors,
        labels,
        base,
        expansion["top_k"],
        LARGE_MOVE_THRESHOLD,
    )
    variants: dict[str, dict] = {}
    specifications = (
        ("pairwise_percentile_s005", "pairwise", "session_percentile", 0.05),
        ("pairwise_percentile_s010", "pairwise", "session_percentile", 0.10),
        ("pairwise_percentile_s025", "pairwise", "session_percentile", 0.25),
        ("pairwise_percentile_s050", "pairwise", "session_percentile", 0.50),
        ("linear_percentile", "linear", "session_percentile", 0.25),
        (
            "linear_large_move",
            "linear",
            "large_move_then_magnitude",
            0.25,
        ),
    )
    for name, learner, target_scheme, shrinkage in specifications:
        metrics: dict[str, dict] = {}
        fold_weights = []
        for test_start in range(41, len(train_sessions), 20):
            inner = train_sessions[: test_start - 1]
            test = train_sessions[test_start : test_start + 20]
            weights = _fit(
                inner,
                vectors,
                labels,
                anchor,
                learner,
                target_scheme,
                shrinkage,
                LARGE_MOVE_THRESHOLD,
            )
            fold_weights.append(weights.tolist())
            metrics.update(
                _evaluate(
                    test,
                    vectors,
                    labels,
                    weights,
                    expansion["top_k"],
                    LARGE_MOVE_THRESHOLD,
                )
            )
        delta = _paired_delta(metrics, baseline)
        fold_deltas = [
            float(
                np.mean(
                    [
                        metrics[session]["ndcg_at_k"]
                        - baseline[session]["ndcg_at_k"]
                        for session in train_sessions[start : start + 20]
                        if session in metrics
                    ]
                )
            )
            for start in range(41, len(train_sessions), 20)
        ]
        variants[name] = {
            "learner": learner,
            "target_scheme": target_scheme,
            "shrinkage": shrinkage,
            "paired_ndcg_delta": delta,
            "fold_deltas": fold_deltas,
            "winning_folds": int(sum(value > 0 for value in fold_deltas)),
            "fold_count": len(fold_deltas),
            "fold_weights": fold_weights,
        }
    winner = max(
        variants,
        key=lambda name: (
            variants[name]["winning_folds"],
            min(variants[name]["fold_deltas"]),
            variants[name]["paired_ndcg_delta"]["mean"],
            name,
        ),
    )
    specification = next(item for item in specifications if item[0] == winner)
    final_weights = _fit(
        train_sessions,
        vectors,
        labels,
        anchor,
        specification[1],
        specification[2],
        specification[3],
        LARGE_MOVE_THRESHOLD,
    )
    final_metrics = _evaluate(
        train_sessions,
        vectors,
        labels,
        final_weights,
        expansion["top_k"],
        LARGE_MOVE_THRESHOLD,
    )
    winner_ci = variants[winner]["paired_ndcg_delta"]["95ci"]
    mean_return_non_degrading = _mean_metric(
        final_metrics,
        "mean_absolute_return_at_k",
    ) >= _mean_metric(baseline, "mean_absolute_return_at_k")
    coverage_non_degrading = _mean_metric(
        final_metrics,
        "selection_coverage",
    ) >= _mean_metric(baseline, "selection_coverage")
    promotable = bool(
        winner_ci[0] > 0
        and variants[winner]["winning_folds"] > variants[winner]["fold_count"] / 2
        and min(variants[winner]["fold_deltas"]) > -0.02
        and mean_return_non_degrading
        and coverage_non_degrading
    )
    coarse_recall = _coarse_recall(
        train_sessions,
        vectors,
        labels,
        final_weights,
        expansion["deep_research_candidates"],
        expansion["top_k"],
    )
    sectors = {
        item["symbol"]: item["sector"]
        for item in json.loads(
            (dataset / "universe/symbols.json").read_text()
        )["securities"]
    }
    report = {
        "version": 2,
        "dataset": str(dataset),
        "label_mode": label_mode,
        "train_session_count": len(train_sessions),
        "symbol_count": len(next(iter(vectors.values()))),
        "top_k": expansion["top_k"],
        "large_move_threshold": LARGE_MOVE_THRESHOLD,
        "deep_research_candidates": expansion["deep_research_candidates"],
        "feature_diagnostics": feature_diagnostics,
        "baseline": _summary(baseline),
        "winner": winner,
        "winner_weights": dict(
            zip(MARKET_FEATURE_FIELDS, final_weights.tolist(), strict=True)
        ),
        "winner_metrics": _summary(final_metrics),
        "industry_subgroups": _industry_subgroups(
            final_metrics,
            labels,
            sectors,
            LARGE_MOVE_THRESHOLD,
        ),
        "variants": variants,
        "coarse_screen_recall_at_k": coarse_recall,
        "top_k_overlap_with_anchor": _top_k_overlap(final_metrics, baseline),
        "mean_return_non_degrading": mean_return_non_degrading,
        "coverage_non_degrading": coverage_non_degrading,
        "promotable": promotable,
        "candidate_weight_source": (
            "train_winner_experimental"
            if use_train_winner and not promotable
            else "promoted_train_winner"
            if promotable
            else "anchor_fallback"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "train_cv_report.json"
    _write_json_atomic(report_path, report)
    _write_candidate_config(
        candidate_path,
        base,
        expansion,
        final_weights if promotable or use_train_winner else anchor,
        dataset,
    )
    return report_path, report


def _anchor_weights(base: object) -> np.ndarray:
    weights = np.zeros(len(MARKET_FEATURE_FIELDS), dtype=float)
    market_total = (
        base.ranking.final_weights.volatility
        + base.ranking.final_weights.anomaly
    )
    volatility_share = base.ranking.final_weights.volatility / market_total
    anomaly_share = base.ranking.final_weights.anomaly / market_total
    values = {
        "median_abs_oc_return_20d": (
            volatility_share
            * 0.5
            * base.ranking.volatility_weights.median_abs_oc_return_20d
        ),
        "realized_volatility_20d": (
            volatility_share
            * 0.5
            * base.ranking.volatility_weights.realized_volatility_20d
        ),
        "return_5d": (
            anomaly_share * 0.5 * base.ranking.anomaly_weights.abs_return_5d
        ),
        "previous_session_relative_volume": (
            anomaly_share
            * 0.5
            * base.ranking.anomaly_weights.previous_session_relative_volume
        ),
        "distance_to_ma20": (
            anomaly_share
            * 0.5
            * base.ranking.anomaly_weights.abs_distance_to_ma20
        ),
    }
    enhanced_volatility = (
        "realized_volatility_5d",
        "realized_volatility_10d",
        "realized_volatility_60d",
        "median_abs_oc_return_5d",
        "median_abs_oc_return_10d",
        "median_abs_oc_return_60d",
        "median_range_20d",
        "atr_14d",
        "market_residual_volatility_20d",
        "industry_residual_volatility_20d",
    )
    enhanced_anomaly = (
        "robust_volume_zscore_20d",
        "momentum_10d",
        "momentum_20d",
        "reversal_1d",
        "volatility_trend_5_20",
    )
    values.update(
        {
            field: volatility_share * 0.5 / len(enhanced_volatility)
            for field in enhanced_volatility
        }
    )
    values.update(
        {
            field: anomaly_share * 0.5 / len(enhanced_anomaly)
            for field in enhanced_anomaly
        }
    )
    for field, value in values.items():
        weights[MARKET_FEATURE_FIELDS.index(field)] = value
    return weights / weights.sum()


def _fit(
    sessions: list[str],
    vectors: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, float]],
    anchor: np.ndarray,
    learner: str,
    target_scheme: str,
    shrinkage: float,
    large_move_threshold: float,
) -> np.ndarray:
    rows = []
    targets = []
    session_ids = []
    for session in sessions:
        symbols = sorted(vectors[session].keys() & labels[session].keys())
        transformed = transform_rank_target(
            np.asarray([labels[session][symbol] for symbol in symbols]),
            target_scheme,
            large_move_threshold=large_move_threshold,
        )
        rows.extend(vectors[session][symbol] for symbol in symbols)
        targets.extend(transformed.tolist())
        session_ids.extend([session] * len(symbols))
    features = np.asarray(rows)
    target = np.asarray(targets)
    if learner == "pairwise":
        return fit_pairwise_ranker(
            features,
            target,
            np.asarray(session_ids),
            anchor,
            shrinkage=shrinkage,
            maximum_pairs_per_session=100,
        )
    return fit_nonnegative_linear_ranker(
        features,
        target,
        anchor,
        shrinkage=shrinkage,
    )


def _baseline_metrics(
    sessions: list[str],
    vectors: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, float]],
    base: object,
    top_k: int,
    large_move_threshold: float,
) -> dict[str, dict]:
    anchor = _anchor_weights(base)
    return _evaluate(
        sessions,
        vectors,
        labels,
        anchor,
        top_k,
        large_move_threshold,
    )


def _evaluate(
    sessions: list[str],
    vectors: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, float]],
    weights: np.ndarray,
    top_k: int,
    large_move_threshold: float,
) -> dict[str, dict]:
    result = {}
    for session in sessions:
        ranked = sorted(
            vectors[session],
            key=lambda symbol: (
                -float(vectors[session][symbol] @ weights),
                symbol,
            ),
        )
        result[session] = evaluate_ranking(
            ranked[:top_k],
            labels[session],
            k=top_k,
            eligible_count=len(labels[session]),
            large_move_threshold=large_move_threshold,
        )
    return result


def _paired_delta(candidate: dict[str, dict], baseline: dict[str, dict]) -> dict:
    differences = [
        candidate[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
        for session in sorted(candidate.keys() & baseline.keys())
    ]
    return {
        "mean": float(np.mean(differences)),
        "95ci": _block_bootstrap_ci(differences, 329, 2000, 5),
        "block_sensitivity": {
            str(block): _block_bootstrap_ci(differences, 329, 2000, block)
            for block in (5, 10, 20)
        },
    }


def _coarse_recall(
    sessions: list[str],
    vectors: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, float]],
    anchor: np.ndarray,
    candidate_count: int,
    top_k: int,
) -> float:
    recalls = []
    for session in sessions:
        candidates = set(
            sorted(
                vectors[session],
                key=lambda symbol: (
                    -float(vectors[session][symbol] @ anchor),
                    symbol,
                ),
            )[:candidate_count]
        )
        actual = set(
            sorted(
                labels[session],
                key=lambda symbol: (-labels[session][symbol], symbol),
            )[:top_k]
        )
        recalls.append(len(candidates & actual) / top_k)
    return float(np.mean(recalls))


def _summary(metrics: dict[str, dict]) -> dict:
    result = {"session_count": len(metrics), "monthly": {}}
    for name in (
        "ndcg_at_k",
        "mean_absolute_return_at_k",
        "precision_at_k",
        "selection_coverage",
    ):
        values = [item[name] for item in metrics.values() if item[name] is not None]
        result[f"average_{name}"] = float(np.mean(values))
        result[f"{name}_95ci"] = _block_bootstrap_ci(values, 329, 2000, 5)
    for month in sorted({session[:7] for session in metrics}):
        monthly = [
            value
            for session, value in metrics.items()
            if session.startswith(month)
        ]
        result["monthly"][month] = {
            name: float(np.mean([item[name] for item in monthly]))
            for name in (
                "ndcg_at_k",
                "mean_absolute_return_at_k",
                "precision_at_k",
                "selection_coverage",
            )
        }
    return result


def _mean_metric(metrics: dict[str, dict], name: str) -> float:
    return float(np.mean([item[name] for item in metrics.values()]))


def _top_k_overlap(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
) -> dict:
    values = []
    for session in sorted(candidate.keys() & baseline.keys()):
        left = set(candidate[session]["selected_symbols"])
        right = set(baseline[session]["selected_symbols"])
        values.append(len(left & right) / max(1, len(left)))
    return {
        "mean_fraction": float(np.mean(values)),
        "95ci": _block_bootstrap_ci(values, 329, 2000, 5),
    }


def _industry_subgroups(
    metrics: dict[str, dict],
    labels: dict[str, dict[str, float]],
    sectors: dict[str, str],
    large_move_threshold: float,
) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    for session, item in metrics.items():
        for symbol in item["selected_symbols"]:
            grouped.setdefault(sectors[symbol], []).append(labels[session][symbol])
    return {
        sector: {
            "selection_count": len(values),
            "mean_absolute_return": float(np.mean(values)),
            "large_move_rate": float(
                np.mean([value >= large_move_threshold for value in values])
            ),
        }
        for sector, values in sorted(grouped.items())
    }


def _write_candidate_config(
    path: Path,
    base: object,
    expansion: dict,
    weights: np.ndarray,
    dataset: Path,
) -> None:
    payload = base.model_dump(mode="json")
    final_session = expansion["evaluation_end"]
    payload["task"].update(
        {
            "universe_snapshot_id": expansion["snapshot_id"],
            "target_session": final_session,
            "decision_cutoff": datetime.combine(
                date.fromisoformat(final_session),
                time(9, 0),
                ZoneInfo("America/New_York"),
            ).isoformat(),
            "top_k": expansion["top_k"],
        }
    )
    relative = dataset.relative_to(ROOT)
    payload["data"].update(
        {
            "universe": str(relative / "universe/symbols.json"),
            "market_bars": str(relative / "market/daily_bars.parquet"),
            "corporate_actions": str(relative / "market/corporate_actions.jsonl"),
            "company_evidence": str(relative / "company/evidence.jsonl"),
            "macro_events": str(relative / "macro/events.jsonl"),
            "macro_documents": str(relative / "macro/documents"),
            "input_adjustment_policy": "raw_unadjusted",
            "snapshot_adjustment_policy": "split_adjusted",
        }
    )
    payload["screening"].update(
        {
            "candidate_multiplier": 1,
            "minimum_candidates": expansion["deep_research_candidates"],
        }
    )
    payload["ranking"]["minimum_score"] = 0.0
    payload["ranking"]["market_feature_weights"] = dict(
        zip(MARKET_FEATURE_FIELDS, weights.tolist(), strict=True)
    )
    payload["ranking"]["final_weights"] = {
        "volatility": 0.5833333333333334,
        "anomaly": 0.4166666666666667,
        "company_event": 0.0,
        "macro": 0.0,
    }
    payload["agents"]["mode"] = "scripted"
    payload["agents"]["model"] = "scripted"
    payload["agents"]["recordings_path"] = str(relative / "llm/recordings")
    payload["output"]["runs_path"] = "runs/expanded"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data_expansion.yaml",
    )
    parser.add_argument(
        "--label-mode",
        choices=("open_to_close", "prev_close_to_close"),
        default="open_to_close",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument(
        "--use-train-winner",
        action="store_true",
        help="Write the train-only winner for an explicitly experimental candidate.",
    )
    args = parser.parse_args()
    path, report = train_expanded(
        args.config,
        label_mode=args.label_mode,
        output_dir=args.output_directory,
        feature_cache_path=args.feature_cache,
        candidate_path=args.candidate_path,
        use_train_winner=args.use_train_winner,
    )
    print(json.dumps({"report": str(path), "promotable": report["promotable"]}))


if __name__ == "__main__":
    main()
