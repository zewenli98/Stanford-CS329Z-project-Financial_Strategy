"""Command-line entry points."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from financial_strategy.audit.replay import replay_run
from financial_strategy.config import load_tuning_config
from financial_strategy.evaluation.holdout import (
    assess_holdout_promotion,
    mark_holdout_consumed,
    prepare_future_holdout,
    require_holdout_ready,
)
from financial_strategy.evaluation.runner import run_evaluation
from financial_strategy.service import run_screening
from financial_strategy.tuning import (
    prepare_artifacts,
    run_train_cross_validation,
    run_tuning,
)

load_dotenv(override=False)

app = typer.Typer(no_args_is_help=True)


@app.command()
def screen(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create one new immutable screening run."""

    run_dir, watchlist = run_screening(config, run_id=run_id)
    typer.echo(json.dumps({"run_dir": str(run_dir), "watchlist": watchlist}, indent=2))


@app.command()
def replay(
    run_id: str,
    runs_path: Annotated[Path, typer.Option()] = Path("runs"),
) -> None:
    """Verify and rebuild one existing successful run."""

    run_dir = runs_path / run_id
    if not run_dir.exists():
        raise typer.BadParameter(f"RUN_NOT_FOUND: {run_id}")
    typer.echo(json.dumps(replay_run(run_dir), indent=2))


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Create historical predictions, then evaluate their frozen outputs."""

    output, report = run_evaluation(config)
    typer.echo(json.dumps({"report_path": str(output), "report": report}, indent=2))


@app.command("prepare-tuning")
def prepare_tuning(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    split: Annotated[str, typer.Option()] = "all",
    retry_failed: Annotated[bool, typer.Option()] = False,
) -> None:
    """Create one immutable Agent run per requested tuning session."""

    tuning = load_tuning_config(config)
    index = prepare_artifacts(
        tuning,
        limit=limit,
        split_name=split,
        retry_failed=retry_failed,
    )
    typer.echo(json.dumps(index, indent=2))


@app.command()
def tune(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Tune on train artifacts and consume the validation holdout once."""

    output, report = run_tuning(load_tuning_config(config))
    typer.echo(json.dumps({"report_path": str(output), "report": report}, indent=2))


@app.command("cross-validate")
def cross_validate(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Run deterministic train-only nested CV without consuming validation."""

    output, report = run_train_cross_validation(load_tuning_config(config))
    typer.echo(json.dumps({"report_path": str(output), "report": report}, indent=2))


@app.command("prepare-holdout")
def prepare_holdout(
    manifest: Annotated[Path, typer.Option()],
    challenger_config: Annotated[Path, typer.Option(exists=True, readable=True)],
    start_session: Annotated[str, typer.Option()] = "2026-10-15",
    as_of: Annotated[str, typer.Option()] = "2026-10-30",
    minimum_sessions: Annotated[int, typer.Option(min=1)] = 60,
) -> None:
    """Advance an unlabeled future holdout without changing its challenger."""

    result = prepare_future_holdout(
        manifest,
        challenger_config,
        start_session=date.fromisoformat(start_session),
        as_of=date.fromisoformat(as_of),
        minimum_sessions=minimum_sessions,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate-holdout")
def evaluate_holdout(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)],
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Run and irreversibly record one evaluation of a ready future holdout."""

    require_holdout_ready(manifest)
    output, report = run_evaluation(config)
    promotion = assess_holdout_promotion(report)
    ledger = mark_holdout_consumed(
        manifest,
        evaluation_report=output,
        consumed_at=datetime.now(UTC).isoformat(),
        promotion_decision=promotion,
    )
    typer.echo(
        json.dumps(
            {
                "report_path": str(output),
                "report": report,
                "promotion": promotion,
                "holdout": ledger,
            },
            indent=2,
        )
    )
