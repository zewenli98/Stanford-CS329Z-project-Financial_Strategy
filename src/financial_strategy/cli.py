"""Command-line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from financial_strategy.audit.replay import replay_run
from financial_strategy.evaluation.runner import run_evaluation
from financial_strategy.service import run_screening

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


@app.command("prepare-range-labels")
def prepare_range_labels(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Derive high/low labels and a coverage report without changing source data."""
    from uuid import uuid4

    from financial_strategy.adaptive.config import load_adaptive_config
    from financial_strategy.adaptive.data import derive_range_labels
    from financial_strategy.audit import RunWriter
    from financial_strategy.config import _load_yaml

    raw = _load_yaml(config)
    screening = load_adaptive_config(raw.get("screening_config", config))
    sessions = raw.get("evaluation", {}).get("sessions", [screening.task.target_session])
    from datetime import date
    sessions = [date.fromisoformat(value) if isinstance(value, str) else value
                for value in sessions]
    universe = json.loads(screening.data.universe.read_text())
    labels, coverage = derive_range_labels(
        screening.data.market_bars, sessions, [s["symbol"] for s in universe["securities"]]
    )
    output = output or Path(screening.output["runs_path"]) / f"range_labels_{uuid4().hex[:10]}"
    writer = RunWriter(output.parent, output.name)
    writer.write_parquet("labels.parquet", labels)
    writer.write_json("coverage.json", coverage)
    path = writer.publish("completed")
    typer.echo(json.dumps({"output": str(path), "coverage": coverage}, indent=2))
