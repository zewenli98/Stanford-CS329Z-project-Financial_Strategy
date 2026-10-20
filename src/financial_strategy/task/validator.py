"""Fail-closed screening task validation before any agent call."""

from __future__ import annotations

from uuid import uuid4

from financial_strategy.config import MVPConfig
from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.errors import ConfigError
from financial_strategy.schemas import ScreeningTask, SymbolMacroExposure

SUPPORTED_VENUES = {"XNYS", "XNAS"}


def validate_and_freeze_task(
    config: MVPConfig,
    calendar: TradingCalendar,
    universe: dict,
    sector_templates: dict,
) -> tuple[ScreeningTask, list[SymbolMacroExposure], list[str]]:
    """Validate all cheap invariants and materialize complete exposure vectors."""

    task_config = config.task
    if not calendar.is_session(task_config.target_session):
        raise ConfigError(
            "INVALID_TARGET_SESSION",
            f"{task_config.target_session} is not an XNYS session",
        )
    market_open = calendar.session_open(task_config.target_session)
    if task_config.decision_cutoff.tzinfo is None:
        raise ConfigError("NAIVE_CUTOFF", "decision_cutoff must include a timezone")
    if task_config.decision_cutoff >= market_open:
        raise ConfigError(
            "CUTOFF_NOT_PREMARKET",
            f"cutoff {task_config.decision_cutoff} must be before {market_open}",
        )
    if universe.get("snapshot_id") != task_config.universe_snapshot_id:
        raise ConfigError("UNIVERSE_SNAPSHOT_NOT_FOUND", "universe snapshot ID mismatch")
    securities = universe.get("securities")
    if not securities:
        raise ConfigError("EMPTY_UNIVERSE", "universe must contain securities")

    warnings: list[str] = []
    if task_config.top_k > len(securities):
        warnings.append("TOP_K_EXCEEDS_UNIVERSE")

    symbols: set[str] = set()
    exposures: list[SymbolMacroExposure] = []
    required_factors = set(config.ranking.macro.supported_risk_factors)
    template_version = str(sector_templates.get("version", "unknown"))
    templates = sector_templates.get("sectors", {})

    for security in securities:
        symbol = security["symbol"]
        if symbol in symbols:
            raise ConfigError("DUPLICATE_SYMBOL", f"duplicate symbol: {symbol}")
        symbols.add(symbol)
        if security.get("security_type") != "common_stock":
            raise ConfigError("UNSUPPORTED_SECURITY_TYPE", symbol)
        if security.get("venue") not in SUPPORTED_VENUES:
            raise ConfigError("UNSUPPORTED_VENUE", symbol)
        if security.get("country") != "US" or security.get("session_profile") != "us_regular":
            raise ConfigError("UNSUPPORTED_SESSION_PROFILE", symbol)
        if not security.get("permanent_entity_id") or not security.get("sector"):
            raise ConfigError("INCOMPLETE_SECURITY_METADATA", symbol)

        template = templates.get(security["sector"])
        if template is None:
            raise ConfigError("MISSING_SECTOR_TEMPLATE", security["sector"])
        if set(template) != required_factors:
            raise ConfigError(
                "INCOMPLETE_EXPOSURE_TEMPLATE",
                f"{security['sector']} must define exactly {sorted(required_factors)}",
            )
        exposures.append(
            SymbolMacroExposure(
                symbol=symbol,
                sector=security["sector"],
                exposure_method="sector_heuristic_v1",
                template_version=template_version,
                exposures=template,
            )
        )

    task = ScreeningTask(
        task_id=f"task_{uuid4().hex}",
        universe_snapshot_id=task_config.universe_snapshot_id,
        decision_cutoff=task_config.decision_cutoff,
        target_session=task_config.target_session,
        top_k=task_config.top_k,
        price_lookback_sessions=task_config.price_lookback_sessions,
        news_lookback_days=task_config.news_lookback_days,
        max_revision_rounds=config.revision["max_rounds"],
    )
    return task, exposures, warnings
