"""Domain-specific errors with stable codes for routing and audit records."""


class StrategyError(Exception):
    """Base error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ConfigError(StrategyError):
    """Raised before a task starts when configuration is invalid."""


class DataError(StrategyError):
    """Raised when point-in-time data contracts are violated."""


class AuditError(StrategyError):
    """Raised when immutable artifact guarantees cannot be met."""
