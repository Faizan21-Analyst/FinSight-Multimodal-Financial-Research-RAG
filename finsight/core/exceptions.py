"""Project-wide exceptions."""


class FinSightError(Exception):
    """Base class for all FinSight errors."""


class ConfigError(FinSightError):
    pass


class IngestionError(FinSightError):
    pass


class RetrievalError(FinSightError):
    pass


class UnsafeQueryError(FinSightError):
    """Raised when generated SQL fails guardrail validation."""