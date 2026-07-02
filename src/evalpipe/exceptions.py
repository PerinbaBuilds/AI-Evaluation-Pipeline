"""Exception hierarchy for EvalPipe.

Every error raised intentionally by this package derives from
:class:`EvalPipeError`, so callers can catch one type at the boundary.
"""


class EvalPipeError(Exception):
    """Base class for all EvalPipe errors."""


class ConfigError(EvalPipeError):
    """Raised when an evaluation config file is missing or invalid."""


class DatasetError(EvalPipeError):
    """Raised when a dataset cannot be loaded or fails validation."""


class ProviderError(EvalPipeError):
    """Raised when a model provider fails to produce a response."""


class StorageError(EvalPipeError):
    """Raised on persistence failures (unknown run ids, corrupt rows, ...)."""
