"""Domain exceptions for repository scanning."""


class RepositoryScanError(ValueError):
    """Raised when a repository scan cannot start."""


class ModelConfigurationError(ValueError):
    """Raised when required runtime model settings are missing."""


class ModelResponseError(ValueError):
    """Raised when model output cannot become a grounded answer."""


class ModelCallError(RuntimeError):
    """Raised when the configured model request fails."""
