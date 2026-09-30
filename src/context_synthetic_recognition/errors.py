"""Exception hierarchy.

Every error raised on purpose by the package derives from :class:`CSRError`, so applications
(CLI, GUI) can report them as user-facing messages and let genuine bugs surface as tracebacks.
"""


class CSRError(Exception):
    """Base class of all deliberate errors of the package."""


class ConfigError(CSRError, ValueError):
    """An experiment configuration is invalid (unknown key, value out of range, bad format)."""


class DatasetError(CSRError, ValueError):
    """A dataset cannot be loaded or does not match its schema."""


class RegistryError(CSRError, LookupError):
    """A plug-in name is unknown or registered twice."""


class ModelUndefinedError(CSRError, RuntimeError):
    """A stage of the pipeline is undefined for the given data (Definition 1 / Property 1).

    Example: no k is permitted because the smallest class has fewer than two objects.
    """
