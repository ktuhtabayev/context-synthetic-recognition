"""Plug-in registries.

Metrics, normalizers, k strategies, synthetic-feature encoders, weights, majorizing functions,
decision rules, evaluation protocols and baselines are chosen by name from a :class:`Registry`.
A plug-in may declare a parameter type (a dataclass); the configuration layer validates the
``params`` of a :class:`~context_synthetic_recognition.config.models.PluginSpec` against it and the
GUI builds its parameter form from it.

Third-party packages add plug-ins through entry points, e.g. in their ``pyproject.toml``::

    [project.entry-points."context_synthetic_recognition.metrics"]
    my-metric = "my_package.metrics:my_metric"
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from context_synthetic_recognition.errors import RegistryError

T = TypeVar("T")

ENTRY_POINT_PREFIX = "context_synthetic_recognition."


def normalize_name(name: str) -> str:
    """Canonical registry key: lower case, ``_`` and spaces become ``-``."""
    key = "-".join(name.strip().lower().replace("_", " ").split())
    if not key:
        raise RegistryError("A plug-in name must not be empty.")
    return key


@dataclass(frozen=True)
class PluginInfo(Generic[T]):
    """A registered plug-in with its metadata."""

    name: str
    """Canonical name (see :func:`normalize_name`)."""
    obj: T
    """The registered object (function, class or factory)."""
    summary: str = ""
    """One line shown in listings, tooltips and docs."""
    aliases: tuple[str, ...] = ()
    """Other accepted names."""
    params_type: type | None = None
    """Dataclass describing the plug-in's parameters, or ``None`` if it takes none."""


class Registry(Generic[T]):
    """Name → plug-in mapping of one kind (``metrics``, ``normalizers``, …)."""

    def __init__(self, kind: str) -> None:
        """Create an empty registry; ``kind`` names it in messages and entry-point groups."""
        self.kind = kind
        self._plugins: dict[str, PluginInfo[T]] = {}
        self._aliases: dict[str, str] = {}

    def add(
        self,
        name: str,
        obj: T,
        *,
        summary: str = "",
        aliases: tuple[str, ...] = (),
        params_type: type | None = None,
        replace: bool = False,
    ) -> PluginInfo[T]:
        """Register ``obj`` under ``name`` (and ``aliases``).

        Raises:
            RegistryError: The name or an alias is already taken and ``replace`` is false.
        """
        key = normalize_name(name)
        alias_keys = tuple(normalize_name(alias) for alias in aliases)
        if not replace:
            for taken in (key, *alias_keys):
                if taken in self._plugins or taken in self._aliases:
                    raise RegistryError(f"{self.kind}: '{taken}' is already registered.")
        elif key in self._plugins:
            self._remove(key)
        info = PluginInfo(key, obj, summary, alias_keys, params_type)
        self._plugins[key] = info
        for alias in alias_keys:
            self._aliases[alias] = key
        return info

    def register(
        self,
        name: str,
        *,
        summary: str = "",
        aliases: tuple[str, ...] = (),
        params_type: type | None = None,
    ) -> Callable[[T], T]:
        """Decorator form of :meth:`add`."""

        def decorate(obj: T) -> T:
            self.add(name, obj, summary=summary, aliases=aliases, params_type=params_type)
            return obj

        return decorate

    def info(self, name: str) -> PluginInfo[T]:
        """Return the plug-in registered under ``name`` or one of its aliases.

        Raises:
            RegistryError: Unknown name; the message lists the available ones.
        """
        key = normalize_name(name)
        key = self._aliases.get(key, key)
        try:
            return self._plugins[key]
        except KeyError:
            available = ", ".join(self.names()) or "none"
            raise RegistryError(f"Unknown {self.kind} '{name}'. Available: {available}.") from None

    def get(self, name: str) -> T:
        """Return the registered object for ``name``."""
        return self.info(name).obj

    def names(self) -> list[str]:
        """Canonical names in registration order."""
        return list(self._plugins)

    def load_entry_points(self) -> int:
        """Register the plug-ins of the entry-point group ``context_synthetic_recognition.<kind>``.

        Returns:
            Number of plug-ins added (already registered names are skipped).
        """
        added = 0
        for entry_point in entry_points(group=ENTRY_POINT_PREFIX + self.kind):
            if normalize_name(entry_point.name) in self:
                continue
            self.add(entry_point.name, entry_point.load())
            added += 1
        return added

    def _remove(self, key: str) -> None:
        info = self._plugins.pop(key)
        for alias in info.aliases:
            self._aliases.pop(alias, None)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` (or an alias) is registered."""
        if not isinstance(name, str):
            return False
        key = normalize_name(name)
        return key in self._plugins or key in self._aliases

    def __iter__(self) -> Iterator[PluginInfo[T]]:
        """Iterate over the registered plug-ins in registration order."""
        return iter(self._plugins.values())

    def __len__(self) -> int:
        """Number of registered plug-ins (aliases not counted)."""
        return len(self._plugins)

    def __repr__(self) -> str:
        """Short description listing the registered names."""
        return f"Registry({self.kind!r}, {self.names()!r})"
