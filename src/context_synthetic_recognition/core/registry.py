"""Plug-in registries.

Metrics, normalizers, k strategies, synthetic-feature encoders, weights, majorizing functions,
decision rules, evaluation protocols and baselines are chosen by name from a :class:`Registry`.
A plug-in may declare a parameter type — a frozen pydantic dataclass configured with
:data:`PARAMS_CONFIG`; :meth:`Registry.make_params` validates the ``params`` of a
:class:`~context_synthetic_recognition.config.models.PluginSpec` against it when the pipeline
resolves the plug-in, and the GUI builds its parameter form from its fields.

Third-party packages add plug-ins through entry points, e.g. in their ``pyproject.toml``::

    [project.entry-points."context_synthetic_recognition.metrics"]
    my-metric = "my_package.metrics:my_metric"
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

from pydantic import ConfigDict, TypeAdapter, ValidationError

from context_synthetic_recognition.errors import ConfigError, RegistryError

T = TypeVar("T")

ENTRY_POINT_PREFIX = "context_synthetic_recognition."

PARAMS_CONFIG = ConfigDict(extra="forbid")
"""pydantic configuration of plug-in parameter types: unknown keys are errors.

Use it as ``@pydantic.dataclasses.dataclass(frozen=True, config=PARAMS_CONFIG)``.
"""


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

    def make_params(self, name: str, params: Mapping[str, object] | None = None) -> Any:
        """Validate ``params`` against the parameter type of plug-in ``name``.

        Returns:
            An instance of the plug-in's parameter type, or ``None`` if it takes no parameters.

        Raises:
            RegistryError: Unknown plug-in name.
            ConfigError: Unknown parameter, wrong type or value out of range.
        """
        info = self.info(name)
        given = dict(params or {})
        if info.params_type is None:
            if given:
                raise ConfigError(
                    f"{self.kind} '{info.name}' takes no parameters (got: {', '.join(given)})."
                )
            return None
        try:
            return TypeAdapter(info.params_type).validate_python(given)
        except ValidationError as error:
            lines = [f"{self.kind} '{info.name}': invalid parameters"]
            for item in error.errors():
                location = ".".join(str(part) for part in item["loc"]) or "(parameters)"
                lines.append(f"  {location}: {item['msg']}")
            raise ConfigError("\n".join(lines)) from error

    def names(self) -> list[str]:
        """Canonical names in registration order."""
        return list(self._plugins)

    @property
    def entry_point_group(self) -> str:
        """``context_synthetic_recognition.<kind>`` (e.g. ``….metrics``, ``….k-strategies``)."""
        return ENTRY_POINT_PREFIX + normalize_name(self.kind)

    def load_entry_points(self) -> int:
        """Register the plug-ins of the entry-point group :attr:`entry_point_group`.

        A module may register its plug-in itself on import (with :meth:`register`, which also
        records the parameter type); otherwise the loaded object is added under the entry
        point's name.

        Returns:
            Number of plug-ins added (already registered names are skipped).
        """
        added = 0
        for entry_point in entry_points(group=self.entry_point_group):
            if normalize_name(entry_point.name) in self:
                continue
            obj = entry_point.load()
            if normalize_name(entry_point.name) not in self:
                self.add(entry_point.name, obj)
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
