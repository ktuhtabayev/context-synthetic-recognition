"""Context-synthetic model of recognition algorithms (CS-model).

Implements the pipeline of the article "Контекстно-синтетическая модель алгоритмов распознавания
на основе локальных метрических отношений"::

    (S, E) → Ψ_ρ,k(S, E) → D(S, E) → Y(S, E) → R(Y(S, E))

Local k-nearest-neighbour contexts of the base operators (metric × feature subset) become
synthetic features Ψ(r), the hierarchical agglomerative grouping (HAG) turns them into
additional (latent) features D, and the meta-algorithm R classifies in the space Y. A new object
is represented and classified without its class (Theorem of the article).

Layers: :mod:`.core` (pure numerical core), :mod:`.config`, :mod:`.services` (runs, persistence),
:mod:`.cli`; the data layer, evaluation, exporters and the GUI follow in later milestones.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("context-synthetic-recognition")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

# A library never configures logging on import; applications call log.configure_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__"]
