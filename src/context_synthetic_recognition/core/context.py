"""Local contexts Ψ_ρ,k(S, E) → synthetic features Ψ(r) with weights and contributions.

Steps 1–8 of the workbook, fitted on a training sample E:

1. scale unification fitted on E (:mod:`.normalizers`);
2. distances of every base operator (:mod:`.operators`, :mod:`.metrics`);
3. neighbour order by (distance, index), self excluded (:mod:`.neighbours`);
4. permitted k from the training class sizes (:mod:`.k_strategies`) and same-class counts μ;
5. synthetic features aᵤ by formula (5) from χ₁ (:mod:`.encoders`);
6. membership (1), stability (2) and the bit masks (:mod:`.membership`);
7. boundary (3) and informativeness ω (4);
8. contributions η by formula (6) and the weights (:mod:`.contributions`).

:meth:`ContextModel.represent` describes further objects relative to the fixed E with the same
operators, scaling and tie rule, using only the classes of their neighbours: it has no label
argument, so the class of a new object cannot enter Ψ (Theorem of the article, ADR-009).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.core.arrays import FloatArray, IntArray, as_float_matrix
from context_synthetic_recognition.core.contributions import (
    WEIGHTS,
    WeightInputs,
    contribution_values,
    contributions,
    gradation_counts,
)
from context_synthetic_recognition.core.encoders import (
    ENCODERS,
    Encoder,
    k1_counts,
    same_class_counts,
)
from context_synthetic_recognition.core.k_strategies import permitted_k
from context_synthetic_recognition.core.membership import (
    MAX_MASK_BITS,
    bit_masks,
    boundary,
    correct_side,
    informativeness,
    membership_table,
    stability,
)
from context_synthetic_recognition.core.neighbours import neighbour_order
from context_synthetic_recognition.core.normalizers import NORMALIZERS, Scaling
from context_synthetic_recognition.core.operators import BaseOperator, resolve_operators
from context_synthetic_recognition.core.trace import (
    BitMaskRepresentation,
    ContextTrace,
    ObjectContext,
    OperatorContext,
    SyntheticFeature,
)
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import DatasetError, ModelUndefinedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class ContextModel:
    """Steps 1–8 fitted on a training sample E; describes new objects without their class."""

    trace: ContextTrace
    """Every intermediate table of the fit."""
    scaling: Scaling
    """Scale unification fitted on E."""
    operators: tuple[BaseOperator, ...]
    """The base operators, in configuration order."""
    reference: FloatArray
    """(m × n) unified values of the training objects."""
    reference_classes: IntArray
    """(m,) class index of the training objects — the neighbours' classes."""
    encoder: Encoder
    """Synthetic-feature encoder (formula (5) by default)."""
    encoder_params: Any
    """Its validated parameters."""
    decimals: int
    """Rounding of the distances."""

    @property
    def ks(self) -> tuple[int, ...]:
        """The permitted k."""
        return self.trace.permitted_k.ks

    def represent(self, X: npt.ArrayLike, *, exclude: npt.ArrayLike | None = None) -> ObjectContext:
        """Describe objects relative to E — Ψ_ρ,k(S, E) and Ψ(r) without the class of S (Theorem).

        Args:
            X: (q × n) original feature values of the objects (or one object as a vector).
            exclude: One training index per object to leave out of its context — used when a
                training object is passed through this path (it is then not its own neighbour
                and gets exactly its training row). ``None`` for new objects.

        Returns:
            Distances, neighbour orders, χ₁, the synthetic features aᵤ and their contribution
            values for every object.
        """
        values = np.asarray(X, dtype=np.float64)
        Z = self.scaling.transform(as_float_matrix(values[None, :] if values.ndim == 1 else values))
        distances: list[FloatArray] = []
        orders: list[IntArray] = []
        chi_by_operator: list[IntArray] = []
        for operator in self.operators:
            D = operator.distances(Z, self.reference, decimals=self.decimals)
            order = neighbour_order(D, exclude)
            distances.append(D)
            orders.append(order)
            chi_by_operator.append(k1_counts(order, self.reference_classes, self.ks))
        position = {k: c for c, k in enumerate(self.ks)}
        chi1 = np.column_stack(
            [chi_by_operator[f.operator][:, position[f.k]] for f in self.trace.features]
        )
        a = np.column_stack(
            [
                _encode(self.encoder, chi1[:, u], f.k, self.encoder_params)
                for u, f in enumerate(self.trace.features)
            ]
        )
        contribution = np.column_stack(
            [contribution_values(a[:, u], f.eta) for u, f in enumerate(self.trace.features)]
        )
        return ObjectContext(Z, tuple(distances), tuple(orders), chi1, a, contribution)


def _encode(encoder: Encoder, chi1: IntArray, k: int, params: Any) -> IntArray:
    values = np.asarray(encoder(chi1, k, params), dtype=np.int64)
    if not np.isin(values, (1, 2)).all():
        raise ValueError("a synthetic-feature encoder must return the gradations 1 or 2")
    return values


def _binary_classes(dataset: Dataset) -> tuple[int, int]:
    sizes = dataset.class_sizes
    if len(sizes) == 1:
        raise ModelUndefinedError(
            f"{dataset.name}: only one class ({dataset.classes[0]}); the model needs K1 and K2"
        )
    if len(sizes) > 2:
        raise DatasetError(
            f"{dataset.name}: {len(sizes)} classes — the CS-model of the article is binary; "
            "a K-class formulation is not specified yet (ADR-010)"
        )
    return sizes[0], sizes[1]


def fit_context(dataset: Dataset, config: ExperimentConfig | None = None) -> ContextModel:
    """Fit Steps 1–8 on the training sample ``dataset``.

    Args:
        dataset: The training sample E (two classes).
        config: Experiment configuration; the template preset by default.

    Raises:
        ModelUndefinedError: A stage is undefined for the data (e.g. no permitted k,
            Definition 1).
        DatasetError: More than two classes (ADR-010).
        ConfigError: Unknown plug-in or invalid plug-in parameters.
    """
    cfg = config or ExperimentConfig()
    started = time.perf_counter()
    sizes = _binary_classes(dataset)
    m = dataset.m
    y = dataset.class_index
    quantitative = dataset.quantitative

    # Step 1 — scale unification on the training objects
    spec = cfg.preprocessing.normalizer
    scaling = NORMALIZERS.get(spec.name)(
        dataset.X, quantitative, NORMALIZERS.make_params(spec.name, spec.params)
    )
    Z = scaling.transform(dataset.X)

    # Steps 2–3 — distances and neighbour order per base operator
    decimals = cfg.context.distance_decimals
    operators, skipped = resolve_operators(cfg.context, dataset.feature_names, quantitative)
    self_index = np.arange(m, dtype=np.int64)
    contexts = []
    for operator in operators:
        D = operator.distances(Z, Z, decimals=decimals)
        contexts.append(
            OperatorContext(
                operator.label,
                operator.metric_name,
                operator.features,
                D,
                neighbour_order(D, self_index),
            )
        )

    # Step 4 — permitted k from the training class sizes
    ks_result = permitted_k(cfg.k.name, cfg.k.params, sizes, m - 1)
    ks = ks_result.ks

    # Steps 4–8 — per operator and k: μ, χ₁, (5), (1)–(4), (6)
    encoder_spec, weight_spec = cfg.synthetic.encoder, cfg.synthetic.weights
    encoder = ENCODERS.get(encoder_spec.name)
    encoder_params = ENCODERS.make_params(encoder_spec.name, encoder_spec.params)
    weight_fn = WEIGHTS.get(weight_spec.name)
    weight_params = WEIGHTS.make_params(weight_spec.name, weight_spec.params)
    features: list[SyntheticFeature] = []
    masks: list[BitMaskRepresentation] = []
    for o, context in enumerate(contexts):
        mu = same_class_counts(context.order, y, y, ks)
        chi1 = k1_counts(context.order, y, ks)
        for c, k in enumerate(ks):
            values = _encode(encoder, chi1[:, c], k, encoder_params)
            table = membership_table(mu[:, c], y, sizes, beta=k)
            g = stability(table, m)
            bound = boundary(table)
            object_f = table.f_of(mu[:, c])
            correct = correct_side(object_f, y, bound.G)
            omega = informativeness(correct)
            weight = float(weight_fn(WeightInputs(omega, g, bound.G), weight_params))
            alpha = gradation_counts(values, y)
            eta = contributions(alpha, sizes, weight)
            features.append(
                SyntheticFeature(
                    index=len(features),
                    operator=o,
                    operator_label=context.label,
                    k=k,
                    mu=mu[:, c].copy(),
                    chi1=chi1[:, c].copy(),
                    values=values,
                    membership=table,
                    stability=g,
                    boundary=bound,
                    object_membership=object_f,
                    correct=correct,
                    omega=omega,
                    weight=weight,
                    alpha=alpha,
                    eta=eta,
                    contributions=contribution_values(values, eta),
                )
            )
        if len(ks) <= MAX_MASK_BITS:
            bits, mask_values = bit_masks(mu, ks)
            mask_table = membership_table(mask_values, y, sizes, beta=(1 << len(ks)) - 1)
            masks.append(
                BitMaskRepresentation(
                    o, context.label, ks, bits, mask_values, mask_table, stability(mask_table, m)
                )
            )

    trace = ContextTrace(
        object_ids=dataset.object_ids,
        feature_names=dataset.feature_names,
        classes=dataset.classes,
        class_index=y,
        class_sizes=sizes,
        scaling=scaling,
        normalized=Z,
        operators=tuple(contexts),
        skipped_operators=skipped,
        permitted_k=ks_result,
        features=tuple(features),
        bit_masks=tuple(masks),
    )
    logger.info(
        "context fitted (Steps 1–8)",
        extra={
            "dataset": dataset.name,
            "m": m,
            "k": ks_result.label,
            "r": trace.r,
            "operators": [c.label for c in contexts],
            "seconds": round(time.perf_counter() - started, 3),
        },
    )
    return ContextModel(
        trace=trace,
        scaling=scaling,
        operators=operators,
        reference=Z,
        reference_classes=y,
        encoder=encoder,
        encoder_params=encoder_params,
        decimals=decimals,
    )
