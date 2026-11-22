"""Local contexts Ψ_ρ,k(S, E) → synthetic features Ψ(r) with informativeness and contributions.

Steps 1–8 of the workbook, fitted on a training sample E:

1. normalization fitted on E (:mod:`..local_context.normalizers`);
2. distances of every base operator (:mod:`..local_context.operators`,
   :mod:`..local_context.metrics`);
3. neighbour order by (distance, index), self excluded (:mod:`..local_context.neighbours`);
4. admissible k from the training class sizes (:mod:`..local_context.admissible_k`) and
   same-class counts μ;
5. synthetic features aᵤ by formula (5) from χ₁ (:mod:`.encoders`);
6. membership (1), stability (2) and the majority chains (:mod:`.membership`);
7. boundary (3) and informativeness ω (4);
8. contributions η by formula (6) with the informativeness (:mod:`.contributions`).

:meth:`PsiOperator.represent` describes further objects relative to the fixed E with the same
operators, scaling and tie rule, using only the classes of their neighbours: it has no label
argument, so the class of a new object cannot enter Ψ (Theorem of the article, ADR-009).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.config.models import ExperimentConfig, PluginSpec
from context_synthetic_recognition.core.arrays import FloatArray, IntArray, as_float_matrix
from context_synthetic_recognition.core.local_context.admissible_k import admissible_k
from context_synthetic_recognition.core.local_context.neighbours import neighbour_order
from context_synthetic_recognition.core.local_context.normalizers import NORMALIZERS, Scaling
from context_synthetic_recognition.core.local_context.operators import (
    BaseOperator,
    resolve_operators,
)
from context_synthetic_recognition.core.synthetic_features.bit_representations import (
    MAX_CHAIN_BITS,
    class_sequence_masks,
    majority_chains,
)
from context_synthetic_recognition.core.synthetic_features.contributions import (
    INFORMATIVENESS,
    InformativenessInputs,
    contribution_values,
    contributions,
    gradation_counts,
)
from context_synthetic_recognition.core.synthetic_features.encoders import (
    ENCODERS,
    Encoder,
    k1_counts,
    own_class_counts,
)
from context_synthetic_recognition.core.synthetic_features.membership import (
    boundary,
    correct_side,
    indistinguishable,
    informativeness,
    membership_table,
    stability,
)
from context_synthetic_recognition.core.trace import (
    ClassSequenceMasks,
    LocalContext,
    MajorityChains,
    PsiRepresentation,
    PsiTrace,
    SkippedFeature,
    SyntheticFeature,
)
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import ConfigError, DatasetError, ModelUndefinedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class PsiOperator:
    """Steps 1–8 fitted on a training sample E; describes new objects without their class."""

    trace: PsiTrace
    """Every intermediate table of the fit."""
    scaling: Scaling
    """The local contexts' normalization fitted on E (the default of every operator)."""
    operators: tuple[BaseOperator, ...]
    """The base operators, in configuration order."""
    reference: FloatArray
    """(m × n) values of the training objects under :attr:`scaling`."""
    scalings: tuple[Scaling, ...]
    """The normalization of every operator (its own or :attr:`scaling`), fitted on E."""
    references: tuple[FloatArray, ...]
    """(m × n) values of the training objects under each operator's normalization."""
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
        """The admissible k."""
        return self.trace.admissible_k.ks

    def represent(
        self, X: npt.ArrayLike, *, exclude: npt.ArrayLike | None = None
    ) -> PsiRepresentation:
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
        matrix = as_float_matrix(values[None, :] if values.ndim == 1 else values)
        Z = self.scaling.transform(matrix)
        distances: list[FloatArray] = []
        orders: list[IntArray] = []
        chi_by_operator: list[IntArray] = []
        for operator, scaling, reference in zip(
            self.operators, self.scalings, self.references, strict=True
        ):
            Zo = Z if scaling is self.scaling else scaling.transform(matrix)
            D = operator.distances(Zo, reference, decimals=self.decimals)
            order = neighbour_order(D, exclude)
            distances.append(D)
            orders.append(order)
            chi_by_operator.append(k1_counts(order, self.reference_classes, self.ks))
        position = {k: c for c, k in enumerate(self.ks)}
        q = matrix.shape[0]
        chi1 = np.column_stack(
            [
                np.zeros(q, dtype=np.int64)
                if f.is_original
                else chi_by_operator[f.operator][:, position[f.k]]
                for f in self.trace.features
            ]
        )
        a = np.column_stack(
            [
                _codes(matrix[:, f.original])
                if f.is_original
                else _encode(self.encoder, chi1[:, u], f.k, self.encoder_params)
                for u, f in enumerate(self.trace.features)
            ]
        )
        contribution = np.column_stack(
            [
                contribution_values(a[:, u], f.eta, f.gradations)
                for u, f in enumerate(self.trace.features)
            ]
        )
        return PsiRepresentation(Z, tuple(distances), tuple(orders), chi1, a, contribution)


def _encode(encoder: Encoder, chi1: IntArray, k: int, params: Any) -> IntArray:
    values = np.asarray(encoder(chi1, k, params), dtype=np.int64)
    if not np.isin(values, (1, 2)).all():
        raise ValueError("a synthetic-feature encoder must return the gradations 1 or 2")
    return values


def _codes(column: FloatArray) -> IntArray:
    """The codes of a nominal feature as integers (a new object's codes as they are)."""
    return np.asarray(np.rint(column), dtype=np.int64)


def _original_indices(dataset: Dataset, chosen: str | tuple[str, ...]) -> list[int]:
    """The original nominal features combined with the synthetic ones (ADR-065).

    Raises:
        ConfigError: An unknown or a quantitative feature, or codes that are not non-negative
            integers.
    """
    names = dataset.feature_names
    if chosen == "nominal":
        indices = [j for j in range(len(names)) if not dataset.quantitative[j]]
    else:
        unknown = [name for name in chosen if name not in names]
        if unknown:
            raise ConfigError(
                f"synthetic_features.original_features: unknown features {unknown} "
                f"(features: {', '.join(names)})"
            )
        indices = [names.index(name) for name in chosen]
    quantitative = [names[j] for j in indices if dataset.quantitative[j]]
    if quantitative:
        raise ConfigError(
            "synthetic_features.original_features: only nominal features can be combined with "
            f"the synthetic ones; {', '.join(quantitative)} are quantitative (ADR-065)"
        )
    for j in indices:
        column = dataset.X[:, j]
        if np.any(column < 0) or not np.array_equal(column, np.rint(column)):
            raise ConfigError(
                f"synthetic_features.original_features: the codes of {names[j]} must be "
                "non-negative integers"
            )
    return indices


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


def fit_psi(dataset: Dataset, config: ExperimentConfig | None = None) -> PsiOperator:
    """Fit Steps 1–8 on the training sample ``dataset``.

    Args:
        dataset: The training sample E (two classes).
        config: Experiment configuration; the default configuration if ``None``.

    Raises:
        ModelUndefinedError: A stage is undefined for the data (e.g. no admissible k,
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

    # Step 1 — normalization on the training objects (once per distinct normalizer)
    fitted: dict[str, tuple[Scaling, FloatArray]] = {}

    def normalized(spec: PluginSpec) -> tuple[Scaling, FloatArray]:
        key = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
        if key not in fitted:
            made = NORMALIZERS.get(spec.name)(
                dataset.X, quantitative, NORMALIZERS.make_params(spec.name, spec.params)
            )
            fitted[key] = (made, made.transform(dataset.X))
        return fitted[key]

    scaling, Z = normalized(cfg.local_context.normalizer)

    # Steps 2–3 — distances and neighbour order per base operator
    decimals = cfg.local_context.distance_decimals
    resolved, skipped = resolve_operators(cfg.local_context, dataset.feature_names, quantitative)
    own = [normalized(o.normalizer or cfg.local_context.normalizer) for o in resolved]
    operators = tuple(o.fit(Zo) for o, (_, Zo) in zip(resolved, own, strict=True))
    self_index = np.arange(m, dtype=np.int64)
    contexts = []
    for operator, (operator_scaling, Zo) in zip(operators, own, strict=True):
        D = operator.distances(Zo, Zo, decimals=decimals)
        contexts.append(
            LocalContext(
                operator.label,
                operator.metric_name,
                operator.features,
                D,
                neighbour_order(D, self_index),
                operator_scaling.normalizer,
            )
        )

    # Step 4 — admissible k from the training class sizes
    ks_result = admissible_k(cfg.admissible_k.name, cfg.admissible_k.params, sizes, m - 1)
    ks = ks_result.ks

    # Steps 4–8 — per operator and k: μ, χ₁, (5), (1)–(4), (6)
    encoder_spec, informativeness_spec = (
        cfg.synthetic_features.encoder,
        cfg.synthetic_features.informativeness,
    )
    encoder = ENCODERS.get(encoder_spec.name)
    encoder_params = ENCODERS.make_params(encoder_spec.name, encoder_spec.params)
    informativeness_fn = INFORMATIVENESS.get(informativeness_spec.name)
    informativeness_params = INFORMATIVENESS.make_params(
        informativeness_spec.name, informativeness_spec.params
    )
    features: list[SyntheticFeature] = []
    skipped_features: list[SkippedFeature] = []
    masks: list[MajorityChains] = []
    sequences: list[ClassSequenceMasks] = []
    for o, context in enumerate(contexts):
        mu = own_class_counts(context.order, y, y, ks)
        chi1 = k1_counts(context.order, y, ks)
        for c, k in enumerate(ks):
            values = _encode(encoder, chi1[:, c], k, encoder_params)
            table = membership_table(mu[:, c], y, sizes, beta=k)
            g = stability(table, m)
            bound = boundary(table)
            object_f = table.f_of(mu[:, c])
            correct = correct_side(object_f, y, bound.G)
            omega = informativeness(correct)
            value = float(
                informativeness_fn(InformativenessInputs(omega, g, bound.G), informativeness_params)
            )
            alpha = gradation_counts(values, y)
            eta = contributions(alpha, sizes, value)
            zeros = int(np.count_nonzero(indistinguishable(table, mu[:, c], sizes)))
            if cfg.synthetic_features.skip_constant and np.all(values == values[0]):
                skipped_features.append(
                    SkippedFeature(context.label, k, int(values[0]), omega, "constant", zeros)
                )
                continue
            if cfg.synthetic_features.zero_value and zeros:
                skipped_features.append(
                    SkippedFeature(context.label, k, int(values[0]), omega, "zero", zeros)
                )
                continue
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
                    informativeness=value,
                    alpha=alpha,
                    eta=eta,
                    contributions=contribution_values(values, eta),
                    zeros=zeros,
                )
            )
        mask_ks, sequence_masks = class_sequence_masks(context.order, y, ks)
        if mask_ks:
            tables = tuple(
                membership_table(sequence_masks[:, c], y, sizes, beta=(1 << k) - 1)
                for c, k in enumerate(mask_ks)
            )
            sequences.append(
                ClassSequenceMasks(
                    o,
                    context.label,
                    mask_ks,
                    sequence_masks,
                    tables,
                    np.array([stability(table, m) for table in tables]),
                )
            )
        if len(ks) <= MAX_CHAIN_BITS:
            bits, mask_values = majority_chains(mu, ks)
            mask_table = membership_table(mask_values, y, sizes, beta=(1 << len(ks)) - 1)
            masks.append(
                MajorityChains(
                    o, context.label, ks, bits, mask_values, mask_table, stability(mask_table, m)
                )
            )

    # original nominal features combined with the synthetic ones (ADR-065)
    for j in _original_indices(dataset, cfg.synthetic_features.original_features):
        codes = _codes(dataset.X[:, j])
        known = tuple(int(v) for v in np.unique(codes))
        table = membership_table(codes, y, sizes, beta=int(codes.max()))
        g = stability(table, m)
        bound = boundary(table)
        object_f = table.f_of(codes)
        correct = correct_side(object_f, y, bound.G)
        omega = informativeness(correct)
        value = float(
            informativeness_fn(InformativenessInputs(omega, g, bound.G), informativeness_params)
        )
        alpha = gradation_counts(codes, y, known)
        eta = contributions(alpha, sizes, value)
        features.append(
            SyntheticFeature(
                index=len(features),
                operator=-1,
                operator_label=dataset.feature_names[j],
                k=0,
                mu=codes,
                chi1=np.zeros(m, dtype=np.int64),
                values=codes,
                membership=table,
                stability=g,
                boundary=bound,
                object_membership=object_f,
                correct=correct,
                omega=omega,
                informativeness=value,
                alpha=alpha,
                eta=eta,
                contributions=contribution_values(codes, eta, known),
                zeros=int(np.count_nonzero(indistinguishable(table, codes, sizes))),
                gradations=known,
                original=j,
            )
        )

    if skipped_features:
        logger.warning(
            "synthetic features left out (constant: ADR-030; zero values: ADR-064)",
            extra={
                "dataset": dataset.name,
                "skipped": [f"{s.operator_label}·{s.k}" for s in skipped_features],
            },
        )
    if not features:
        raise ModelUndefinedError(
            f"{dataset.name}: every synthetic feature was left out on the training sample "
            "(synthetic_features.skip_constant, ADR-030; synthetic_features.zero_value, ADR-064)"
        )
    trace = PsiTrace(
        object_ids=dataset.object_ids,
        feature_names=dataset.feature_names,
        classes=dataset.classes,
        class_index=y,
        class_sizes=sizes,
        scaling=scaling,
        normalized=Z,
        operators=tuple(contexts),
        skipped_operators=skipped,
        admissible_k=ks_result,
        features=tuple(features),
        majority_chains=tuple(masks),
        class_sequences=tuple(sequences),
        skipped_features=tuple(skipped_features),
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
    return PsiOperator(
        trace=trace,
        scaling=scaling,
        operators=operators,
        reference=Z,
        scalings=tuple(s for s, _ in own),
        references=tuple(Zo for _, Zo in own),
        reference_classes=y,
        encoder=encoder,
        encoder_params=encoder_params,
        decimals=decimals,
    )
