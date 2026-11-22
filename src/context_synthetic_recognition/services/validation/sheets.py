"""The sheet names of the author's workbooks and of the Excel mirror.

The validation reads the author's workbooks as they are, so their sheet names are quoted here —
and only here — exactly as the files spell them. The Excel mirror names some sheets in the
article's terms; :data:`MIRROR_NAMES` maps a workbook sheet to the mirror's name, so that one map
validates both the original workbook and an exported mirror.
"""

from __future__ import annotations

# ---------------------------------------------------------------- the Excel experiment workbook

OVERVIEW = "Overview"
PARAMETERS = "Parameters"
SWITCHES = "Template Deviations"
"""The workbook's sheet on the two switches of the agglomerative algorithm."""
DATASET = "Dataset"
QUANTITATIVE = "Quantitative"
NOMINAL = "Nominal"
NORMALIZED = "Normalized Dataset"
DISTANCES = "Zhuravlev Distances"
SORTED_NEIGHBOURS = "Sorted Neighbors ({label})"
"""Formatted with the operator label (ρ, ρ_I, ρ_J)."""
OWN_CLASS_COUNTS = "Synthetic Features (k-NN)"
BINARY_FEATURES = "Ψ(r) Binary Features"
MEMBERSHIP = "Membership & Stability"
INFORMATIVENESS = "Informativeness ω"
CONTRIBUTIONS = "Ψ(r) Contribution & Weight"
AGGLOMERATIVE_STEP = "Greedy upon Weight ({j}-Latent)"
"""Formatted with the iteration j: the iteration that produces the additional feature dⱼ."""
META_DESCRIPTION = "Dataset for Meta-algorithm"
NEW_OBJECT = "Brace for Meta-algorithm"
META_ALGORITHM = "Meta-algorithm"
META_ALL = "Meta-algorithm (All Objects)"
MARGINS = "Margin Analysis"
ACCURACY = "Accuracy"
CONFUSION = "Confusion Matrix"
PRECISION = "Precision, Recall, F1 Score"
ROC = "ROC Curve & AUC"
LEAVE_ONE_OUT = "Leave-One-Out"
SENSITIVITY = "Sensitivity (Switches)"
PROPERTIES = "Model Properties"
VALIDATION = "Validation"

# ---------------------------------------------------------------- the two check workbooks

CONTRIBUTION_INPUT = "Dataset (Contribution & Weight)"
"""The input sheet of the additional-features check workbook."""


def agglomerative_step(j: int) -> str:
    """The workbook's sheet of iteration ``j`` (1-based)."""
    return AGGLOMERATIVE_STEP.format(j=j)


def sorted_neighbours(label: str) -> str:
    """The workbook's sheet of the sorted neighbours of an operator."""
    return SORTED_NEIGHBOURS.format(label=label)


# ---------------------------------------------------------------- the Excel mirror

MIRROR_STEP = "Additional Feature {j}"
"""The mirror's sheet of iteration j (it produces the additional feature dⱼ)."""


def mirror_step(j: int) -> str:
    """The mirror's sheet of iteration ``j`` (1-based)."""
    return MIRROR_STEP.format(j=j)


MAX_STEPS = 64
"""Iterations for which the step sheets are mapped."""

MIRROR_NAMES: dict[str, str] = {
    DISTANCES: "Zhuravlyov Distances",
    SWITCHES: "Switches",
    CONTRIBUTIONS: "Ψ(r) Contributions",
    META_DESCRIPTION: "Meta-description",
    NEW_OBJECT: "New Object",
    SENSITIVITY: "Switch Sensitivity",
    **{agglomerative_step(j): mirror_step(j) for j in range(1, MAX_STEPS + 1)},
}
"""Workbook sheet → the mirror's name, where they differ."""


def mirror_name(name: str) -> str:
    """The mirror's name of a workbook sheet."""
    return MIRROR_NAMES.get(name, name)
