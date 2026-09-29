"""csmodel — reference implementation of the context-synthetic (CS) model experiment.

Pipeline:  load → unify scales → Zhuravlev distances (per operator) → sorted neighbours
           → synthetic features Ψ(r) (formula (5)) → membership (1), stability (2), boundary (3),
             informativeness (4), contributions (6) → HAG (Steps 1–5) → meta-dataset → meta-algorithm.
"""
