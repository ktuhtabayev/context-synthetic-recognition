"""An independent, dependency-free implementation of the CS-model experiment (a test oracle).

Pipeline:  load → unify scales → Zhuravlyov distances (per operator) → sorted neighbours
           → synthetic features Ψ(r) (formula (5)) → membership (1), stability (2), boundary (3),
             informativeness (4), contributions (6) → hierarchical agglomerative algorithm (Steps 1–5) → meta-dataset → meta-algorithm.
"""
