"""Pure numerical core of the CS-model: no I/O, no GUI, deterministic.

Every public function names the article formula or algorithm step it implements. The packages
follow the stages of the method:

- :mod:`.rips` — R-IPS, the choice of the training and test samples;
- :mod:`.local_context` — normalizers, metrics, base operators, neighbours, admissible k;
- :mod:`.synthetic_features` — formula (5), membership (1), stability (2), informativeness (4),
  contributions (6), bit representations and the operator Ψ (:mod:`.synthetic_features.psi`);
- :mod:`.additional_features` — majorizing functions and the hierarchical agglomerative algorithm;
- :mod:`.meta_algorithm` — the meta-algorithm R and its decision rule.

:mod:`.model` puts the stages together, :mod:`.trace` holds the intermediate tables,
:mod:`.properties` checks Definitions 1, 4 and 6 and the Theorem, and :mod:`.registry` provides
the plug-in mechanism.
"""
