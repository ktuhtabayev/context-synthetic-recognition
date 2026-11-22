# Context-synthetic recognition

Python implementation of the **context-synthetic model of recognition algorithms (CS-model)** of
the author's article (in preparation): a library, the `csr` command line and a desktop GUI. The
article is the specification and the defaults follow it (ADR-058); the author's Excel experiment
on Heart-Disease (10, 13, 2) remains an acceptance test.

R-IPS forms the precedent base: it splits the sample into a training and a test sample. Local
k-nearest-neighbour contexts of several base operators (metric × feature subset) become synthetic
features Ψ(r); the hierarchical agglomerative algorithm turns them into additional (latent)
features d₁ … d_p; the meta-algorithm classifies in that space or refuses. A new object is
represented and classified without its class.

!!! warning "Four switches deviate from the article"
    - `rips.keep_blocked_cover` — in R-IPS a blocked object keeps covering other objects with
      |r(S)| although blocking makes r(S) negative (ADR-061).
    - `rips.training: moved` — the objects R-IPS moves to D form the training sample instead of
      the objects that remain in E₁ (ADR-061).
    - `additional_features.class_means: running` — θ and γ are measured from running partial
      class means instead of the final class means M₁ and M₂ (ADR-002).
    - `additional_features.majorizing_passes: 2` — in Step 4 the majorizing function is applied
      twice instead of once (ADR-003).

    All four are off by default. The `excel-experiment` preset turns on the last two, because the
    Excel experiment computes that way; with it the package reproduces the workbook cell for cell.

## Contents

- [Architecture](architecture.md) — layers, the packages of the stages, trace objects, extension
  points
- [Algorithm reference](algorithm-reference.md) — article element → module → workbook sheet
- [Glossary](glossary.md) — the article's terms, symbols and their names in the code
- [Reproducing the Excel experiment](reproduction.md) — the workbooks checked cell for cell, every
  result step by step, repeating a saved run
- [Computational experiment](experiments.md) — the article's Tables 1–6 and the k-NN comparison
  rebuilt from real runs
- [Standard classifiers](comparison.md) — Tables 7 and 8: SVM, random forest and k-NN on the
  article's training and test samples, and the CS-model on the same samples; `csr compare`
- [Datasets](datasets.md) — formats, feature types, classes, the dataset folder, `csr data info`
- [Evaluation](evaluation.md) — protocols (R-IPS split, nested cross-validation, leave-one-out, …),
  metrics, baselines, run folders, large datasets
- [Exporting](exporting.md) — the Excel workbook, CSV/JSON, Markdown/LaTeX tables, figures,
  the HTML/PDF report
- [Desktop application](gui.md) — the workflow from dataset to export, with screenshots
- [Extending](extending.md) — plug-ins: metrics, normalizers, k strategies, encoders,
  informativeness, majorizing functions, decision rules, protocols
- [Decisions](DECISIONS.md) — the ADR log
- [Development](development.md) — setup, commands, conventions, CI
