# Context-synthetic recognition

Python implementation of the **context-synthetic model of recognition algorithms (CS-model)** from
the article *«Контекстно-синтетическая модель алгоритмов распознавания на основе локальных
метрических отношений»* (Ensembles of algorithms and Zhuravlev's algebra – 6): a library, the `csr`
command line and a desktop GUI.

Local k-nearest-neighbour contexts of several base operators (metric × feature subset) become
synthetic features; the hierarchical agglomerative grouping turns them into additional (latent)
features; the meta-algorithm classifies in that space. A new object is represented and classified
without its class.

!!! warning "Two template calculations differ from the article"
    - θ and γ are measured from running partial class means instead of the final class means M₁
      and M₂.
    - In STEP 4 the majorizer is applied twice instead of once.

    Both are configuration switches; the default reproduces the template workbooks, the `article`
    preset follows the article. See [DECISIONS.md](DECISIONS.md) (ADR-002 – ADR-004).

## Contents

- [Architecture](architecture.md) — layers, pipeline, trace objects, extension points
- [Algorithm reference](algorithm-reference.md) — article element → workbook sheet → module
- [Decisions](DECISIONS.md) — the ADR log
- [Development](development.md) — setup, commands, conventions, CI
