# Architecture

## Layers

The computational core is pure and deterministic; everything that touches files, users or screens
sits above it. The CLI and the GUI call the same services.

```mermaid
flowchart TB
    subgraph interfaces["Interfaces"]
        api["Python API"]
        cli["CLI · csr (typer)"]
        gui["GUI · PySide6"]
    end
    subgraph services["Application services"]
        runner["experiment runner"]
        persist["runs · manifests"]
        validate["workbook validation"]
        export["exporters"]
    end
    subgraph core["Computational core (numpy, no I/O)"]
        direction LR
        pipe["normalizers → operators → neighbours → k → encoders → (1)–(6) → HAG → meta"]
        trace["trace objects"]
    end
    config["config (pydantic)"]
    data["data layer · schema · loaders"]
    interfaces --> services --> core
    services --> data
    services --> config
    core -. "plug-ins" .- registry["registries"]
```

| Package | Role | Status |
|---|---|---|
| `core` | Numerical pipeline, plug-in registries, trace objects, model properties | registry ✓ (M1), Steps 1–8 + trace ✓ (M2), HAG, meta-algorithm, model ✓ (M3), properties ✓ (M4) |
| `config` | Typed configuration, presets, the two switches, YAML/TOML/JSON, hash | ✓ M1 |
| `data` | Dataset schema, loaders (workbook, template CSV/DAT, CSV/XLSX/Parquet), built-in datasets, hashing | ✓ M2 |
| `evaluation` | Protocols, metrics, AUC/ROC, margins, baselines | ✓ M4 |
| `services` | Runs and manifests, validation against the workbook and the templates, dataset summaries, experiment runner, switch sensitivity, configuration checks | ✓ M1 → M4 |
| `export` | Excel mirror, CSV/JSON, LaTeX/Markdown, figures, report | M5 |
| `cli` | `csr` command | `config` ✓ (M1), `data`, `validate` ✓ (M2), `fit`, `classify` ✓ (M3), `run` ✓ (M4) |
| `gui` | Desktop application | M6 |

## Pipeline and information flow

```mermaid
flowchart LR
    X["X, types I/J"] --> N["scale unification<br/>(fitted on E)"]
    N --> D["distances per operator<br/>ρ, ρ_I, ρ_J (rounded 1e-10)"]
    D --> O["neighbour order<br/>(distance, index)"]
    O --> MU["μ — same-class count<br/>(training side only)"]
    O --> CHI["χ₁ — K1 count<br/>(class-free)"]
    CHI --> A["Ψ(r): aᵤ ∈ {1,2}<br/>formula (5)"]
    MU --> F["f, g, G, ω<br/>(1)–(4)"]
    A --> ETA["η — (6)"]
    F --> ETA
    ETA --> HAG["HAG Steps 1–5<br/>TUPLAM, r₁…r_p"]
    HAG --> META["meta-algorithm<br/>B1/B2 → class or 0"]
    A --> META
```

The class of an object enters only through μ, which feeds the training-side formulas (1)–(4). A new
object reaches the meta-algorithm through χ₁ and formula (5) alone (Theorem, ADR-009): its class
is not an argument of `represent` or `predict`.

## Trace objects

Every intermediate table the workbook shows — normalized data, distance and rank matrices,
neighbour blocks with μ and χ₁, Ψ(r), f/g, ω, η, each HAG iteration with every candidate's b,
centres, θ, γ and θ/γ, the meta-algorithm's B1/B2 per step, margins and metrics — is captured as
immutable data. The GUI tables, the exporters and the golden tests read the same trace, so the
workbook, the software and the article's tables cannot drift apart.

## Extension points

| Kind | Default | Planned alternatives |
|---|---|---|
| metrics | Zhuravlyov on all / I / J | HEOM, HVDM, Gower (ADR-011), then Euclidean, Manhattan, Chebyshev, Minkowski, Canberra, cosine, Mahalanobis, Hamming, weighted Zhuravlyov |
| normalizers | min–max (training data) | none ✓; z-score, robust, max-abs, decimal scaling, rank, unit length (M7) |
| k strategies | formula (ADR-005/006) | article rule, explicit list, range ✓ (ADR-022) |
| encoders | formula (5) | — (μ and bit masks are training-side gradations, ADR-021) |
| weights | ω by (4) | Criterion-1, λ·β (template) |
| majorizers | logistic sigmoid | tanh, arctan, softsign ✓ (scaled to (0, 1), ADR-026), custom |
| decision rules | article Step 4 (refusal = 0) ✓ | — |
| protocols | resubstitution, leave-one-out | stratified k-fold, repeated k-fold, hold-out ✓ (M4); nested CV |
| baselines | k-NN vote | scikit-learn classifiers ✓ (M4, extra `[sklearn]`) |

Plug-ins register by name in a `Registry`; external packages add them through the entry-point
group `context_synthetic_recognition.<kind>`.
