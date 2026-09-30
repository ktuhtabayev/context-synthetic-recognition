# Development

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,docs]"
.\.venv\Scripts\pre-commit.exe install
```

The virtual environment stores absolute paths; if the project folder moves, create it again.

## Everyday commands

```powershell
.\.venv\Scripts\python.exe -m pytest --cov              # tests + coverage (≥ 90 %)
.\.venv\Scripts\python.exe -m pytest -m golden          # acceptance tests against the workbook
.\.venv\Scripts\python.exe -m pytest -m "not slow"      # skip the long tests
.\.venv\Scripts\ruff.exe check .                         # lint
.\.venv\Scripts\ruff.exe format .                        # format
.\.venv\Scripts\mypy.exe                                 # strict type check of src/
.\.venv\Scripts\pre-commit.exe run --all-files           # all hooks
.\.venv\Scripts\mkdocs.exe serve                         # documentation at http://127.0.0.1:8000
```

## Conventions

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`, `chore:`, `ci:`), with a body explaining *why*.
- **Branches:** one per milestone, `feat/m<n>-<topic>`, merged into `main` with a merge commit once
  CI is green; each milestone is tagged `v0.<n>.0` and recorded in `CHANGELOG.md`.
- **Decisions:** every decision goes into [DECISIONS.md](DECISIONS.md). Anything that changes
  numerical results is agreed with the author first.
- **Protected material:** `resources/experiments/` and `docs/handoff/` are never modified;
  linters and hooks exclude them and Git stores them byte for byte.
- **Template workbooks:** `tests/data/templates/` holds byte-for-byte copies of the HAG and
  meta-algorithm templates of `../hag-regularized-stacking-boosting-meta` (read-only), so CI can
  replicate them; a test compares the copies with the originals when that project is present
  (ADR-029).
- **Docstrings:** Google style; every public function names the article formula or step it
  implements.

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`):

- ruff (lint and format check) and mypy on Linux;
- pytest with coverage on Windows and Linux, Python 3.11, 3.12 and 3.13;
- `mkdocs build --strict`.
