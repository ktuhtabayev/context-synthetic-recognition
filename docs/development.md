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
.\.venv\Scripts\python.exe -m pytest -m gui             # the desktop application (no window appears)
.\.venv\Scripts\python.exe scripts\gui_screenshots.py   # the screenshots of docs/gui.md
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

## Translations

The interface is English in the code; Russian and Uzbek (Latin script) are the files
`src\context_synthetic_recognition\gui\translations\csr_ru.ts` and `csr_uz.ts` (ADR-054).

```powershell
.\.venv\Scripts\python.exe scripts\update_translations.py           # after texts were added or changed
.\.venv\Scripts\python.exe scripts\update_translations.py --check   # verify only
```

The script collects the literal texts of `tr(...)` and `mark(...)` in the `gui` package, adds new
ones to both `.ts` files without a translation (and lists them), removes texts that are no
longer used, and compiles the `.qm` files. Translate by editing the `<translation>` elements — in
a text editor or in Qt Linguist (`.\.venv\Scripts\pyside6-linguist.exe`) — then run the script
again and commit the `.ts` and `.qm` files together. A translation keeps the placeholders
(`{name}`), the `&` of a menu accelerator and the `(*.ext)` patterns of a file filter; the tests
check this, and that every text is translated and the `.qm` files are current. A text shown
through a variable — `tr(page.title)` — must be marked where it is defined: `title = mark("Run")`.

To add a language: add its code to `LANGUAGES` in `gui\i18n.py` and to `LOCALES` in the script,
run the script and translate the new `.ts` file.

## The Windows executable

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui,pdf,release]"
.\.venv\Scripts\python.exe scripts\build_exe.py
```

PyInstaller freezes the desktop application into `dist\csr-gui\` (a folder with `csr-gui.exe`,
to be copied as a whole) and packs it as `dist\csr-gui-<version>-windows.zip`. The script then
starts `csr-gui.exe --self-test` in an empty folder — every language, the default experiment, a
table, a figure and an export of the run — and fails if the frozen application does not pass;
the report is `dist\self-test.txt` (ADR-055).

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`), every job on Windows (ADR-049):

- ruff (lint and format check) and mypy;
- pytest with coverage on Python 3.11, 3.12 and 3.13 — including the GUI tests, which run on
  Qt's `offscreen` platform (ADR-048);
- `mkdocs build --strict`; on `main` the built site is published to GitHub Pages (ADR-056);
- the Windows executable: built, self-tested and kept as the workflow artifact
  `csr-gui-windows` of every run.

## Releasing

A milestone is released from its branch once CI is green: merge into `main` with a merge commit,
tag the merge commit `v0.<n>.0`, push both. The CI run of the tag builds the executable of
exactly that commit; a GitHub Release for the tag carries that `csr-gui-<version>-windows.zip`
and the changelog entry as its notes.
