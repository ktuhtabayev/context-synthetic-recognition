r"""The comparison with standard classifiers on the article's training and test samples.

Run from the repository root::

    .\.venv\Scripts\python.exe scripts\run_comparison.py

It reads the split files of ``datasets/splits/`` and writes ``experiments/comparison/`` (or the
folder given as the first argument): the Standard classifiers and k-NN by metric tables as CSV,
Markdown and LaTeX, the results of every split and method, ``summary.json`` and
``checksums.sha256`` — the same as ``csr compare run``. The settings are fixed in
:mod:`context_synthetic_recognition.comparison` (ADR-079); nothing is tuned and nothing is chosen
on the test samples.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from context_synthetic_recognition.services.comparison import (
    OUTPUT_DIR,
    run_comparison,
    splits_folder,
    write_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.joinpath(*OUTPUT_DIR)


def run(out: Path = OUT) -> list[Path]:
    """Run the comparison on the project's split files and write it to ``out``."""
    return write_comparison(run_comparison(splits_folder(ROOT)), out)


def main() -> int:
    """Write the results to ``experiments/comparison`` (or to the folder given)."""
    started = time.perf_counter()
    files = run(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    sys.stdout.write(f"comparison written in {time.perf_counter() - started:.1f} s\n")
    for path in files:
        sys.stdout.write(f"  {path.as_posix()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
