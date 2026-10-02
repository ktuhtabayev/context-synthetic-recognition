r"""Update the translation files of the desktop application.

Run from the repository root after interface texts were added or changed::

    .\.venv\Scripts\python.exe scripts\update_translations.py

The script collects every text the interface translates — the literal arguments of ``tr(...)``
and ``mark(...)`` in the ``gui`` package — and brings ``gui/translations/csr_<code>.ts`` up to
date for every language: new texts are added without a translation (and listed), texts that are
no longer used are removed, existing translations are kept. It then compiles each ``.ts`` into
the ``.qm`` the application loads (``pyside6-lrelease``).

Translate by editing the ``<translation>`` elements of the ``.ts`` files — in a text editor or in
Qt Linguist (``.\.venv\Scripts\pyside6-linguist.exe``) — and run the script again. ``--check``
changes nothing and fails if a text is missing, untranslated or stale.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src" / "context_synthetic_recognition" / "gui"
TRANSLATIONS = GUI / "translations"
CONTEXT = "csr"
LOCALES = {"ru": "ru_RU", "uz": "uz_UZ"}
"""Language code → the locale written into the ``.ts`` file."""
CALLS = ("tr", "mark")


def source_texts(package: Path = GUI) -> dict[str, list[str]]:
    """Every translatable text of the interface → the places that use it (``file:line``)."""
    texts: dict[str, list[str]] = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in CALLS
                and node.args
            ):
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                where = f"{path.relative_to(package).as_posix()}:{node.lineno}"
                texts.setdefault(argument.value, []).append(where)
    return texts


def unmarked_calls(package: Path = GUI) -> list[str]:
    """``tr(...)`` calls whose argument is not a literal: their texts must be marked elsewhere."""
    found = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "tr"
                and node.args
                and not isinstance(node.args[0], ast.Constant)
            ):
                where = f"{path.relative_to(package).as_posix()}:{node.lineno}"
                found.append(f"{where}: tr({ast.unparse(node.args[0])})")
    return found


def read_ts(path: Path) -> dict[str, dict[str, str]]:
    """The translations of a ``.ts`` file: context → source text → translation."""
    contexts: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return contexts
    for context in ET.parse(path).getroot().iter("context"):
        messages = contexts.setdefault(context.findtext("name") or "", {})
        for message in context.iter("message"):
            messages[message.findtext("source") or ""] = message.findtext("translation") or ""
    return contexts


def write_ts(path: Path, locale: str, contexts: dict[str, dict[str, str]]) -> None:
    """Write a ``.ts`` file (UTF-8, LF): one context after the other, texts in the given order."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<!DOCTYPE TS>",
        f'<TS version="2.1" language="{locale}" sourcelanguage="en">',
    ]
    for name, messages in contexts.items():
        lines += ["<context>", f"    <name>{escape(name)}</name>"]
        for source, translation in messages.items():
            lines.append("    <message>")
            lines.append(f"        <source>{escape(source)}</source>")
            if translation:
                lines.append(f"        <translation>{escape(translation)}</translation>")
            else:
                lines.append('        <translation type="unfinished"></translation>')
            lines.append("    </message>")
        lines.append("</context>")
    lines.append("</TS>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def merged(
    existing: dict[str, dict[str, str]], texts: dict[str, list[str]]
) -> dict[str, dict[str, str]]:
    """The interface's texts in source order with the existing translations; other contexts kept."""
    known = existing.get(CONTEXT, {})
    result = {CONTEXT: {text: known.get(text, "") for text in texts}}
    result.update({name: dict(messages) for name, messages in existing.items() if name != CONTEXT})
    return result


def problems(code: str, texts: dict[str, list[str]]) -> list[str]:
    """What is wrong with a language's ``.ts`` file (empty if it is complete and current)."""
    known = read_ts(TRANSLATIONS / f"csr_{code}.ts").get(CONTEXT, {})
    found = [f"{code}: not in the file: {text!r}" for text in texts if text not in known]
    found += [f"{code}: no longer used: {text!r}" for text in known if text not in texts]
    found += [
        f"{code}: not translated: {text!r} ({texts[text][0]})"
        for text, translation in known.items()
        if text in texts and not translation
    ]
    return found


def lrelease() -> Path:
    """The ``pyside6-lrelease`` program next to the running Python."""
    folder = Path(sys.executable).parent
    for name in ("pyside6-lrelease.exe", "pyside6-lrelease"):
        if (folder / name).is_file():
            return folder / name
    raise SystemExit(
        "pyside6-lrelease was not found next to python (pip install PySide6-Essentials)"
    )


def compile_qm(code: str) -> Path:
    """Compile ``csr_<code>.ts`` into ``csr_<code>.qm``."""
    source = TRANSLATIONS / f"csr_{code}.ts"
    target = source.with_suffix(".qm")
    subprocess.run([str(lrelease()), "-silent", str(source), "-qm", str(target)], check=True)
    return target


def say(text: str) -> None:
    """Print a line; the texts hold characters a Windows code page cannot encode."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
    print(text)  # noqa: T201 - a script


def main(argv: list[str] | None = None) -> int:
    """Update (or with ``--check`` only verify) the translation files."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify only; change nothing")
    arguments = parser.parse_args(argv)
    texts = source_texts()
    if arguments.check:
        found = [line for code in LOCALES for line in problems(code, texts)]
        say("\n".join(found) or f"{len(texts)} texts, {len(LOCALES)} languages: complete")
        return 1 if found else 0
    for code, locale in LOCALES.items():
        path = TRANSLATIONS / f"csr_{code}.ts"
        write_ts(path, locale, merged(read_ts(path), texts))
        target = compile_qm(code)
        missing = [line for line in problems(code, texts) if "not translated" in line]
        say(f"{path.name}: {len(texts)} texts, {len(missing)} not translated → {target.name}")
        for line in missing:
            say("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
