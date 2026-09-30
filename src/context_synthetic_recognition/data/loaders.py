"""Dataset loaders: the experiment workbook, the template CSV/DAT layout, CSV, XLSX and Parquet.

Formats (registry :data:`LOADERS`, names as in ``dataset.format`` of the configuration):

``cs-workbook``
    The *Dataset* sheet of the author's Excel experiment: a header row with ``Class`` above the
    class column, one row per object (features in the columns left of ``Class``), then the
    feature-type row (1 = quantitative, 0 = nominal).
``template-extended``
    The layout of the template project (``.csv`` comma-separated or ``.dat`` whitespace-separated,
    decimal commas accepted): a first row ``m, n, c``; m rows of n features and the class label;
    a last row of n type flags.
``csv``, ``xlsx``, ``parquet``
    A table with a header row: one column per feature, a class column (``class_column``, the last
    column by default) and optionally an id column. Feature types come from ``feature_types``;
    features not listed there are inferred (numeric → quantitative, otherwise nominal) with a
    warning. Nominal text values are encoded as category codes.

Missing values are rejected with a clear error (ADR-015).
"""

from __future__ import annotations

import csv
import importlib
import logging
import math
import re
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.data.schema import Dataset, FeatureType, Label, as_labels
from context_synthetic_recognition.errors import DatasetError

if TYPE_CHECKING:
    from context_synthetic_recognition.config.models import DatasetConfig

logger = logging.getLogger(__name__)

WORKBOOK_SHEET = "Dataset"
"""Default sheet of the ``cs-workbook`` format."""
_MAX_REPORTED = 10


@dataclass(frozen=True)
class LoadOptions:
    """Reading options shared by all loaders (mirrors ``dataset`` of the configuration)."""

    sheet: str | None = None
    """Excel sheet; ``Dataset`` for the workbook format, the first sheet for ``xlsx``."""
    class_column: str | None = None
    """Class column of a table (the last column by default)."""
    id_column: str | None = None
    """Optional column of object ids."""
    feature_types: Mapping[str, str] | None = None
    """Feature name → ``quantitative`` / ``nominal``; overrides the types stored in the file."""
    name: str | None = None
    """Dataset name; the file stem (or the workbook sheet title) by default."""


class Loader(Protocol):
    """Reads one file format into a :class:`Dataset`."""

    def __call__(self, path: Path, options: LoadOptions, /) -> Dataset:
        """Load ``path``."""
        ...


LOADERS: Registry[Loader] = Registry("loaders")
"""Dataset formats by name."""


# ---------------------------------------------------------------- public entry points


def detect_format(path: Path, sheet: str | None = None) -> str:
    """Format name for ``path`` from its extension and, for CSV and Excel files, its content.

    Raises:
        DatasetError: The extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "cs-workbook" if _looks_like_cs_workbook(path, sheet or WORKBOOK_SHEET) else "xlsx"
    if suffix in {".csv", ".txt"}:
        first = next(_csv_rows(path), None)
        return "template-extended" if first is not None and _meta_header(first) else "csv"
    if suffix == ".dat":
        return "template-extended"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise DatasetError(
        f"{path.name}: unknown dataset extension '{suffix}' "
        "(supported: .xlsx, .xlsm, .csv, .txt, .dat, .parquet)"
    )


def load_dataset(
    path: Path | str,
    format: str = "auto",
    *,
    sheet: str | None = None,
    class_column: str | None = None,
    id_column: str | None = None,
    feature_types: Mapping[str, str] | None = None,
    name: str | None = None,
) -> Dataset:
    """Load a dataset file.

    Args:
        path: The file.
        format: A name from :data:`LOADERS` or ``auto`` (detected by :func:`detect_format`).
        sheet: Excel sheet.
        class_column: Class column of a table.
        id_column: Optional object-id column of a table.
        feature_types: Feature name → type; overrides or completes the types in the file.
        name: Dataset name.

    Raises:
        DatasetError: The file is missing, unreadable or does not match its format.
    """
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"dataset file not found: {file}")
    chosen = detect_format(file, sheet) if format == "auto" else format
    options = LoadOptions(sheet, class_column, id_column, feature_types, name)
    dataset = LOADERS.get(chosen)(file, options)
    logger.info(
        "dataset loaded",
        extra={"dataset": dataset.name, "format": chosen, "m": dataset.m, "n": dataset.n},
    )
    return dataset


def load_from_config(config: DatasetConfig, base_dir: Path | None = None) -> Dataset:
    """Load the dataset described by the ``dataset`` section of a configuration.

    Relative paths are resolved against ``base_dir`` (the configuration file's folder, ADR-016).
    """
    if config.path is None:
        raise DatasetError("the configuration names no dataset (dataset.path is empty)")
    path = Path(config.path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return load_dataset(
        path,
        config.format,
        sheet=config.sheet,
        class_column=config.class_column,
        id_column=config.id_column,
        feature_types=config.feature_types,
    )


# ---------------------------------------------------------------- value parsing

_DECIMAL_COMMA = re.compile(r"[+-]?\d*,\d+([eE][+-]?\d+)?")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _number(value: object) -> float | None:
    """The numeric value of a cell, or ``None`` if it is not a number (decimal commas accepted)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            if _DECIMAL_COMMA.fullmatch(text):
                return float(text.replace(",", "."))
    return None


def _label(value: object) -> Label:
    """Class label of a cell: integral numbers become ``int``, other text stays text."""
    number = _number(value)
    if number is not None and math.isfinite(number) and number.is_integer():
        return int(number)
    return str(value).strip()


def _flag(value: object, where: str) -> FeatureType:
    number = _number(value)
    if number is None or number not in (0.0, 1.0):
        raise DatasetError(f"{where}: feature-type flags must be 1 or 0, got {value!r}")
    return FeatureType.from_flag(int(number))


def _override_types(dataset: Dataset, types: Mapping[str, str] | None) -> Dataset:
    return dataset.with_feature_types(types) if types else dataset


# ---------------------------------------------------------------- the experiment workbook


def _open_workbook(path: Path) -> Any:
    openpyxl = importlib.import_module("openpyxl")
    try:
        with warnings.catch_warnings():
            # openpyxl warns about Excel extensions it does not read (data validation, …)
            warnings.simplefilter("ignore", UserWarning)
            return openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as error:  # openpyxl raises many types for damaged or foreign files
        raise DatasetError(f"{path.name}: cannot read the Excel file ({error})") from error


def _sheet_rows(path: Path, sheet: str | None) -> tuple[str, list[tuple[Any, ...]]]:
    """Title and cell values of a sheet (the first sheet if ``sheet`` is ``None``)."""
    workbook = _open_workbook(path)
    try:
        names = list(workbook.sheetnames)
        if sheet is not None and sheet not in names:
            raise DatasetError(f"{path.name}: no sheet '{sheet}' (sheets: {', '.join(names)})")
        worksheet = workbook[sheet] if sheet is not None else workbook[names[0]]
        return worksheet.title, [tuple(row) for row in worksheet.iter_rows(values_only=True)]
    finally:
        workbook.close()


def _class_header(rows: Sequence[tuple[Any, ...]]) -> tuple[int, int] | None:
    """(row, column) of the ``Class`` header cell within the first rows of a sheet."""
    for r, row in enumerate(rows[:10]):
        for c, value in enumerate(row):
            if isinstance(value, str) and value.strip().lower() == "class":
                return r, c
    return None


def _looks_like_cs_workbook(path: Path, sheet: str) -> bool:
    try:
        _, rows = _sheet_rows(path, sheet)
    except DatasetError:
        return False
    header = _class_header(rows)
    if header is None or header[0] + 1 >= len(rows):
        return False
    first = rows[header[0] + 1]
    return len(first) > header[1] and _number(first[header[1]]) is not None


def load_cs_workbook(path: Path, options: LoadOptions) -> Dataset:
    """The *Dataset* sheet of the experiment workbook (format ``cs-workbook``).

    Layout: a header row with ``Class`` above the class column; features in the columns between
    the first column (object ids, optional) and the class column; one row per object; then the
    feature-type row (1 = quantitative, 0 = nominal). The Office-Math symbols drawn over the cells
    (x₁, S₁, …) are not cell values; default names are used instead.
    """
    sheet = options.sheet or WORKBOOK_SHEET
    title, rows = _sheet_rows(path, sheet)
    header = _class_header(rows)
    if header is None:
        raise DatasetError(f"{path.name} [{title}]: no 'Class' header cell in the first rows")
    header_row, class_col = header
    feature_cols = range(1, class_col)
    if not feature_cols:
        raise DatasetError(f"{path.name} [{title}]: no feature columns left of 'Class'")

    def cell(row: tuple[Any, ...], col: int) -> Any:
        return row[col] if col < len(row) else None

    objects: list[tuple[Any, ...]] = []
    r = header_row + 1
    while r < len(rows) and _number(cell(rows[r], class_col)) is not None:
        objects.append(rows[r])
        r += 1
    if not objects:
        raise DatasetError(f"{path.name} [{title}]: no object rows below the header")
    if r >= len(rows):
        raise DatasetError(
            f"{path.name} [{title}]: the feature-type row after the objects is missing"
        )
    type_row = rows[r]

    X = np.empty((len(objects), len(feature_cols)))
    missing: list[str] = []
    for i, row in enumerate(objects):
        for j, col in enumerate(feature_cols):
            value = _number(cell(row, col))
            if value is None:
                missing.append(f"row {header_row + 2 + i}, column {col + 1}")
                value = math.nan
            X[i, j] = value
    if missing:
        shown = "; ".join(missing[:_MAX_REPORTED])
        raise DatasetError(f"{path.name} [{title}]: missing or non-numeric values at {shown}")
    type_where = f"{path.name} [{title}] row {r + 1}"
    types = tuple(_flag(cell(type_row, col), type_where) for col in feature_cols)
    header_cells = [cell(rows[header_row], col) for col in feature_cols]
    names = tuple(v.strip() if isinstance(v, str) else "" for v in header_cells)
    ids = tuple("" if _is_missing(cell(row, 0)) else str(cell(row, 0)).strip() for row in objects)
    top = rows[0][0] if rows and rows[0] else None
    dataset = Dataset(
        X=X,
        y=as_labels([_label(cell(row, class_col)) for row in objects]),
        feature_types=types,
        feature_names=names if all(names) else (),
        object_ids=ids if all(ids) else (),
        name=options.name or (top.strip() if isinstance(top, str) and top.strip() else path.stem),
        source=str(path),
    )
    return _override_types(dataset, options.feature_types)


# ---------------------------------------------------------------- the template layout


def _csv_rows(path: Path, delimiter: str = ",") -> Iterator[list[str]]:
    """Non-empty rows of a text table, trailing empty cells removed."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle, delimiter=delimiter):
                while row and row[-1].strip() == "":
                    row.pop()
                if row:
                    yield row
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DatasetError(f"{path.name}: cannot read the file ({error})") from error


def _dat_rows(path: Path) -> Iterator[list[str]]:
    """Non-empty rows of a whitespace-separated file."""
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                tokens = line.split()
                if tokens:
                    yield tokens
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"{path.name}: cannot read the file ({error})") from error


def _meta_header(row: Sequence[str]) -> tuple[int, int, int] | None:
    """(m, n, c) if ``row`` is the template's first row, else ``None``."""
    cells = [cell.strip() for cell in row if cell.strip()]
    if len(cells) != 3:
        return None
    numbers = [_number(cell) for cell in cells]
    if any(v is None or not v.is_integer() or v <= 0 for v in numbers):
        return None
    m, n, c = (int(v) for v in numbers if v is not None)
    return m, n, c


def load_template_extended(path: Path, options: LoadOptions) -> Dataset:
    """The template project's layout (format ``template-extended``).

    Row 1: ``m, n, c``; rows 2 … m + 1: n feature values and the class label; last row: n type
    flags (1 = quantitative, 0 = nominal), optionally followed by a placeholder under the class
    column. ``.dat`` files are whitespace-separated and may use decimal commas; other files are
    comma-separated.
    """
    rows = list(_dat_rows(path) if path.suffix.lower() == ".dat" else _csv_rows(path))
    if not rows:
        raise DatasetError(f"{path.name}: the file is empty")
    header = _meta_header(rows[0])
    if header is None:
        raise DatasetError(f"{path.name}: the first row must be 'm, n, c', got {rows[0]}")
    m, n, c = header
    if len(rows) != m + 2:
        raise DatasetError(
            f"{path.name}: the header says m = {m}, so {m + 2} rows are expected "
            f"(header, {m} objects, type flags); the file has {len(rows)} non-empty rows"
        )
    X = np.empty((m, n))
    labels: list[Label] = []
    for i, row in enumerate(rows[1 : m + 1]):
        cells = [cell.strip() for cell in row]
        if len(cells) != n + 1:
            raise DatasetError(
                f"{path.name}: object row {i + 1} has {len(cells)} values, expected {n + 1}"
            )
        for j, cell in enumerate(cells[:n]):
            value = _number(cell)
            if value is None or not math.isfinite(value):
                raise DatasetError(
                    f"{path.name}: object row {i + 1}, feature {j + 1}: {cell!r} is not a number"
                )
            X[i, j] = value
        labels.append(_label(cells[n]))
    flags = [cell for cell in rows[-1] if cell.strip()]
    if len(flags) == n + 1:
        flags = flags[:n]  # some template files put a placeholder under the class column
    if len(flags) != n:
        raise DatasetError(f"{path.name}: the type row has {len(flags)} flags, expected n = {n}")
    types = tuple(_flag(flag, f"{path.name} type row") for flag in flags)
    dataset = Dataset(
        X=X,
        y=as_labels(labels),
        feature_types=types,
        name=options.name or path.stem,
        source=str(path),
    )
    if len(dataset.classes) != c:
        raise DatasetError(
            f"{path.name}: the header says c = {c} classes, the data have {len(dataset.classes)}"
        )
    return _override_types(dataset, options.feature_types)


# ---------------------------------------------------------------- tables with a header row


def table_to_dataset(
    header: Sequence[object],
    rows: Iterable[Sequence[object]],
    options: LoadOptions,
    *,
    name: str,
    source: str | None = None,
) -> Dataset:
    """Build a dataset from a header row and value rows (formats ``csv``, ``xlsx``, ``parquet``).

    Raises:
        DatasetError: Unknown class/id column, missing values, non-numeric quantitative values.
    """
    columns = [str(h).strip() if h is not None else "" for h in header]
    if any(not c for c in columns) or len(set(columns)) != len(columns):
        raise DatasetError(f"{name}: column names must be non-empty and unique, got {columns}")
    class_column = options.class_column or columns[-1]
    for role, column in (("class", class_column), ("id", options.id_column)):
        if column is not None and column not in columns:
            raise DatasetError(f"{name}: no {role} column '{column}' (columns: {columns})")
    data = [list(row) + [None] * (len(columns) - len(row)) for row in rows]
    if not data:
        raise DatasetError(f"{name}: the table has no rows")
    missing = [
        f"row {i + 1} '{column}'"
        for i, row in enumerate(data)
        for column, value in zip(columns, row, strict=False)
        if _is_missing(value)
    ]
    if missing:
        raise DatasetError(f"{name}: missing values at {'; '.join(missing[:_MAX_REPORTED])}")

    feature_columns = [c for c in columns if c not in (class_column, options.id_column)]
    if not feature_columns:
        raise DatasetError(f"{name}: no feature columns besides the class and id columns")
    given = dict(options.feature_types or {})
    unknown = sorted(set(given) - set(feature_columns))
    if unknown:
        raise DatasetError(f"{name}: feature types given for unknown columns {unknown}")

    X = np.empty((len(data), len(feature_columns)))
    types: list[FeatureType] = []
    categories: dict[str, tuple[str, ...]] = {}
    inferred: dict[str, FeatureType] = {}
    for j, column in enumerate(feature_columns):
        values = [row[columns.index(column)] for row in data]
        numbers = [_number(v) for v in values]
        numeric = all(v is not None and math.isfinite(v) for v in numbers)
        if column in given:
            ftype = FeatureType(given[column])
        else:
            ftype = FeatureType.QUANTITATIVE if numeric else FeatureType.NOMINAL
            inferred[column] = ftype
        if ftype is FeatureType.QUANTITATIVE and not numeric:
            bad = next(v for v, x in zip(values, numbers, strict=True) if x is None)
            raise DatasetError(f"{name}: quantitative feature '{column}' has the value {bad!r}")
        if numeric:
            X[:, j] = [v for v in numbers if v is not None]
        else:
            texts = [str(v).strip() for v in values]
            levels = tuple(sorted(set(texts)))
            categories[column] = levels
            X[:, j] = [levels.index(t) for t in texts]
        types.append(ftype)
    if inferred:
        logger.warning(
            "feature types inferred (set dataset.feature_types to fix them)",
            extra={
                "dataset": name,
                "quantitative": [c for c, t in inferred.items() if t is FeatureType.QUANTITATIVE],
                "nominal": [c for c, t in inferred.items() if t is FeatureType.NOMINAL],
            },
        )
    ids = (
        tuple(str(row[columns.index(options.id_column)]).strip() for row in data)
        if options.id_column
        else ()
    )
    return Dataset(
        X=X,
        y=as_labels([_label(row[columns.index(class_column)]) for row in data]),
        feature_types=tuple(types),
        feature_names=tuple(feature_columns),
        object_ids=ids,
        name=name,
        categories=categories,
        source=source,
    )


def load_csv(path: Path, options: LoadOptions) -> Dataset:
    """A CSV table with a header row (format ``csv``); the delimiter is detected (, ; tab |)."""
    try:
        sample = path.read_text(encoding="utf-8-sig")[:65536]
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"{path.name}: cannot read the file ({error})") from error
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(_csv_rows(path, delimiter))
    if not rows:
        raise DatasetError(f"{path.name}: the file is empty")
    return table_to_dataset(
        rows[0], rows[1:], options, name=options.name or path.stem, source=str(path)
    )


def load_xlsx(path: Path, options: LoadOptions) -> Dataset:
    """An Excel table with a header row (format ``xlsx``); the first sheet by default."""
    title, rows = _sheet_rows(path, options.sheet)
    filled = [row for row in rows if any(not _is_missing(v) for v in row)]
    if not filled:
        raise DatasetError(f"{path.name} [{title}]: the sheet is empty")
    header = list(filled[0])
    while header and _is_missing(header[-1]):
        header.pop()
    body = [row[: len(header)] for row in filled[1:]]
    return table_to_dataset(header, body, options, name=options.name or path.stem, source=str(path))


def load_parquet(path: Path, options: LoadOptions) -> Dataset:
    """A Parquet table (format ``parquet``); needs the optional dependency ``pyarrow``."""
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise DatasetError(
            "reading Parquet files needs pyarrow: "
            'pip install "context-synthetic-recognition[parquet]"'
        ) from error
    try:
        columns: dict[str, list[Any]] = parquet.read_table(path).to_pydict()
    except Exception as error:  # pyarrow raises its own error types
        raise DatasetError(f"{path.name}: cannot read the Parquet file ({error})") from error
    header = list(columns)
    rows = list(zip(*columns.values(), strict=True))
    return table_to_dataset(header, rows, options, name=options.name or path.stem, source=str(path))


LOADERS.add("cs-workbook", load_cs_workbook, summary="Dataset sheet of the Excel experiment")
LOADERS.add(
    "template-extended",
    load_template_extended,
    summary="template layout: m, n, c / objects / type flags (.csv, .dat)",
)
LOADERS.add("csv", load_csv, summary="CSV table with a header row")
LOADERS.add("xlsx", load_xlsx, summary="Excel table with a header row")
LOADERS.add("parquet", load_parquet, summary="Parquet table (needs pyarrow)")
