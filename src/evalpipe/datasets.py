"""Dataset loading and validation.

Datasets are flat files of evaluation items. Two formats are supported:

- **JSONL**: one JSON object per line with ``id``, ``prompt`` and optionally
  ``expected`` and ``metadata`` keys.
- **CSV**: a header row containing at least ``id`` and ``prompt``; an
  ``expected`` column is optional and any extra columns are collected into
  ``metadata``.

Loading is strict by design: malformed rows fail fast with the offending line
number rather than being silently dropped, because a silently truncated
dataset invalidates every downstream metric.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evalpipe.exceptions import DatasetError

_SUPPORTED_SUFFIXES = {".jsonl", ".csv"}
_RESERVED_CSV_COLUMNS = {"id", "prompt", "expected"}


class DatasetItem(BaseModel):
    """A single evaluation item."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_dataset` — errors are fatal, warnings are not."""

    item_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_dataset(path: str | Path) -> list[DatasetItem]:
    """Load and validate a dataset file, raising :class:`DatasetError` on any problem."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset file not found: {path}")
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise DatasetError(
            f"unsupported dataset format {path.suffix!r} for {path}; expected .jsonl or .csv"
        )

    items = _load_jsonl(path) if path.suffix.lower() == ".jsonl" else _load_csv(path)

    if not items:
        raise DatasetError(f"dataset is empty: {path}")
    _check_duplicate_ids(items, path)
    return items


def validate_dataset(path: str | Path) -> ValidationReport:
    """Validate a dataset without raising; returns a human-readable report."""
    report = ValidationReport()
    try:
        items = load_dataset(path)
    except DatasetError as exc:
        report.errors.append(str(exc))
        return report

    report.item_count = len(items)
    missing_expected = [item.id for item in items if item.expected is None]
    if missing_expected:
        report.warnings.append(
            f"{len(missing_expected)} item(s) have no 'expected' value; "
            "reference-based evaluators will score them 0 "
            f"(first few: {', '.join(missing_expected[:5])})"
        )
    return report


def _load_jsonl(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise DatasetError(
                    f"{path}:{line_no}: expected a JSON object, got {type(payload).__name__}"
                )
            items.append(_build_item(payload, f"{path}:{line_no}"))
    return items


def _load_csv(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = {"id", "prompt"} - set(reader.fieldnames)
        if missing:
            raise DatasetError(
                f"{path}: missing required CSV column(s): {', '.join(sorted(missing))}"
            )
        for line_no, row in enumerate(reader, start=2):
            payload: dict[str, object] = {
                "id": row.get("id") or "",
                "prompt": row.get("prompt") or "",
            }
            expected = row.get("expected")
            if expected:
                payload["expected"] = expected
            metadata = {
                key: value
                for key, value in row.items()
                if key not in _RESERVED_CSV_COLUMNS and value is not None and value != ""
            }
            if metadata:
                payload["metadata"] = metadata
            items.append(_build_item(payload, f"{path}:{line_no}"))
    return items


def _build_item(payload: dict[str, object], location: str) -> DatasetItem:
    try:
        return DatasetItem.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first["loc"]) or "item"
        raise DatasetError(f"{location}: invalid item ({loc}: {first['msg']})") from exc


def _check_duplicate_ids(items: list[DatasetItem], path: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item.id in seen:
            duplicates.append(item.id)
        seen.add(item.id)
    if duplicates:
        raise DatasetError(
            f"{path}: duplicate item id(s): {', '.join(sorted(set(duplicates))[:10])}"
        )
