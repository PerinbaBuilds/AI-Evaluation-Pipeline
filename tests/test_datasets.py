"""Dataset loading and validation, including malformed inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalpipe.datasets import DatasetItem, load_dataset, validate_dataset
from evalpipe.exceptions import DatasetError


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestJsonlLoading:
    def test_loads_valid_items(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "data.jsonl",
            '{"id": "a", "prompt": "Q1", "expected": "A1"}\n'
            '{"id": "b", "prompt": "Q2", "metadata": {"context": "ctx"}}\n',
        )
        items = load_dataset(path)
        assert len(items) == 2
        assert items[0] == DatasetItem(id="a", prompt="Q1", expected="A1")
        assert items[1].expected is None
        assert items[1].metadata == {"context": "ctx"}

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "data.jsonl",
            '{"id": "a", "prompt": "Q1"}\n\n\n{"id": "b", "prompt": "Q2"}\n',
        )
        assert len(load_dataset(path)) == 2

    def test_invalid_json_reports_line_number(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.jsonl", '{"id": "a", "prompt": "Q1"}\n{broken\n')
        with pytest.raises(DatasetError, match=r"data\.jsonl:2"):
            load_dataset(path)

    def test_non_object_line_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.jsonl", '["not", "an", "object"]\n')
        with pytest.raises(DatasetError, match="expected a JSON object"):
            load_dataset(path)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.jsonl", '{"id": "a"}\n')
        with pytest.raises(DatasetError, match="prompt"):
            load_dataset(path)

    def test_blank_prompt_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.jsonl", '{"id": "a", "prompt": "   "}\n')
        with pytest.raises(DatasetError, match="prompt"):
            load_dataset(path)

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "data.jsonl",
            '{"id": "a", "prompt": "Q1"}\n{"id": "a", "prompt": "Q2"}\n',
        )
        with pytest.raises(DatasetError, match="duplicate"):
            load_dataset(path)

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.jsonl", "")
        with pytest.raises(DatasetError, match="empty"):
            load_dataset(path)

    def test_unicode_content(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "data.jsonl",
            '{"id": "u", "prompt": "Qué é isso? 日本語", "expected": "réponse"}\n',
        )
        items = load_dataset(path)
        assert items[0].prompt == "Qué é isso? 日本語"


class TestCsvLoading:
    def test_loads_valid_rows(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "data.csv",
            "id,prompt,expected,topic\na,Q1,A1,math\nb,Q2,,science\n",
        )
        items = load_dataset(path)
        assert len(items) == 2
        assert items[0].expected == "A1"
        assert items[0].metadata == {"topic": "math"}
        assert items[1].expected is None

    def test_missing_columns_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.csv", "prompt,expected\nQ1,A1\n")
        with pytest.raises(DatasetError, match="missing required CSV column"):
            load_dataset(path)

    def test_blank_row_reports_line(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.csv", "id,prompt\na,Q1\nb,\n")
        with pytest.raises(DatasetError, match=r"data\.csv:3"):
            load_dataset(path)


class TestGeneralHandling:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nope.jsonl")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.txt", "hello")
        with pytest.raises(DatasetError, match="unsupported dataset format"):
            load_dataset(path)

    def test_validate_reports_instead_of_raising(self, tmp_path: Path) -> None:
        report = validate_dataset(tmp_path / "nope.jsonl")
        assert not report.ok
        assert "not found" in report.errors[0]

    def test_validate_warns_on_missing_expected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.jsonl", '{"id": "a", "prompt": "Q1"}\n')
        report = validate_dataset(path)
        assert report.ok
        assert report.item_count == 1
        assert "no 'expected'" in report.warnings[0]
