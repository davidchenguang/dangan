"""测试 core/exporter.py — 多格式导出"""

import json
import os
import tempfile

import pytest

from core.exporter import CsvRenderer, ExcelRenderer, JsonRenderer, ResultExporter
from core.models import ChangeRecord, HouseholdCard


@pytest.fixture
def sample_card() -> HouseholdCard:
    return HouseholdCard(
        name="张三",
        gender="男",
        native_place="浙江绍兴",
        ethnicity="汉",
        source_image="test.jpg",
        migration_in=[
            ChangeRecord(date="1953年", content="由绍兴迁来"),
        ],
    )


@pytest.fixture
def sample_cards(sample_card: HouseholdCard) -> list[HouseholdCard]:
    card2 = HouseholdCard(
        name="李四",
        gender="女",
        native_place="江苏南京",
        source_image="test2.jpg",
    )
    return [sample_card, card2]


@pytest.fixture
def tmp_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestCsvRenderer:
    def test_single_card(self, sample_card: HouseholdCard, tmp_dir: str):
        filepath = os.path.join(tmp_dir, "test.csv")
        CsvRenderer().render([sample_card], filepath)

        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8-sig") as f:
            content = f.read()
        assert "张三" in content
        assert "男" in content
        assert "浙江绍兴" in content

    def test_batch_cards(self, sample_cards: list[HouseholdCard], tmp_dir: str):
        filepath = os.path.join(tmp_dir, "batch.csv")
        CsvRenderer().render(sample_cards, filepath)

        with open(filepath, "r", encoding="utf-8-sig") as f:
            content = f.read()
        assert "张三" in content
        assert "李四" in content


class TestJsonRenderer:
    def test_single_card(self, sample_card: HouseholdCard, tmp_dir: str):
        filepath = os.path.join(tmp_dir, "test.json")
        JsonRenderer().render([sample_card], filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["name"] == "张三"
        assert data[0]["gender"] == "男"

    def test_batch_cards(self, sample_cards: list[HouseholdCard], tmp_dir: str):
        filepath = os.path.join(tmp_dir, "batch.json")
        JsonRenderer().render(sample_cards, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2


class TestExcelRenderer:
    def test_single_card(self, sample_card: HouseholdCard, tmp_dir: str):
        filepath = os.path.join(tmp_dir, "test.xlsx")
        ExcelRenderer().render([sample_card], filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0


class TestResultExporter:
    def test_export_csv(self, sample_card: HouseholdCard, tmp_dir: str):
        filepath = os.path.join(tmp_dir, "test.csv")
        exporter = ResultExporter()
        exporter.export([sample_card], filepath)
        assert os.path.exists(filepath)

    def test_export_json(self, sample_card: HouseholdCard, tmp_dir: str):
        filepath = os.path.join(tmp_dir, "test.json")
        exporter = ResultExporter()
        exporter.export([sample_card], filepath)
        assert os.path.exists(filepath)

    def test_export_unsupported_format(self, sample_card: HouseholdCard, tmp_dir: str):
        filepath = os.path.join(tmp_dir, "test.xyz")
        exporter = ResultExporter()
        with pytest.raises(ValueError, match="不支持的导出格式"):
            exporter.export([sample_card], filepath)

    def test_export_multi(self, sample_cards: list[HouseholdCard], tmp_dir: str):
        paths = [
            os.path.join(tmp_dir, "test.csv"),
            os.path.join(tmp_dir, "test.json"),
        ]
        exporter = ResultExporter()
        exported = exporter.export_multi(sample_cards, paths)
        assert len(exported) == 2
        for p in exported:
            assert os.path.exists(p)

    def test_supported_formats(self):
        exporter = ResultExporter()
        formats = exporter.supported_formats
        assert ".csv" in formats
        assert ".xlsx" in formats
        assert ".json" in formats
        assert ".pdf" in formats
        assert ".docx" in formats
