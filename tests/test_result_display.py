"""诊断测试 — 验证结果显示链路（不涉及真实 OCR/GPU）

测试: 模拟 OCR worker 返回结果后，检查 GUI 信号链路是否正确触发 display_cards
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

# 跳过无 GUI 环境
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from core.models import HouseholdCard
from core.pipeline import PipelineResult


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication(sys.argv)
    yield a


class TestOcrResultSignalChain:
    """验证 OCR 结果回来后 GUI 正确显示"""

    def test_pipeline_result_cards_accessible(self):
        """PipelineResult.cards 在 success=True 时应可访问"""
        cards = [HouseholdCard(name="王盛祥", gender="男")]
        result = PipelineResult(cards=cards, success=True)
        assert result.success is True
        assert len(result.cards) == 1
        assert result.cards[0].name == "王盛祥"

    def test_pipeline_result_card_compat(self):
        """向后兼容 .card 属性 返回第一张"""
        cards = [HouseholdCard(name="王盛祥")]
        result = PipelineResult(cards=cards)
        assert result.card.name == "王盛祥"

    def test_pipeline_result_empty_cards_no_crash(self):
        """空 cards 时 .card 返回空 HouseholdCard 不崩溃"""
        result = PipelineResult()
        assert result.card.name == ""

    def test_result_table_display_cards_no_crash(self, app):
        """display_cards 收到单列卡片不崩溃"""
        from gui.result_table import ResultTable
        table = ResultTable()
        cards = [HouseholdCard(name="王盛祥", gender="男", native_place="辽宁")]
        table.display_cards(cards)  # 不应抛出异常
        assert table._table.columnCount() == 2  # 1字段列 + 1值列
        # 检查姓名行
        name_row = next(
            i for i, (attr, _) in enumerate(table.DISPLAY_FIELDS)
            if attr == "name"
        )
        val = table._table.item(name_row, 1)
        assert val is not None
        assert val.text() == "王盛祥"

    def test_result_table_display_two_cards(self, app):
        """两列卡片时表格有3列"""
        from gui.result_table import ResultTable
        table = ResultTable()
        cards = [
            HouseholdCard(name="王盛祥", gender="男", relation_to_household_head="户主"),
            HouseholdCard(name="马妇女", gender="女", relation_to_household_head="妻"),
        ]
        table.display_cards(cards)
        assert table._table.columnCount() == 3  # 字段列 + 户主 + 妻
        name_row = next(
            i for i, (attr, _) in enumerate(table.DISPLAY_FIELDS)
            if attr == "name"
        )
        assert table._table.item(name_row, 1).text() == "王盛祥"
        assert table._table.item(name_row, 2).text() == "马妇女"

    def test_result_table_get_cards_roundtrip(self, app):
        """display_cards 后 get_cards 返回相同数据"""
        from gui.result_table import ResultTable
        table = ResultTable()
        original = [HouseholdCard(name="王盛祥", gender="男")]
        table.display_cards(original)
        result_cards = table.get_cards()
        assert len(result_cards) == 1
        assert result_cards[0].name == "王盛祥"
        assert result_cards[0].gender == "男"

    def test_result_table_title_updates(self, app):
        """有姓名时标题包含姓名"""
        from gui.result_table import ResultTable
        table = ResultTable()
        table.display_cards([HouseholdCard(name="王盛祥")])
        assert "王盛祥" in table._title.text()

    def test_result_table_no_data_shows_raw(self, app):
        """字段都空但有 raw_markdown 时，显示原始文本区域"""
        from gui.result_table import ResultTable
        table = ResultTable()
        card = HouseholdCard(raw_markdown="原始识别文本blahblah")
        table.display_cards([card])
        # isVisible() 要求控件及所有祖先可见，测试环境无父窗口故用 isHidden()
        assert not table._raw_text_edit.isHidden()
        assert "原始识别文本" in table._raw_text_edit.toPlainText()
