"""测试 core/field_extractor.py — 字段提取器"""

import json
from typing import List
import pytest

from core.field_extractor import FieldExtractor
from core.models import HouseholdCard


@pytest.fixture
def extractor() -> FieldExtractor:
    return FieldExtractor()


class TestJsonExtraction:
    """Level 1: JSON 解析测试"""

    def test_direct_json(self, extractor: FieldExtractor):
        data = json.dumps({
            "姓名": "张三",
            "性别": "男",
            "籍贯": "浙江绍兴",
        })
        card = extractor.extract(data)
        assert card.name == "张三"
        assert card.gender == "男"
        assert card.native_place == "浙江绍兴"

    def test_json_in_code_block(self, extractor: FieldExtractor):
        data = '```json\n{"姓名": "李四", "性别": "女"}\n```'
        card = extractor.extract(data)
        assert card.name == "李四"
        assert card.gender == "女"

    def test_bare_json_in_text(self, extractor: FieldExtractor):
        data = '识别结果如下：\n{"姓名": "王五", "民族": "汉"}\n以上为识别结果。'
        card = extractor.extract(data)
        assert card.name == "王五"
        assert card.ethnicity == "汉"

    def test_nested_fields(self, extractor: FieldExtractor):
        data = json.dumps({
            "出生日期": "1930年3月",
            "出生地址": "浙江绍兴",
            "公民证代号号码": "001234",
        })
        card = extractor.extract(data)
        assert card.birth.date == "1930年3月"
        assert card.birth.address == "浙江绍兴"
        assert card.citizen_cert.code_number == "001234"


class TestMarkdownTableExtraction:
    """Level 2: Markdown 表格解析测试"""

    def test_basic_table(self, extractor: FieldExtractor):
        text = (
            "| 字段名 | 值 |\n"
            "|--------|-----|\n"
            "| 姓名 | 张三 |\n"
            "| 性别 | 男 |\n"
            "| 籍贯 | 浙江 |\n"
        )
        card = extractor.extract(text)
        assert card.name == "张三"
        assert card.gender == "男"
        assert card.native_place == "浙江"


class TestDescriptiveTextExtraction:
    """Level 3: 描述性文本提取测试"""

    def test_markdown_bold_fields(self, extractor: FieldExtractor):
        text = (
            "**姓名**：张三\n"
            "**性别**：男\n"
            "**籍贯**：浙江绍兴\n"
            "**民族**：汉\n"
        )
        card = extractor.extract(text)
        assert card.name == "张三"
        assert card.gender == "男"
        assert card.native_place == "浙江绍兴"
        assert card.ethnicity == "汉"

    def test_skip_empty_values(self, extractor: FieldExtractor):
        text = (
            "**姓名**：张三\n"
            "**别名**：无\n"
            "**籍贯**：\n"
        )
        card = extractor.extract(text)
        assert card.name == "张三"
        assert card.alias == ""  # "无" 应被跳过
        assert card.native_place == ""  # 空值应被跳过

    def test_skip_blacklist_keys(self, extractor: FieldExtractor):
        text = (
            "**标题**：户籍登记表\n"
            "**姓名**：张三\n"
            "**图片内容描述**：这是一张旧的户籍卡\n"
        )
        card = extractor.extract(text)
        assert card.name == "张三"

    def test_first_occurrence_wins(self, extractor: FieldExtractor):
        """重复字段应保留首次出现的值"""
        text = (
            "**姓名**：张三\n"
            "**姓名**：李四\n"
        )
        card = extractor.extract(text)
        assert card.name == "张三"


class TestFallback:
    """降级策略测试"""

    def test_empty_text(self, extractor: FieldExtractor):
        card = extractor.extract("")
        assert card.raw_markdown == ""

    def test_unparseable_text(self, extractor: FieldExtractor):
        text = "这是一段完全无法解析的随机文本，没有任何结构化信息。"
        card = extractor.extract(text)
        assert card.raw_markdown == text

    def test_raw_markdown_preserved(self, extractor: FieldExtractor):
        """所有成功的提取都应保留 raw_markdown"""
        data = json.dumps({"姓名": "张三"})
        card = extractor.extract(data)
        assert card.raw_markdown == data


class TestMultiColumnExtraction:
    """多列宽表提取测试 — extract_multi() 应返回多个 HouseholdCard"""

    def test_two_column_markdown_table(self, extractor: FieldExtractor):
        """标准宽表：| 字段 | 成员1值 | 成员2值 | 应解析出 2 个 HouseholdCard"""
        text = (
            "| 字段名 | 户主 | 妻 |\n"
            "|--------|------|-----|\n"
            "| 姓名 | 王盛祥 | 马妇女 |\n"
            "| 性别 | 男 | 女 |\n"
            "| 籍贯 | 辽宁省抚顺县 | 河北省任邱县 |\n"
            "| 民族 | 汉 | 汉 |\n"
        )
        cards = extractor.extract_multi(text)
        assert isinstance(cards, list)
        assert len(cards) == 2
        assert cards[0].name == "王盛祥"
        assert cards[0].gender == "男"
        assert cards[0].relation_to_household_head == "户主"
        assert cards[1].name == "马妇女"
        assert cards[1].gender == "女"
        assert cards[1].relation_to_household_head == "妻"

    def test_two_column_native_place(self, extractor: FieldExtractor):
        """多列宽表中籍贯字段正确分配"""
        text = (
            "| 字段名 | 户主 | 妻 |\n"
            "|--------|------|-----|\n"
            "| 籍贯 | 辽宁省抚顺县肖家沟 | 河北省任邱县七间房乡 |\n"
        )
        cards = extractor.extract_multi(text)
        assert len(cards) == 2
        assert "辽宁" in cards[0].native_place
        assert "河北" in cards[1].native_place

    def test_single_column_returns_one_card(self, extractor: FieldExtractor):
        """单列宽表仍返回含一个 HouseholdCard 的列表"""
        text = (
            "| 字段名 | 值 |\n"
            "|--------|-----|\n"
            "| 姓名 | 张三 |\n"
            "| 性别 | 男 |\n"
        )
        cards = extractor.extract_multi(text)
        assert isinstance(cards, list)
        assert len(cards) >= 1
        assert cards[0].name == "张三"

    def test_raw_markdown_set_on_all_cards(self, extractor: FieldExtractor):
        """所有返回的卡片都应有 raw_markdown"""
        text = (
            "| 字段名 | 户主 | 妻 |\n"
            "|--------|------|-----|\n"
            "| 姓名 | 王盛祥 | 马妇女 |\n"
        )
        cards = extractor.extract_multi(text)
        for card in cards:
            assert card.raw_markdown == text

    def test_three_column_table(self, extractor: FieldExtractor):
        """三列宽表应解析出 3 个 HouseholdCard"""
        text = (
            "| 字段名 | 成员1 | 成员2 | 成员3 |\n"
            "|--------|-------|-------|-------|\n"
            "| 姓名 | 张大 | 张二 | 张三 |\n"
            "| 性别 | 男 | 女 | 男 |\n"
        )
        cards = extractor.extract_multi(text)
        assert len(cards) == 3
        assert cards[0].name == "张大"
        assert cards[1].name == "张二"
        assert cards[2].name == "张三"

    def test_fallback_to_extract_for_non_table(self, extractor: FieldExtractor):
        """非多列格式时 extract_multi 回退为 [extract()]"""
        data = json.dumps({"姓名": "张三", "性别": "男"})
        cards = extractor.extract_multi(data)
        assert isinstance(cards, list)
        assert len(cards) >= 1
        assert cards[0].name == "张三"
