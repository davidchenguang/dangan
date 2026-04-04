"""测试 core/field_extractor.py — 字段提取器"""

import json

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
