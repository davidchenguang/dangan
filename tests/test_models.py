"""测试 core/models.py — 数据结构定义"""

import pytest

from core.models import (
    ATTR_TO_LABEL,
    BirthInfo,
    CARD_OCR_PROMPT,
    ChangeRecord,
    CitizenCertificate,
    FIELD_TO_ATTR,
    HouseholdCard,
    OccupationInfo,
    PreprocessConfig,
    ProcessingStatus,
    PROMPT_STRUCTURED,
    PROMPT_VERBATIM,
)


class TestHouseholdCard:
    """HouseholdCard 数据结构测试"""

    def test_default_values(self):
        card = HouseholdCard()
        assert card.name == ""
        assert card.alias == ""
        assert card.gender == ""
        assert card.raw_markdown == ""
        assert card.reviewed is False
        assert card.migration_in == []
        assert card.migration_local == []
        assert card.cancellation == []
        assert card.changes == []

    def test_with_values(self):
        card = HouseholdCard(
            name="张三",
            gender="男",
            native_place="浙江绍兴",
        )
        assert card.name == "张三"
        assert card.gender == "男"
        assert card.native_place == "浙江绍兴"

    def test_composite_fields(self):
        card = HouseholdCard()
        assert card.birth.date == ""
        assert card.birth.address == ""
        assert card.citizen_cert.code_number == ""
        assert card.occupation.occupation == ""

    def test_change_records(self):
        card = HouseholdCard(
            migration_in=[
                ChangeRecord(date="1953年", content="由绍兴迁来"),
                ChangeRecord(date="1955年", content="由杭州迁来"),
            ],
        )
        assert len(card.migration_in) == 2
        assert card.migration_in[0].date == "1953年"
        assert card.migration_in[1].content == "由杭州迁来"

    def test_independent_instances(self):
        """确保默认工厂产生独立实例"""
        card1 = HouseholdCard()
        card2 = HouseholdCard()
        card1.name = "张三"
        card1.migration_in.append(ChangeRecord(date="1953", content="test"))
        assert card2.name == ""
        assert card2.migration_in == []


class TestPreprocessConfig:
    """预处理配置测试"""

    def test_defaults(self):
        config = PreprocessConfig()
        assert config.scale == 2.0
        assert config.clahe_clip == 3.0
        assert config.denoise is False
        assert config.binarize is False
        assert config.enhance_contrast is True

    def test_custom_values(self):
        config = PreprocessConfig(scale=3.0, clahe_clip=5.0)
        assert config.scale == 3.0
        assert config.clahe_clip == 5.0


class TestProcessingStatus:
    """处理状态枚举测试"""

    def test_values(self):
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.DONE.value == "done"
        assert ProcessingStatus.FAILED.value == "failed"


class TestFieldMapping:
    """字段映射测试"""

    def test_field_to_attr_complete(self):
        """FIELD_TO_ATTR 应包含所有已知字段"""
        expected_fields = [
            "姓名", "别名", "性别", "籍贯", "民族",
            "宗教信仰", "婚姻状况", "文化程度",
            "出生日期", "出生地址", "职业", "服务处所",
        ]
        for f in expected_fields:
            assert f in FIELD_TO_ATTR, f"缺少字段映射: {f}"

    def test_attr_to_label_reverse(self):
        """ATTR_TO_LABEL 应是 FIELD_TO_ATTR 的反向映射"""
        for cn, attr in FIELD_TO_ATTR.items():
            assert attr in ATTR_TO_LABEL
            assert ATTR_TO_LABEL[attr] == cn


class TestPromptTemplates:
    """Prompt 模板测试"""

    def test_verbatim_prompt(self):
        assert "<image>" in PROMPT_VERBATIM
        assert "逐字" in PROMPT_VERBATIM
        assert PROMPT_VERBATIM.startswith("<image>")

    def test_structured_prompt(self):
        assert "<image>" in PROMPT_STRUCTURED
        assert "<|grounding|>" not in PROMPT_STRUCTURED
        assert "户主或与户主关系" in PROMPT_STRUCTURED
        assert "姓名" in PROMPT_STRUCTURED
        assert PROMPT_STRUCTURED.startswith("<image>")

    def test_json_prompt(self):
        assert "<image>" in CARD_OCR_PROMPT
        assert "<|grounding|>" in CARD_OCR_PROMPT
        assert "JSON" in CARD_OCR_PROMPT
        assert "姓名" in CARD_OCR_PROMPT
        assert CARD_OCR_PROMPT.startswith("<image>")
