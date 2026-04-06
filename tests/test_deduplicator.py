"""测试 core/deduplicator.py — 去重引擎"""

import pytest

from core.deduplicator import (
    _line_similarity,
    _truncate_mutating_duplicates,
    deduplicate_output,
)


class TestDeduplicateOutput:
    """去重引擎测试"""

    def test_empty_string(self):
        assert deduplicate_output("") == ""

    def test_no_duplicates(self):
        text = "第一行\n第二行\n第三行"
        assert deduplicate_output(text) == text

    def test_consecutive_duplicates(self):
        text = "有效内容1\n有效内容2\n重复行\n重复行\n重复行\n重复行"
        result = deduplicate_output(text)
        assert "有效内容1" in result
        assert "有效内容2" in result
        assert result.count("重复行") == 1
        assert "截断" in result

    def test_min_repeat_threshold(self):
        """低于阈值的重复不应被截断"""
        text = "行A\n行A\n行A"  # 恰好3次
        # min_repeat=3 时，3次应触发截断
        result = deduplicate_output(text, min_repeat=3)
        assert "截断" in result

    def test_below_threshold(self):
        """只有2次重复不应被截断"""
        text = "行A\n行A"
        result = deduplicate_output(text, min_repeat=3)
        assert result == text

    def test_short_lines_detected(self):
        """短行（>=2字符）也会参与重复检测"""
        text = "ab\nab\nab\nab\nab"
        result = deduplicate_output(text, min_repeat=3)
        assert "截断" in result

    def test_repeating_paragraphs(self):
        """测试段落级重复"""
        para = "| 字段 | 值 |\n|------|----|\n| 姓名 | 张三 |"
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
        result = deduplicate_output(text, min_repeat=3)
        assert result.count("姓名") < 4  # 应被截断

    def test_preserves_valid_content_before_duplicates(self):
        """截断时应保留重复行之前的有效内容"""
        text = "重要信息1\n重要信息2\n重复行X\n重复行X\n重复行X"
        result = deduplicate_output(text)
        assert "重要信息1" in result
        assert "重要信息2" in result


class TestLineSimilarity:
    """行相似度计算"""

    def test_identical_lines(self):
        assert _line_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        assert _line_similarity("abc", "xyz") == 0.0

    def test_similar_lines(self):
        sim = _line_similarity(
            "何时由何地迁来本市：1952年由无籍迁入",
            "何时由何地迁来本市的：1952由无籍迁入",
        )
        assert 0.7 < sim < 1.0


class TestMutatingDuplicates:
    """渐变式重复检测"""

    def test_truncates_gradually_mutating_lines(self):
        """应截断逐行渐变的重复内容"""
        lines = [
            "有效内容第一行",
            "何时由何地迁来本市：1952年由无籍迁入",
            "何时由何地迁来本市的：1952由无籍迁入",
            "何时由何地迁来本市的：1952由无籍迁入",
            "何时由何种地迁来本市：1952由无籍迁入",
        ]
        result = _truncate_mutating_duplicates(lines, min_repeat=3)
        assert result is not None
        assert "有效内容第一行" in result

    def test_preserves_different_content(self):
        """不相似的内容不应被截断"""
        lines = ["姓名：张三", "性别：男", "民族：汉"]
        result = _truncate_mutating_duplicates(lines, min_repeat=3)
        assert result is None
