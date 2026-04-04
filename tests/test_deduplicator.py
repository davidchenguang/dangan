"""测试 core/deduplicator.py — 去重引擎"""

import pytest

from core.deduplicator import deduplicate_output


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
