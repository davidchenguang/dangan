"""测试 core/layout_analyzer.py — 版面分析（占位实现）"""

from core.layout_analyzer import LayoutAnalyzer, LayoutResult


class TestLayoutAnalyzer:
    def test_placeholder_returns_empty(self):
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze("test.jpg")
        assert isinstance(result, LayoutResult)
        assert result.regions == {}

    def test_layout_result_default(self):
        result = LayoutResult()
        assert result.regions == {}
