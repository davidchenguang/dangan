"""测试 core/pipeline.py — 流水线编排"""
import pytest
from unittest.mock import MagicMock, patch

from core.models import HouseholdCard, PreprocessConfig
from core.pipeline import OcrPipeline, PipelineResult, PROMPT_MAP


class TestPromptMap:
    """Prompt 映射"""

    def test_verbatim_exists(self):
        assert "verbatim" in PROMPT_MAP

    def test_structured_exists(self):
        assert "structured" in PROMPT_MAP

    def test_markdown_exists(self):
        assert "markdown" in PROMPT_MAP

    def test_json_exists(self):
        assert "json" in PROMPT_MAP

    def test_fallback_is_verbatim(self):
        assert PROMPT_MAP.get("unknown", PROMPT_MAP["verbatim"]) == PROMPT_MAP["verbatim"]


class TestPipelineResult:
    """PipelineResult 数据结构"""

    def test_default_values(self):
        result = PipelineResult(card=HouseholdCard())
        assert result.success is True
        assert result.error == ""
        assert result.confidence is None
        assert result.elapsed_seconds == 0.0

    def test_error_result(self):
        result = PipelineResult(
            card=HouseholdCard(),
            success=False,
            error="测试错误",
        )
        assert result.success is False
        assert result.error == "测试错误"


class TestOcrPipelineInit:
    """OcrPipeline 初始化"""

    def test_default_voting_rounds(self):
        pipeline = OcrPipeline(model_path="/fake")
        assert pipeline._voting_rounds == 1
        assert pipeline._prompt_name == "structured"

    def test_custom_voting_rounds(self):
        pipeline = OcrPipeline(model_path="/fake", voting_rounds=3)
        assert pipeline._voting_rounds == 3

    def test_prompt_name_stored(self):
        pipeline = OcrPipeline(model_path="/fake", prompt="markdown")
        assert pipeline._prompt_name == "markdown"


class TestOcrPipelineProcess:
    """process() 方法 — 用 mock 隔离"""

    def _make_pipeline(self):
        pipeline = OcrPipeline(model_path="/fake", voting_rounds=1)
        pipeline._initialized = True
        pipeline._engine = MagicMock()
        pipeline._preprocessor = MagicMock()
        pipeline._preprocessor.process.return_value = "preprocessed.jpg"
        return pipeline

    def test_single_round_calls_recognize_once(self):
        """单轮模式应调用 recognize 1 次"""
        pipeline = self._make_pipeline()
        pipeline._engine.recognize.return_value = "**姓名**：张三"

        result = pipeline.process("input.jpg", preprocess=True)

        pipeline._engine.recognize.assert_called_once()
        assert result.success is True

    def test_preprocess_called_when_enabled(self):
        """preprocess=True 时应调用预处理器"""
        pipeline = self._make_pipeline()
        pipeline._engine.recognize.return_value = "**姓名**：张三"

        result = pipeline.process("input.jpg", preprocess=True)

        pipeline._preprocessor.process.assert_called_once()

    def test_preprocess_skipped_when_disabled(self):
        """preprocess=False 时不应调用预处理器"""
        pipeline = self._make_pipeline()
        pipeline._engine.recognize.return_value = "**姓名**：张三"

        result = pipeline.process("input.jpg", preprocess=False)

        pipeline._preprocessor.process.assert_not_called()

    def test_error_returns_failure_result(self):
        """异常时返回失败结果"""
        pipeline = self._make_pipeline()
        pipeline._engine.recognize.side_effect = RuntimeError("OCR 失败")

        result = pipeline.process("input.jpg")

        assert result.success is False
        assert "OCR 失败" in result.error

    def test_multi_round_calls_voting(self):
        """voting_rounds > 1 应使用投票引擎"""
        pipeline = self._make_pipeline()
        pipeline._voting_rounds = 3

        # Mock 投票引擎
        with patch("core.voting.OcrVotingEngine") as MockVoter:
            mock_result = MagicMock()
            mock_result.card = HouseholdCard(name="张三")
            mock_result.confidence = {"name": 1.0}
            mock_result.rounds = 3
            MockVoter.return_value.recognize_with_voting.return_value = mock_result

            result = pipeline.process("input.jpg")

            MockVoter.return_value.recognize_with_voting.assert_called_once()
            assert result.confidence == {"name": 1.0}
