"""测试 core/pipeline.py — 流水线集成测试（无模型依赖)"""

import pytest

from core.models import HouseholdCard
from core.pipeline import PROMPT_MAP, PipelineResult


class TestPromptMap:
    """Prompt 映射测试"""

    def test_all_prompts_registered(self):
        assert "verbatim" in PROMPT_MAP
        assert "markdown" in PROMPT_MAP
        assert "json" in PROMPT_MAP

    def test_prompt_content(self):
        for name, prompt in PROMPT_MAP.items():
            assert "<image>" in prompt, f"Prompt '{name}' 缺少 <image> 标记"
            assert len(prompt) > 20, f"Prompt '{name}' 内容过短"


class TestPipelineResult:
    """流水线处理结果"""

    def test_success_result(self):
        card = HouseholdCard(name="test")
        result = PipelineResult(
            card=card,
            preprocessed_path="/tmp/test.jpg",
            elapsed_seconds=30.5,
            success=True,
        )
        assert result.success is True
        assert result.card.name == "test"
        assert result.elapsed_seconds == 30.5
        assert result.error == ""

    def test_failure_result(self):
        result = PipelineResult(
            card=HouseholdCard(),
            success=False,
            error="模型未加载",
        )
        assert result.success is False
        assert "模型" in result.error
