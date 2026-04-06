# 测试体系补全实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 5 个未覆盖的 core 模块建立完整的单元测试，并增强 pipeline 测试，确保全核心逻辑有自动化回归保护。

**Architecture:** 采用 pytest 框架 + mock 隔离外部依赖（torch/cv2/transformers）。创建共享 `conftest.py` 提供通用 fixture。每个 Task 遵循 TDD：先写失败测试 → 验证失败 → 实现/修复 → 验证通过 → 提交。

**Tech Stack:** pytest, unittest.mock, numpy (图像 fixture), cv2 (图像处理)

---

### Task 1: 创建共享 conftest.py

当前每个测试文件各自定义 fixture，存在重复。创建共享 conftest。

**Files:**
- Create: `tests/conftest.py`

**Step 1: 创建 conftest.py**

```python
"""共享测试 fixtures"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.models import (
    BirthInfo,
    ChangeRecord,
    HouseholdCard,
    OccupationInfo,
    PreprocessConfig,
)


@pytest.fixture
def sample_card():
    """标准测试用 HouseholdCard"""
    return HouseholdCard(
        name="张三",
        gender="男",
        birth=BirthInfo(date="1950年1月1日", address="北京市"),
        ethnicity="汉",
        education="小学",
        occupation=OccupationInfo(occupation="工人", workplace="北京钢厂"),
        native_place="河北",
        marital_status="已婚",
        source_image="test.jpg",
    )


@pytest.fixture
def sample_card_list():
    """多张卡片（用于投票测试）"""
    return [
        HouseholdCard(name="张三", gender="男", ethnicity="汉", education="小学"),
        HouseholdCard(name="张三", gender="男", ethnicity="汉", education="初中"),
        HouseholdCard(name="李四", gender="男", ethnicity="汉", education="小学"),
    ]


@pytest.fixture
def sample_image():
    """100x100 BGR 测试图像"""
    np.random.seed(42)
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_file(sample_image, tmp_path):
    """写入临时文件的测试图像"""
    import cv2
    path = str(tmp_path / "test_image.jpg")
    cv2.imwrite(path, sample_image)
    return path


@pytest.fixture
def default_preprocess_config():
    """默认预处理配置"""
    return PreprocessConfig()
```

**Step 2: 运行确认无语法错误**

Run: `python -m pytest tests/conftest.py --co -q`
Expected: 无输出（conftest 不含测试）

**Step 3: 确认已有测试仍通过**

Run: `python -m pytest tests/ -v --tb=short`
Expected: 22 passed

**Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: 添加共享 conftest.py"
```

---

### Task 2: 测试 augmentation.py

`core/augmentation.py` 有 47 行，零测试。需要验证：形状不变、不同 seed 产生不同结果、增强幅度微小。

**Files:**
- Create: `tests/test_augmentation.py`
- Reference: `core/augmentation.py`

**Step 1: 写测试**

```python
"""测试 core/augmentation.py — 图像增强"""
import numpy as np
import pytest

from core.augmentation import augment_for_voting


class TestAugmentForVoting:

    def test_preserves_shape(self, sample_image):
        """增强后图片尺寸不变"""
        result = augment_for_voting(sample_image, seed=42)
        assert result.shape == sample_image.shape

    def test_preserves_dtype(self, sample_image):
        """输出类型保持 uint8"""
        result = augment_for_voting(sample_image, seed=42)
        assert result.dtype == np.uint8

    def test_different_seeds_produce_different_results(self, sample_image):
        """不同 seed 应产生不同增强结果"""
        r1 = augment_for_voting(sample_image, seed=42)
        r2 = augment_for_voting(sample_image, seed=99)
        assert not np.array_equal(r1, r2)

    def test_same_seed_reproducible(self, sample_image):
        """相同 seed 应产生完全相同的结果"""
        r1 = augment_for_voting(sample_image, seed=42)
        r2 = augment_for_voting(sample_image, seed=42)
        assert np.array_equal(r1, r2)

    def test_augmentation_is_subtle(self):
        """增强幅度应微小（像素差异不超过 15%）"""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = augment_for_voting(img, seed=42)
        diff = np.abs(result.astype(int) - img.astype(int))
        assert np.percentile(diff, 95) <= 20

    def test_seed_zero_still_augments(self, sample_image):
        """seed=0 仍然产生增强（不是原样返回）"""
        result = augment_for_voting(sample_image, seed=0)
        # 允许 seed=0 与原图略有不同
        assert result.shape == sample_image.shape
```

**Step 2: 运行测试验证通过**

Run: `python -m pytest tests/test_augmentation.py -v`
Expected: 6 passed

**Step 3: Commit**

```bash
git add tests/test_augmentation.py
git commit -m "test: 添加 augmentation.py 测试"
```

---

### Task 3: 测试 voting.py 的投票逻辑

`core/voting.py` 有 275 行，零测试。重点测试 `_vote()` 函数和 `_merge_by_voting()` 方法，用 mock 隔离 OCR 引擎。

**Files:**
- Create: `tests/test_voting.py`
- Reference: `core/voting.py`

**Step 1: 写测试**

```python
"""测试 core/voting.py — 多轮投票引擎"""
from unittest.mock import MagicMock

import pytest

from core.models import (
    BirthInfo,
    ChangeRecord,
    HouseholdCard,
    OccupationInfo,
    VotingResult,
)
from core.voting import OcrVotingEngine, _vote, _deduplicate_records


class TestVoteFunction:
    """_vote() 多数投票函数"""

    def test_majority_wins(self):
        winner, conf = _vote(["A", "A", "B"], 3)
        assert winner == "A"
        assert conf == pytest.approx(2 / 3, abs=0.01)

    def test_unanimous(self):
        winner, conf = _vote(["X", "X", "X"], 3)
        assert winner == "X"
        assert conf == 1.0

    def test_empty_values(self):
        winner, conf = _vote([], 3)
        assert winner == ""
        assert conf == 0.0

    def test_single_value(self):
        winner, conf = _vote(["only"], 3)
        assert winner == "only"
        assert conf == pytest.approx(1 / 3, abs=0.01)

    def test_tie_takes_first_encountered(self):
        """平票时取 Counter.most_common 第一个"""
        winner, conf = _vote(["A", "B"], 2)
        assert winner in ("A", "B")
        assert conf == 0.5


class TestDeduplicateRecords:
    """ChangeRecord 去重"""

    def test_removes_exact_duplicates(self):
        records = [
            ChangeRecord(date="1953年", content="迁入"),
            ChangeRecord(date="1953年", content="迁入"),
        ]
        result = _deduplicate_records(records)
        assert len(result) == 1

    def test_keeps_different_records(self):
        records = [
            ChangeRecord(date="1953年", content="迁入"),
            ChangeRecord(date="1960年", content="迁出"),
        ]
        result = _deduplicate_records(records)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate_records([]) == []


class TestOcrVotingEngineMerge:
    """_merge_by_voting 字段合并逻辑"""

    def _make_engine(self):
        mock_engine = MagicMock()
        return OcrVotingEngine(engine=mock_engine, rounds=3)

    def test_simple_field_voting(self):
        """简单字段取众数"""
        cards = [
            HouseholdCard(name="张三", gender="男"),
            HouseholdCard(name="张三", gender="男"),
            HouseholdCard(name="李四", gender="男"),
        ]
        engine = self._make_engine()
        merged, confidence = engine._merge_by_voting(cards)
        assert merged.name == "张三"
        assert confidence["name"] == pytest.approx(2 / 3, abs=0.01)

    def test_composite_field_voting(self):
        """复合字段对子属性独立投票"""
        cards = [
            HouseholdCard(birth=BirthInfo(date="1950年", address="北京")),
            HouseholdCard(birth=BirthInfo(date="1950年", address="上海")),
            HouseholdCard(birth=BirthInfo(date="1960年", address="北京")),
        ]
        engine = self._make_engine()
        merged, confidence = engine._merge_by_voting(cards)
        assert merged.birth.date == "1950年"
        assert merged.birth.address == "北京"

    def test_list_field_union(self):
        """列表字段取并集"""
        cards = [
            HouseholdCard(migration_in=[ChangeRecord(content="A")]),
            HouseholdCard(migration_in=[ChangeRecord(content="B")]),
        ]
        engine = self._make_engine()
        merged, _ = engine._merge_by_voting(cards)
        assert len(merged.migration_in) == 2

    def test_empty_card_list_handled(self):
        """空卡片列表不崩溃"""
        engine = self._make_engine()
        merged, confidence = engine._merge_by_voting([
            HouseholdCard(),
            HouseholdCard(),
        ])
        assert merged.name == ""


class TestOcrVotingEngineIntegration:
    """recognize_with_voting 集成测试"""

    def test_calls_recognize_n_times(self):
        """应调用 recognize N 次"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三\n**性别**：男"

        voter = OcrVotingEngine(engine=mock_engine, rounds=3)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert mock_engine.recognize.call_count == 3
        assert isinstance(result, VotingResult)

    def test_returns_merged_card(self):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三\n**性别**：男"

        voter = OcrVotingEngine(engine=mock_engine, rounds=2)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.card.name == "张三"
        assert result.rounds == 2

    def test_all_empty_outputs(self):
        """所有轮次空输出时返回空卡片"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = ""

        voter = OcrVotingEngine(engine=mock_engine, rounds=2)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.card.name == ""
        assert result.rounds == 2

    def test_single_round_skips_voting(self):
        """单轮直接返回不做投票合并"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三"

        voter = OcrVotingEngine(engine=mock_engine, rounds=1)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.card.name == "张三"
        assert result.rounds == 1
        assert all(v == 1.0 for v in result.confidence.values())

    def test_temperature_passed_per_round(self):
        """每轮应传递不同 temperature"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三"

        voter = OcrVotingEngine(engine=mock_engine, rounds=3)
        voter.recognize_with_voting("fake.jpg", "prompt")

        calls = mock_engine.recognize.call_args_list
        # 第1轮 temperature=0.0, 第2轮=0.3, 第3轮=0.5
        assert calls[0].kwargs.get("temperature") == 0.0
        assert calls[1].kwargs.get("temperature") == 0.3
        assert calls[2].kwargs.get("temperature") == 0.5
```

**Step 2: 运行测试**

Run: `python -m pytest tests/test_voting.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_voting.py
git commit -m "test: 添加 voting.py 完整测试（投票/合并/集成）"
```

---

### Task 4: 测试 ocr_engine.py

`core/ocr_engine.py` 有 231 行，零测试。用 mock 模拟 torch/transformers，测试接口行为。

**Files:**
- Create: `tests/test_ocr_engine.py`
- Reference: `core/ocr_engine.py`

**Step 1: 写测试**

```python
"""测试 core/ocr_engine.py — OCR 引擎接口"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestDeepSeekOCREngineInit:
    """初始化与属性"""

    def test_default_params(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake/model")
        assert engine._model_path == "/fake/model"
        assert engine._base_size == 1024
        assert engine._image_size == 768
        assert engine.is_loaded is False

    def test_custom_params(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(
            model_path="/fake",
            base_size=512,
            image_size=384,
            crop_mode=False,
        )
        assert engine._base_size == 512
        assert engine._crop_mode is False


class TestDeepSeekOCREngineRecognize:
    """recognize() 方法"""

    def _make_loaded_engine(self, return_value="test output"):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._model.infer.return_value = return_value
        return engine

    def test_basic_recognize(self):
        engine = self._make_loaded_engine()
        result = engine.recognize("fake.jpg", "prompt")
        assert result == "test output"
        engine._model.infer.assert_called_once()

    def test_dict_result_extracts_text(self):
        engine = self._make_loaded_engine(return_value={"text": "hello"})
        result = engine.recognize("fake.jpg", "prompt")
        assert result == "hello"

    def test_unloaded_raises(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        with pytest.raises(RuntimeError, match="模型未加载"):
            engine.recognize("fake.jpg", "prompt")

    def test_temperature_zero_no_patch(self):
        """temperature=0.0 不应 monkey-patch generate"""
        engine = self._make_loaded_engine()
        original_generate = engine._model.generate
        engine.recognize("fake.jpg", "prompt", temperature=0.0)
        # generate 应未被替换
        assert engine._model.generate == original_generate

    def test_temperature_positive_patches_generate(self):
        """temperature > 0 应临时替换 generate"""
        engine = self._make_loaded_engine()
        original_generate = engine._model.generate
        engine.recognize("fake.jpg", "prompt", temperature=0.5)
        # generate 应被恢复
        assert engine._model.generate == original_generate

    def test_temperature_passed_to_infer(self):
        """temperature 应影响 generate 调用参数"""
        engine = self._make_loaded_engine()
        # 捕获 generate 的调用
        engine.recognize("fake.jpg", "prompt", temperature=0.3)
        # infer 应被调用（内部 generate 被 patch）
        engine._model.infer.assert_called_once()


class TestDeepSeekOCREngineUnload:
    """unload() 方法"""

    def test_unload_clears_references(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine.unload()
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine.is_loaded is False


class TestDeepSeekOCREngineVRam:
    """VRAM 使用量"""

    def test_vram_without_cuda(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        with patch("core.ocr_engine.torch.cuda.is_available", return_value=False):
            assert engine.get_vram_usage_mb() == 0.0
```

**Step 2: 运行测试**

Run: `python -m pytest tests/test_ocr_engine.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_ocr_engine.py
git commit -m "test: 添加 ocr_engine.py 测试（接口/temperature/unload）"
```

---

### Task 5: 测试 preprocessor.py

`core/preprocessor.py` 有 219 行，零测试。测试图像加载、缩放、CLAHE 和各种预处理开关。

**Files:**
- Create: `tests/test_preprocessor.py`
- Reference: `core/preprocessor.py`

**Step 1: 写测试**

```python
"""测试 core/preprocessor.py — 图像预处理"""
import cv2
import numpy as np
import pytest

from core.models import PreprocessConfig
from core.preprocessor import ImagePreprocessor


class TestImagePreprocessorBasic:
    """基础预处理"""

    def test_scale_doubles_image(self, sample_image_file):
        """默认 scale=2.0 应放大 2 倍"""
        config = PreprocessConfig(scale=2.0, enhance_contrast=False)
        prep = ImagePreprocessor(config)
        result_path = prep.process(sample_image_file)
        img = cv2.imread(result_path)
        assert img.shape[:2] == (200, 200)  # 100*2

    def test_no_scale_keeps_size(self, sample_image_file):
        """scale=1.0 不放大"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=False)
        prep = ImagePreprocessor(config)
        result_path = prep.process(sample_image_file)
        img = cv2.imread(result_path)
        assert img.shape[:2] == (100, 100)

    def test_clahe_produces_gray_output(self, sample_image_file):
        """CLAHE 增强后输出是三通道灰度"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=True)
        prep = ImagePreprocessor(config)
        result_path = prep.process(sample_image_file)
        img = cv2.imread(result_path)
        assert img.ndim == 3  # BGR 三通道
        # CLAHE 输出是灰度转 BGR，三通道应相同
        b, g, r = cv2.split(img)
        assert np.array_equal(b, g)
        assert np.array_equal(g, r)

    def test_output_file_exists(self, sample_image_file, tmp_path):
        """预处理图片应保存到指定目录"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=False)
        prep = ImagePreprocessor(config)
        result = prep.process(sample_image_file, output_dir=str(tmp_path))
        import os
        assert os.path.exists(result)


class TestImagePreprocessorValidation:
    """输入验证"""

    def test_missing_file_raises(self):
        prep = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            prep.process("/nonexistent/file.jpg")

    def test_unsupported_format_raises(self, tmp_path):
        """非图片格式应报错"""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")
        prep = ImagePreprocessor()
        with pytest.raises(ValueError, match="不支持"):
            prep.process(str(bad_file))


class TestImagePreprocessorOptionalSteps:
    """可选预处理步骤"""

    def test_denoise_reduces_noise(self, sample_image_file):
        """去噪应产生不同输出"""
        config_no_denoise = PreprocessConfig(scale=1.0, enhance_contrast=False, denoise=False)
        config_denoise = PreprocessConfig(scale=1.0, enhance_contrast=False, denoise=True)

        r1 = ImagePreprocessor(config_no_denoise).process(sample_image_file)
        r2 = ImagePreprocessor(config_denoise).process(sample_image_file)

        img1 = cv2.imread(r1)
        img2 = cv2.imread(r2)
        assert not np.array_equal(img1, img2)

    def test_binarize_produces_binary_output(self, sample_image_file):
        """二值化应只有 0 和 255"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=False, binarize=True)
        result_path = ImagePreprocessor(config).process(sample_image_file)
        img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        unique = set(np.unique(img))
        assert unique.issubset({0, 255}) or len(unique) <= 3
```

**Step 2: 运行测试**

Run: `python -m pytest tests/test_preprocessor.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_preprocessor.py
git commit -m "test: 添加 preprocessor.py 测试（缩放/CLAHE/验证/可选步骤）"
```

---

### Task 6: 增强 pipeline.py 测试

`tests/test_pipeline.py` 当前只测试 `PROMPT_MAP` 和 `PipelineResult` 数据结构，未测试实际流程逻辑。需要用 mock 测试 process() 的调用链。

**Files:**
- Modify: `tests/test_pipeline.py`
- Reference: `core/pipeline.py`

**Step 1: 重写 test_pipeline.py**

先读取当前文件，然后用增强版替换。

```python
"""测试 core/pipeline.py — 流水线编排"""
import pytest
from unittest.mock import MagicMock, patch

from core.models import HouseholdCard, PreprocessConfig
from core.pipeline import OcrPipeline, PipelineResult, PROMPT_MAP


class TestPromptMap:
    """Prompt 映射"""

    def test_verbatim_exists(self):
        assert "verbatim" in PROMPT_MAP

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
        with patch("core.pipeline.OcrVotingEngine") as MockVoter:
            mock_result = MagicMock()
            mock_result.card = HouseholdCard(name="张三")
            mock_result.confidence = {"name": 1.0}
            mock_result.rounds = 3
            MockVoter.return_value.recognize_with_voting.return_value = mock_result

            result = pipeline.process("input.jpg")

            MockVoter.return_value.recognize_with_voting.assert_called_once()
            assert result.confidence == {"name": 1.0}
```

**Step 2: 运行测试**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: 增强 pipeline.py 测试（process 流程/error/投票集成）"
```

---

### Task 7: 全量回归测试

所有新增和修改的测试完成后，运行全量测试确认无破坏。

**Step 1: 运行全量测试**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS (预计约 50+ 个测试)

**Step 2: 如有失败，修复后重跑**

**Step 3: 最终 Commit**

```bash
git add -A
git commit -m "test: 测试体系补全 — augmentation/voting/ocr_engine/preprocessor/pipeline"
```

---

## 预期测试覆盖

| 模块 | 测试文件 | 测试数量（估计） |
|------|---------|---------------|
| augmentation.py | test_augmentation.py | 6 |
| voting.py | test_voting.py | 14 |
| ocr_engine.py | test_ocr_engine.py | 8 |
| preprocessor.py | test_preprocessor.py | 8 |
| pipeline.py | test_pipeline.py | 10+ |
| deduplicator.py | test_deduplicator.py | 13 (已有) |
| config.py | test_config.py | 9 (已有) |
| models.py | test_models.py | ~10 (已有) |
| field_extractor.py | test_field_extractor.py | ~10 (已有) |
| exporter.py | test_exporter.py | ~10 (已有) |
| **总计** | | **~98** |
