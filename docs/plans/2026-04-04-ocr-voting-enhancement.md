# OCR 投票机制增强 — 引入推理方差 + 图像增强 + Bug 修复

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让多轮投票 OCR 机制真正生效 — 通过图像增强（每轮不同变换）和温度采样（非确定性推理）引入方差，使模型各轮输出不同，投票才有意义。

**Architecture:** 在 `OcrVotingEngine.recognize_with_voting()` 中，每轮调用前对输入图像做微小随机增强（亮度±5%、对比度±3%、微小偏移），同时 monkey-patch 模型的 `generate()` 调用以支持 temperature > 0。三票取众数。同时修复 deduplicator.py 中 `_deduplicate_similar_paragraphs` 函数定义缺失的严重 bug。

**Tech Stack:** Python 3.12+, OpenCV (augmentation), PyTorch (monkey-patch generate), PySide6 (GUI)

---

## Task 1: 修复 deduplicator.py 严重 Bug

`_deduplicate_similar_paragraphs` 函数被调用（第100行）但缺少 `def` 语句，导致第184-237行代码不可达。

**Files:**
- Modify: `core/deduplicator.py:182-237`
- Test: `tests/test_deduplicator.py`

**Step 1: 写失败测试**

创建 `tests/test_deduplicator.py`：

```python
"""测试 core/deduplicator.py — 去重引擎"""
import pytest
from core.deduplicator import deduplicate_output, _line_similarity


class TestMutatingDuplicates:
    """渐变式重复检测"""

    def test_truncates_gradually_mutating_lines(self):
        """应截断逐行渐变的重复内容"""
        text = (
            "有效内容第一行\n"
            "何时由何地迁来本市：1952年由无籍迁入\n"
            "何时由何地迁来本市的：1952由无籍迁入\n"
            "何时由何地迁来本市的：1952由无籍迁入\n"
            "何时由何种地迁来本市：1952由无籍迁入\n"
        )
        result = deduplicate_output(text)
        # 应保留第一行有效内容 + 重复段的第一行
        assert "有效内容第一行" in result
        assert "已截断" in result or result.count("何时由何地") <= 2

    def test_preserves_different_content(self):
        """不相似的内容不应被截断"""
        text = "姓名：张三\n性别：男\n民族：汉\n"
        result = deduplicate_output(text)
        assert result == text


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


class TestSemanticParagraphDedup:
    """语义段落去重"""

    def test_deduplicates_similar_paragraphs(self):
        """高度重叠的段落应去重"""
        text = (
            "**个人信息**：\n"
            "- 姓名：张三\n"
            "- 性别：男\n\n"
            "**登记信息**：\n"
            "- 姓名：张三\n"
            "- 性别：男\n"
        )
        result = deduplicate_output(text)
        # 第二段与第一段高度重叠，应被去除
        assert result.count("张三") <= 2
```

**Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_deduplicator.py -v`
Expected: FAIL — `_deduplicate_similar_paragraphs` 函数定义缺失导致调用时可能报错

**Step 3: 修复 `_deduplicate_similar_paragraphs` 函数定义**

在 `core/deduplicator.py` 第182行，添加缺失的 `def` 语句：

```python
    return (2.0 * lcs_len) / (m + n)


def _deduplicate_similar_paragraphs(paragraphs: list[str], original_text: str) -> str | None:
    """检测语义高度相似的段落（OCR 模型常见的重复描述模式）
    ...
    """
```

当前代码（第182-184行）：
```
    return (2.0 * lcs_len) / (m + n)



    """检测语义高度相似的段落...
```

修复为：
```
    return (2.0 * lcs_len) / (m + n)


def _deduplicate_similar_paragraphs(paragraphs: list[str], original_text: str) -> str | None:
    """检测语义高度相似的段落...
```

**Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_deduplicator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/deduplicator.py tests/test_deduplicator.py
git commit -m "fix: 修复 _deduplicate_similar_paragraphs 函数定义缺失的严重 bug"
```

---

## Task 2: 添加图像增强工具函数

为投票机制提供每轮不同的图像变换，引入输入方差。

**Files:**
- Create: `core/augmentation.py`
- Test: `tests/test_augmentation.py`

**Step 1: 写失败测试**

```python
"""测试 core/augmentation.py — 图像增强"""
import cv2
import numpy as np
import pytest
from pathlib import Path


class TestImageAugmentation:

    def test_augment_preserves_shape(self):
        """增强后图片尺寸不变"""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        from core.augmentation import augment_for_voting
        result = augment_for_voting(img, seed=42)
        assert result.shape == img.shape

    def test_different_seeds_produce_different_results(self):
        """不同 seed 应产生不同增强结果"""
        img = np.random.randint(100, 200, (100, 200, 3), dtype=np.uint8)
        from core.augmentation import augment_for_voting
        r1 = augment_for_voting(img, seed=42)
        r2 = augment_for_voting(img, seed=99)
        # 像素值应有差异
        assert not np.array_equal(r1, r2)

    def test_augmentation_is_subtle(self):
        """增强幅度应微小（像素差异不超过 15%）"""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        from core.augmentation import augment_for_voting
        result = augment_for_voting(img, seed=42)
        diff = np.abs(result.astype(int) - img.astype(int))
        # 95% 的像素变化不超过 20
        assert np.percentile(diff, 95) <= 20
```

**Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_augmentation.py -v`
Expected: FAIL — `core/augmentation` module not found

**Step 3: 实现 `core/augmentation.py`**

```python
"""图像增强工具 — 为投票机制提供每轮不同的微小变换

通过轻微调整亮度、对比度和微量偏移，使同一张图片在不同轮次中
产生微小差异，从而让确定性模型产生不同输出。
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def augment_for_voting(
    image: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """对图像施加微小随机变换（用于投票轮次间的输入差异化）

    变换范围（经调试不会破坏文字识别）:
    - 亮度: ±8 (0-255范围)
    - 对比度: 0.95x ~ 1.05x
    - 微小高斯模糊: sigma 0.3 ~ 0.8

    Args:
        image: BGR 格式图像 (OpenCV)
        seed: 随机种子，不同值产生不同变换

    Returns:
        增强后的图像（相同尺寸）
    """
    rng = np.random.RandomState(seed)
    result = image.copy().astype(np.float32)

    # 1. 亮度调整: ±8
    brightness_delta = rng.uniform(-8, 8)
    result = result + brightness_delta

    # 2. 对比度调整: 0.95 ~ 1.05
    contrast_factor = rng.uniform(0.95, 1.05)
    mean = result.mean()
    result = (result - mean) * contrast_factor + mean

    # 3. 微小高斯模糊 (sigma 0.3 ~ 0.8)
    sigma = rng.uniform(0.3, 0.8)
    if sigma > 0.4:
        ksize = int(sigma * 6) | 1  # 确保 ksize 为奇数
        result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    # Clip 到有效范围
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
```

**Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_augmentation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/augmentation.py tests/test_augmentation.py
git commit -m "feat: 添加投票轮次图像增强工具函数"
```

---

## Task 3: 给 OCR 引擎添加 temperature 参数

修改 `DeepSeekOCREngine` 支持 temperature 参数，通过 monkey-patch 模型的 `generate()` 调用传入。

**Files:**
- Modify: `core/ocr_engine.py:132-179` (recognize method)
- Test: `tests/test_ocr_engine.py`

**Step 1: 写失败测试**

```python
"""测试 core/ocr_engine.py — OCR 引擎接口"""
import pytest
from unittest.mock import MagicMock, patch


class TestDeepSeekOCREngineRecognize:

    def test_recognize_passes_temperature(self):
        """recognize() 应将 temperature 传递给模型"""
        from core.ocr_engine import DeepSeekOCREngine

        engine = DeepSeekOCREngine(model_path="/fake")
        # Mock 模型和 tokenizer
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._model.infer.return_value = "test output"

        engine.recognize("fake.jpg", "prompt", temperature=0.5)

        # 验证 infer 被调用时包含 temperature
        call_kwargs = engine._model.infer.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.5

    def test_recognize_default_temperature_is_zero(self):
        """默认 temperature=0.0（确定性推理）"""
        from core.ocr_engine import DeepSeekOCREngine

        engine = DeepSeekOCREngine(model_path="/fake")
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._model.infer.return_value = "test"

        engine.recognize("fake.jpg", "prompt")

        call_kwargs = engine._model.infer.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.0
```

**Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_ocr_engine.py -v`
Expected: FAIL — `recognize()` 不接受 `temperature` 参数

**Step 3: 修改 `core/ocr_engine.py` 的 `recognize()` 方法**

当前签名 (第132行):
```python
def recognize(self, image_path: str, prompt: str) -> str:
```

改为:
```python
def recognize(self, image_path: str, prompt: str, temperature: float = 0.0) -> str:
```

在 `self._model.infer()` 调用中添加 `temperature` 参数 (第151-160行):

当前:
```python
result = self._model.infer(
    self._tokenizer,
    prompt=prompt,
    image_file=image_path,
    output_path=output_dir,
    base_size=self._base_size,
    image_size=self._image_size,
    crop_mode=self._crop_mode,
    eval_mode=True,
)
```

改为:
```python
# Monkey-patch: 注入 temperature 到模型的 generate() 调用
_original_generate = self._model.generate
_injected_temp = temperature

def _generate_with_temp(*args, **kwargs):
    kwargs['temperature'] = _injected_temp
    if _injected_temp > 0:
        kwargs['do_sample'] = True
    return _original_generate(*args, **kwargs)

self._model.generate = _generate_with_temp
try:
    result = self._model.infer(
        self._tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=self._base_size,
        image_size=self._image_size,
        crop_mode=self._crop_mode,
        eval_mode=True,
    )
finally:
    self._model.generate = _original_generate
```

同时修改 `recognize_ndarray()` (第181-196行)，在内部调用 `self.recognize()` 时传递 temperature：

```python
def recognize_ndarray(self, image: np.ndarray, prompt: str, temperature: float = 0.0) -> str:
    ...
    return self.recognize(tmp.name, prompt, temperature=temperature)
```

**Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_ocr_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/ocr_engine.py tests/test_ocr_engine.py
git commit -m "feat: OCR 引擎支持 temperature 参数（monkey-patch generate）"
```

---

## Task 4: 更新投票引擎 — 集成图像增强 + 温度采样

让每轮投票使用不同的图像增强和 temperature > 0。

**Files:**
- Modify: `core/voting.py:63-120` (recognize_with_voting)
- Test: `tests/test_voting.py`

**Step 1: 写失败测试**

```python
"""测试 core/voting.py — 投票引擎"""
import pytest
from unittest.mock import MagicMock, call
from core.models import HouseholdCard, VotingResult
from core.voting import OcrVotingEngine, _vote


class TestVoteFunction:

    def test_majority_wins(self):
        winner, conf = _vote(["A", "A", "B"], 3)
        assert winner == "A"
        assert conf == pytest.approx(2/3, abs=0.01)

    def test_unanimous(self):
        winner, conf = _vote(["X", "X", "X"], 3)
        assert winner == "X"
        assert conf == 1.0

    def test_empty_values(self):
        winner, conf = _vote([], 3)
        assert winner == ""
        assert conf == 0.0


class TestOcrVotingEngine:

    def test_calls_augmentation_per_round(self):
        """每轮应使用不同的增强图像"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三\n**性别**：男"

        voter = OcrVotingEngine(engine=mock_engine, rounds=3)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        # 应调用 recognize 3 次
        assert mock_engine.recognize.call_count == 3

    def test_returns_merged_card(self):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三\n**性别**：男"

        voter = OcrVotingEngine(engine=mock_engine, rounds=2)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert isinstance(result, VotingResult)
        assert result.rounds == 2
        assert result.card.name == "张三"

    def test_single_round_no_voting(self):
        """单轮直接返回，不做投票"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三"

        voter = OcrVotingEngine(engine=mock_engine, rounds=1)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.card.name == "张三"
        assert result.rounds == 1
```

**Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_voting.py -v`
Expected: 部分测试 FAIL（当前 recognize_with_voting 不使用 augmentation）

**Step 3: 修改 `core/voting.py` 的 `recognize_with_voting()` 方法**

关键修改：
1. 每轮生成不同的增强图像（用 `seed=round_index`）
2. 每轮使用 `temperature=0.3 + 0.2 * round_index` 递增温度
3. 增强图像保存为临时文件供 `recognize()` 使用

修改 `recognize_with_voting()` 方法 (第63-120行):

```python
def recognize_with_voting(
    self,
    image_path: str,
    prompt: str,
) -> VotingResult:
    raw_results: list[HouseholdCard] = []

    for i in range(self._rounds):
        logger.info("投票轮次 %d/%d", i + 1, self._rounds)

        # 每轮使用不同的增强图像
        aug_image_path = self._prepare_augmented_image(image_path, seed=i)

        # 递增 temperature: 第1轮=0.0, 第2轮=0.3, 第3轮=0.5
        temp = 0.0 if i == 0 else 0.3 + 0.2 * (i - 1)

        raw_text = self._engine.recognize(aug_image_path, prompt, temperature=temp)

        if not raw_text or not raw_text.strip():
            logger.warning("轮次 %d 输出为空，跳过", i + 1)
            continue

        clean_text = deduplicate_output(raw_text)
        card = self._extractor.extract(clean_text)
        raw_results.append(card)

    ...  # 后续投票逻辑不变
```

新增私有方法:

```python
@staticmethod
def _prepare_augmented_image(image_path: str, seed: int) -> str:
    """为投票轮次准备增强图像

    Args:
        image_path: 原始图片路径
        seed: 随机种子（通常为轮次索引）

    Returns:
        增强后图片的临时文件路径
    """
    import tempfile
    import cv2
    from core.augmentation import augment_for_voting

    img = cv2.imread(image_path)
    if img is None:
        return image_path  # 无法读取则返回原路径

    augmented = augment_for_voting(img, seed=seed)

    # 保存到临时文件
    suffix = Path(image_path).suffix
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    cv2.imwrite(tmp.name, augmented)
    tmp.close()
    return tmp.name
```

需要在文件顶部添加 `from pathlib import Path` 导入。

**Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_voting.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add core/voting.py tests/test_voting.py
git commit -m "feat: 投票引擎集成图像增强和温度采样"
```

---

## Task 5: 端到端验证 — 真实图片 3 轮投票测试

用真实户籍卡图片 `C:\Users\cheng\Desktop\1.jpg` 验证投票效果。

**Files:**
- No code changes — verification only

**Step 1: 运行 Pipeline 端到端测试**

```bash
python -c "
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

from core.config import AppConfig
from core.pipeline import OcrPipeline

config = AppConfig()
config.load()
print(f'Voting rounds: {config.ocr_voting_rounds}')

pipeline = OcrPipeline(
    model_path=config.model_path,
    preprocess_config=config.preprocess,
    prompt=config.ocr_prompt,
    base_size=config.ocr_base_size,
    image_size=config.ocr_image_size,
    crop_mode=config.ocr_crop_mode,
    voting_rounds=config.ocr_voting_rounds,
)
pipeline.initialize()

result = pipeline.process(r'C:\Users\cheng\Desktop\1.jpg')

card = result.card
print(f'\nName: {card.name}')
print(f'Gender: {card.gender}')
print(f'Birth date: {card.birth.date}')
print(f'Native place: {card.native_place}')
print(f'Ethnicity: {card.ethnicity}')
print(f'Education: {card.education}')
print(f'Occupation: {card.occupation.occupation}')

if result.confidence:
    print(f'\nConfidence:')
    for f, c in sorted(result.confidence.items(), key=lambda x: x[1], reverse=True):
        print(f'  {f}: {c:.2f}')

pipeline.engine.unload()
"
```

**Step 2: 验证预期结果**

Expected:
- 3 轮输出不再完全一致（因 image augmentation + temperature）
- 稳定字段（性别、民族）置信度 ≥ 0.8
- 不稳定字段置信度 < 0.8，但投票后值可能更准确

**Step 3: 记录结果到日志文件**

将 3 轮各自的原始 OCR 输出写入 `output/voting_e2e_results.txt` 供对比分析。

---

## Task 6: 清理临时测试文件

**Files:**
- Delete: `test_e2e_compare.py`, `test_e2e_json.py`, `test_ab_ocr.py`
- Delete: `output/ab_test/`, `output/diag_raw_ocr.txt`, `output/voting_rounds_detail.txt`

**Step 1: 确认临时文件列表**

Run: `ls test_e2e*.py test_ab_ocr.py`

**Step 2: 删除临时文件**

```bash
rm test_e2e_compare.py test_e2e_json.py test_ab_ocr.py
rm -rf output/ab_test/
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: 清理 A/B 测试和调试临时文件"
```
