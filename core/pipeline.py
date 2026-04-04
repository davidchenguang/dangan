"""流水线编排 — 串联预处理 → 去重 → OCR → 字段提取

完整流程:
    图片 → ImagePreprocessor → DeepSeekOCREngine → deduplicator → FieldExtractor → HouseholdCard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from core.deduplicator import deduplicate_output
from core.field_extractor import FieldExtractor
from core.models import (
    CARD_OCR_PROMPT,
    HouseholdCard,
    PreprocessConfig,
    PROMPT_MARKDOWN,
    PROMPT_VERBATIM,
)
from core.ocr_engine import DeepSeekOCREngine, OCREngine
from core.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

# Prompt 映射
PROMPT_MAP: dict[str, str] = {
    "verbatim": PROMPT_VERBATIM,
    "markdown": PROMPT_MARKDOWN,
    "json": CARD_OCR_PROMPT,
}


@dataclass
class PipelineResult:
    """流水线处理结果"""
    card: HouseholdCard
    preprocessed_path: str | None = None
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str = ""


class OcrPipeline:
    """OCR 流水线编排器

    使用方式:
        pipeline = OcrPipeline(model_path="...")
        pipeline.initialize()
        result = pipeline.process("image.jpg")
    """

    def __init__(
        self,
        model_path: str,
        preprocess_config: PreprocessConfig | None = None,
        prompt: str = "verbatim",
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
    ) -> None:
        self._engine = DeepSeekOCREngine(
            model_path=model_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
        )
        self._preprocessor = ImagePreprocessor(preprocess_config or PreprocessConfig())
        self._extractor = FieldExtractor()
        self._prompt_name = prompt
        self._initialized = False

    def initialize(self) -> None:
        """初始化流水线（加载模型）"""
        if self._initialized:
            return

        # 应用 transformers 兼容性补丁
        from core.compat import apply_patches
        apply_patches()

        self._engine.load()
        self._initialized = True

    def process(
        self,
        image_path: str,
        output_dir: str | None = None,
        preprocess: bool = True,
    ) -> PipelineResult:
        """处理单张图片

        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            preprocess: 是否预处理

        Returns:
            PipelineResult
        """
        import time as _time

        start = _time.time()

        try:
            # Step 1: 预处理
            actual_image = image_path
            if preprocess:
                actual_image = self._preprocessor.process(image_path, output_dir)
                logger.info("使用预处理图片: %s", actual_image)

            # Step 2: OCR 识别
            prompt = PROMPT_MAP.get(self._prompt_name, PROMPT_VERBATIM)
            raw_text = self._engine.recognize(actual_image, prompt)

            # Step 3: 去重
            clean_text = deduplicate_output(raw_text)

            # 调试：输出 OCR 原始文本前 500 字符
            logger.debug("OCR 原始输出 (前500字): %s", raw_text[:500])
            logger.debug("去重后文本 (前500字): %s", clean_text[:500])

            # Step 4: 字段提取
            card = self._extractor.extract(clean_text)
            card.source_image = str(image_path)

            elapsed = _time.time() - start
            logger.info("处理完成: %s (%.1fs)", Path(image_path).name, elapsed)

            return PipelineResult(
                card=card,
                preprocessed_path=actual_image if preprocess else None,
                elapsed_seconds=elapsed,
                success=True,
            )

        except Exception as e:
            elapsed = _time.time() - start
            logger.error("处理失败: %s - %s", image_path, e)
            return PipelineResult(
                card=HouseholdCard(source_image=str(image_path)),
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
            )

    @property
    def engine(self) -> OCREngine:
        """获取底层 OCR 引擎"""
        return self._engine

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def get_vram_usage_mb(self) -> float:
        """获取 GPU 显存占用"""
        return self._engine.get_vram_usage_mb()
