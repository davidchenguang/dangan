"""OCR 引擎 — 从 test_ocr.py:150-220 迁移并重构

抽象接口 + DeepSeek-OCR-2 实现。
使用前必须先调用 core.compat.apply_patches()。
"""

from __future__ import annotations

import logging
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch

from core.models import CARD_OCR_PROMPT, FieldRegion, PROMPT_VERBATIM

logger = logging.getLogger(__name__)


class OCREngine(ABC):
    """OCR 引擎抽象接口"""

    @abstractmethod
    def load(self) -> None:
        """加载模型（懒加载，首次调用时执行）"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        ...

    @abstractmethod
    def recognize(self, image_path: str, prompt: str) -> str:
        """识别图片中的文字

        Args:
            image_path: 图片文件路径
            prompt: OCR Prompt

        Returns:
            识别出的文本
        """
        ...

    @abstractmethod
    def get_vram_usage_mb(self) -> float:
        """获取当前 GPU 显存占用 (MB)"""
        ...


class DeepSeekOCREngine(OCREngine):
    """DeepSeek-OCR-2 封装

    关键约束:
    - infer() 硬编码参数: max_new_tokens=8192, no_repeat_ngram_size=35
    - eval_mode=True 必须为 True
    - 推理无法中断，GUI 需明确提示用户
    - 兼容性补丁必须在新版本 transformers 上执行
    """

    def __init__(
        self,
        model_path: str,
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
    ) -> None:
        self._model_path = model_path
        self._base_size = base_size
        self._image_size = image_size
        self._crop_mode = crop_mode
        self._tokenizer = None
        self._model = None
        self._load_time: float = 0.0

    def load(self) -> None:
        """加载模型"""
        if self._model is not None:
            return

        # 检查模型路径是否存在
        model_path = Path(self._model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self._model_path}")

        # 检查 CUDA 可用性
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用。请确保已安装 NVIDIA GPU 驱动和 CUDA toolkit。"
            )

        logger.info("正在加载模型: %s", self._model_path)
        start = time.time()

        from core.compat import inject_rotary_embeddings

        # 延迟导入，避免模块加载时触发 transformers
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self._model = self._model.eval().cuda().to(torch.bfloat16)

        # 注入 rotary_emb（兼容性补丁的补充步骤）
        inject_rotary_embeddings(self._model)

        self._load_time = time.time() - start
        vram = self.get_vram_usage_mb()
        logger.info(
            "模型加载完成，耗时 %.1fs，GPU 显存: %.0f MB",
            self._load_time, vram,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_time(self) -> float:
        return self._load_time

    def recognize(self, image_path: str, prompt: str) -> str:
        """执行 OCR 识别

        Args:
            image_path: 图片文件路径
            prompt: OCR Prompt

        Returns:
            识别出的文本
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        output_dir = str(Path("output"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("开始识别: %s", image_path)
        start = time.time()

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

        elapsed = time.time() - start

        if isinstance(result, dict):
            text = result.get("text", str(result))
        else:
            text = str(result)

        # 调试：记录输出长度和前 200 字符
        logger.info(
            "识别完成，耗时 %.1fs，输出类型=%s，长度=%d",
            elapsed, type(result).__name__, len(text),
        )
        if text:
            logger.debug("OCR 输出前 200 字: %s", text[:200])
        else:
            logger.warning("OCR 输出为空")

        return text

    def recognize_ndarray(self, image: np.ndarray, prompt: str) -> str:
        """识别 numpy 数组格式的图片

        Args:
            image: OpenCV 格式图片 (BGR)
            prompt: OCR Prompt

        Returns:
            识别出的文本
        """
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            try:
                return self.recognize(tmp.name, prompt)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    def get_vram_usage_mb(self) -> float:
        """获取当前 GPU 显存占用 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def unload(self) -> None:
        """卸载模型释放显存"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            logger.info("模型已卸载，GPU 显存已释放")
