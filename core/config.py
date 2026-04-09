"""配置管理 — 加载 config.yaml 并提供类型安全的访问接口"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from core.models import PreprocessConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class AppConfig:
    """应用配置（单例模式）

    使用方式:
        config = AppConfig()
        config.load()  # 或传入自定义路径
        model_path = config.model_path
        preprocess_cfg = config.preprocess
    """

    _instance: AppConfig | None = None

    def __new__(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self) -> None:
        if self._loaded:
            return
        self._data: dict[str, Any] = {}
        self._preprocess: PreprocessConfig | None = None
        self._loaded = False

    def load(self, path: str | Path | None = None) -> None:
        """加载配置文件

        Args:
            path: 配置文件路径，None 则使用默认 config.yaml
        """
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._data = yaml.safe_load(f) or {}
                logger.info("已加载配置: %s", config_path)
            except (yaml.YAMLError, OSError) as e:
                logger.error("配置文件解析失败: %s — %s，使用默认值", config_path, e)
                self._data = {}
        else:
            logger.warning("配置文件不存在: %s，使用默认值", config_path)
            self._data = {}

        self._build_preprocess()
        self._loaded = True

    def _build_preprocess(self) -> None:
        """从配置字典构建 PreprocessConfig"""
        p = self._data.get("preprocess", {})
        self._preprocess = PreprocessConfig(
            scale=p.get("scale", 2.0),
            clahe_clip=p.get("clahe_clip", 3.0),
            auto_rotate=p.get("auto_rotate", False),
            repair_edges=p.get("repair_edges", False),
            normalize_background=p.get("normalize_background", False),
            remove_stamps=p.get("remove_stamps", False),
            denoise=p.get("denoise", False),
            enhance_contrast=p.get("enhance_contrast", True),
            binarize=p.get("binarize", False),
        )

    # ── 属性访问 ──

    @property
    def model_path(self) -> str:
        """模型路径"""
        import os
        from pathlib import Path
        path = self._data.get("model", {}).get("path")
        if not path:
            path = os.environ.get(
                "DANGAN_MODEL_PATH", 
                str(_DEFAULT_CONFIG_PATH.parent.parent / "models" / "deepseek-ocr-2")
            )
        # 支持相对路径，相对于项目根目录
        p = Path(path)
        if not p.is_absolute():
            p = _DEFAULT_CONFIG_PATH.parent.parent / p
        return str(p)

    @property
    def model_device(self) -> str:
        device = self._data.get("model", {}).get("device", "auto")
        if device == "auto" or not device:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @property
    def model_dtype(self) -> str:
        dtype = self._data.get("model", {}).get("dtype", "auto")
        if dtype == "auto" or not dtype:
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return "bfloat16"
                return "float16"
            except ImportError:
                return "float32"
        return dtype

    @property
    def preprocess(self) -> PreprocessConfig:
        """预处理配置"""
        if self._preprocess is None:
            self._build_preprocess()
        return self._preprocess  # type: ignore[return-value]

    @property
    def ocr_base_size(self) -> int:
        return self._data.get("ocr", {}).get("base_size", 1024)

    @property
    def ocr_image_size(self) -> int:
        return self._data.get("ocr", {}).get("image_size", 768)

    @property
    def ocr_crop_mode(self) -> bool:
        return self._data.get("ocr", {}).get("crop_mode", True)

    @property
    def ocr_prompt(self) -> str:
        return self._data.get("ocr", {}).get("prompt", "multi_column")

    @property
    def ocr_voting_rounds(self) -> int:
        """OCR 多轮投票次数（1=单轮不投票）"""
        return int(self._data.get("ocr", {}).get("voting_rounds", 1))

    @property
    def export_dir(self) -> str:
        return self._data.get("export", {}).get("default_dir", "./output")

    @property
    def thumbnail_size(self) -> int:
        return self._data.get("gui", {}).get("thumbnail_size", 120)

    def save(self, path: str | Path | None = None) -> None:
        """保存当前配置到文件"""
        save_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self._data, f, allow_unicode=True, default_flow_style=False)
            logger.info("配置已保存: %s", save_path)
        except OSError as e:
            logger.error("配置保存失败: %s — %s", save_path, e)

    @classmethod
    def reset(cls) -> None:
        """重置单例（主要用于测试）"""
        cls._instance = None
