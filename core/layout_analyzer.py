"""版面分析 — 定位表格字段区域（占位实现）

当前 DeepSeek-OCR-2 使用整卡识别策略，不依赖版面分析。
本模块为未来扩展预留，当需要字段级补识别时实现。

设计文档中规划的版面分析功能:
- 基于 RapidLayout 检测表格结构
- 定位各字段区域 (FieldRegion)
- 用于低置信度字段的单独补识别
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.models import FieldRegion

logger = logging.getLogger(__name__)


@dataclass
class LayoutResult:
    """版面分析结果"""
    regions: dict[str, FieldRegion] = field(default_factory=dict)


class LayoutAnalyzer:
    """版面分析器（占位实现）

    当前 DeepSeek-OCR-2 的整卡识别效果足够好，
    暂不需要细粒度的版面分析。未来可扩展:
    - RapidLayout 检测表格结构
    - OpenCV 轮廓检测定位字段区域
    - 深度学习版面分析模型
    """

    def analyze(self, image_path: str) -> LayoutResult:
        """分析图片版面

        Args:
            image_path: 图片文件路径

        Returns:
            LayoutResult（当前返回空结果）
        """
        logger.debug("版面分析（占位实现）: %s", image_path)
        return LayoutResult(regions={})
