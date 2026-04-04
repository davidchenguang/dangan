"""增强版去重引擎 — 去除模型输出中的重复循环内容

DeepSeek-OCR-2 模型在推理时可能出现重复输出模式，例如反复输出同一行内容。
本模块检测并截断这些重复模式，保留有效内容。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def deduplicate_output(text: str, min_repeat: int = 3) -> str:
    """去除模型输出中的重复循环内容

    检测连续重复的行模式并截断。例如模型可能反复输出：
    - 何时由本市迁来本市
    - 何时由本市迁来本市
    - ...

    Args:
        text: 原始输出文本
        min_repeat: 最小重复次数阈值 (默认3)

    Returns:
        去重后的文本
    """
    if not text:
        return text

    lines = text.split('\n')
    if len(lines) < min_repeat:
        return text

    # 策略 1: 检测连续完全相同的行
    result = _truncate_consecutive_duplicates(lines, min_repeat)
    if result is not None:
        logger.debug("检测到连续重复行，已截断")
        return result

    # 策略 2: 检测重复的段落模式（如 Markdown 表格行）
    result = _truncate_repeating_paragraphs(text, min_repeat)
    if result is not None:
        logger.debug("检测到重复段落模式，已截断")
        return result

    return text


def _truncate_consecutive_duplicates(lines: list[str], min_repeat: int) -> str | None:
    """检测并截断连续完全相同的行"""
    for i in range(len(lines)):
        pattern = lines[i].strip()
        if not pattern or len(pattern) < 2:
            continue
        repeat_count = 0
        for j in range(i, len(lines)):
            if lines[j].strip() == pattern:
                repeat_count += 1
            else:
                break
        if repeat_count >= min_repeat:
            kept = lines[:i + 1]
            return '\n'.join(kept) + f'\n... (已截断 {repeat_count - 1} 行重复内容)'

    return None


def _truncate_repeating_paragraphs(text: str, min_repeat: int) -> str | None:
    """检测重复的段落模式（如 Markdown 表格重复段）"""
    # 按空行分段
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) < min_repeat:
        return None

    # 检测连续相同的段落
    for i in range(len(paragraphs)):
        pattern = paragraphs[i].strip()
        if not pattern or len(pattern) < 10:
            continue
        repeat_count = 0
        for j in range(i, len(paragraphs)):
            if paragraphs[j].strip() == pattern:
                repeat_count += 1
            else:
                break
        if repeat_count >= min_repeat:
            kept = paragraphs[:i + 1]
            return '\n\n'.join(kept) + '\n\n... (已截断重复段落)'

    return None
