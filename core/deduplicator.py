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

    # 策略 2: 检测渐变式重复（每行略有变化但核心相同）
    result = _truncate_mutating_duplicates(lines, min_repeat)
    if result is not None:
        logger.debug("检测到渐变式重复，已截断")
        return result

    # 策略 3: 检测重复的段落模式（如 Markdown 表格行）
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

    # 策略 3: 检测语义相似的段落（OCR 常见的"换种说法重复"模式）
    return _deduplicate_similar_paragraphs(paragraphs, text)


def _truncate_mutating_duplicates(lines: list[str], min_repeat: int = 3) -> str | None:
    """检测渐变式重复：连续多行内容高度相似但每行略有变化

    典型模式（DeepSeek-OCR-2 退化循环）:
      - 何时由何地迁来本市：1952年由无籍迁入
      - 何时由何地迁来本市的：1952由无籍迁入
      - 何时由何种地迁来本市：1952由无籍迁入
      - 何时由何种地区迁来本市：1952由无籍迁入

    每行都略有不同，但 80%+ 内容相同。应只保留第一行。
    """
    if len(lines) < min_repeat:
        return None

    for i in range(len(lines)):
        base = lines[i].strip()
        if not base or len(base) < 4:
            continue

        # 计算后续连续行与 base 的相似度
        similar_count = 0
        for j in range(i, len(lines)):
            line = lines[j].strip()
            if not line:
                continue

            similarity = _line_similarity(base, line)
            if similarity >= 0.75:
                similar_count += 1
            else:
                break  # 不再相似，停止

        if similar_count >= min_repeat:
            # 保留到重复起点（含第一行），截断后续
            kept = lines[:i + 1]
            removed = len(lines) - len(kept)
            if removed > 0:
                logger.debug(
                    "渐变式重复检测: 第 %d 行起 %d 行重复 (相似度>75%%)",
                    i + 1, similar_count,
                )
                return '\n'.join(kept) + f'\n... (已截断 {removed} 行渐变重复)'

    return None


def _line_similarity(a: str, b: str) -> float:
    """计算两行的相似度（最长公共子序列比率）

    Returns:
        0.0~1.0, 1.0 表示完全相同
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # 最长公共子序列长度
    m, n = len(a), len(b)
    # 优化：短行直接用完整 DP，长行用截断
    if m > 200 or n > 200:
        # 截取前 200 字符比较
        a, b = a[:200], b[:200]
        m, n = len(a), len(b)

    # 2行滚动 DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    return (2.0 * lcs_len) / (m + n)


def _deduplicate_similar_paragraphs(paragraphs: list[str], original_text: str) -> str | None:
    """检测语义高度相似的段落（OCR 模型常见的重复描述模式）

    例如模型先输出"表格内容如下：..."，再输出"表格中的其他信息还包括：..."
    两段内容高度重叠，保留更早出现的段落（通常更准确）。
    """
    if len(paragraphs) < 2:
        return None

    # 提取每个段落的键值对内容（去除格式标记后比较）
    seen_keys: dict[str, int] = {}  # key → 首次出现的段落索引
    duplicate_para_indices: set[int] = set()

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para or len(para) < 10:
            continue

        # 提取段落中的字段名（**字段名**: 值 或 字段名: 值 格式）
        keys_in_para: set[str] = set()
        for line in para.split('\n'):
            line = line.strip().lstrip('-,').strip()
            # Markdown 加粗格式
            m = re.match(r'\*\*(.+?)\*\*\s*[:：]', line)
            if not m:
                # 纯文本格式
                m = re.match(r'([^:：]{2,20})\s*[:：]', line)
            if m:
                key = m.group(1).strip().strip('*').strip()
                if key and len(key) >= 2:
                    keys_in_para.add(key)

        if not keys_in_para:
            continue

        # 检查该段落的键是否与之前段落高度重叠
        for prev_idx, prev_keys in seen_keys.items():
            if not prev_keys:
                continue
            overlap = len(keys_in_para & prev_keys) / max(len(prev_keys), 1)
            if overlap >= 0.6:  # 60% 以上字段名重复视为语义重复
                duplicate_para_indices.add(i)
                break

        if i not in duplicate_para_indices:
            seen_keys[i] = keys_in_para

    if duplicate_para_indices:
        kept = [p for i, p in enumerate(paragraphs) if i not in duplicate_para_indices]
        result = '\n\n'.join(kept)
        if result.strip() != original_text.strip():
            logger.debug("检测到语义重复段落，已去除 %d 个段落", len(duplicate_para_indices))
            return result

    return None
