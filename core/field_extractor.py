"""字段提取器 — 从 test_ocr.py:223-322 迁移并增强

三级降级策略:
1. JSON 解析 → HouseholdCard
2. Markdown 表格解析 → HouseholdCard
3. 描述性文本正则提取 → HouseholdCard（当前最常用）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import fields as dataclass_fields

from core.models import (
    ATTR_TO_LABEL,
    FIELD_TO_ATTR,
    HouseholdCard,
)

logger = logging.getLogger(__name__)


class FieldExtractor:
    """从 OCR 原始输出中提取结构化字段"""

    # 跳过的字段名黑名单（描述性/非字段内容）
    SKIP_KEYS = frozenset({
        '标题', '表格结构', '图片内容描述', '左侧部分', '右侧部分',
        '整体描述', '图片描述', '表格内容',
    })

    def extract(self, raw_text: str) -> HouseholdCard:
        """从 OCR 输出中提取结构化字段（三级降级）

        Args:
            raw_text: OCR 原始输出文本

        Returns:
            HouseholdCard 结构化结果
        """
        if not raw_text or not raw_text.strip():
            return HouseholdCard(raw_markdown=raw_text or "")

        # Level 1: JSON 解析
        card = self._parse_json(raw_text)
        if card:
            logger.info("JSON 解析成功")
            card.raw_markdown = raw_text
            return card

        # Level 2: Markdown 表格解析
        card = self._parse_markdown_table(raw_text)
        if card:
            logger.info("Markdown 表格解析成功")
            card.raw_markdown = raw_text
            return card

        # Level 3: 描述性文本正则提取
        fields_dict = self._extract_fields_from_text(raw_text)
        if fields_dict:
            card = self._dict_to_card(fields_dict)
            logger.info("描述性文本提取成功，提取 %d 个字段", len(fields_dict))
            card.raw_markdown = raw_text
            return card

        # 全部失败，保留原始文本
        logger.warning("所有提取策略均失败，保留原始文本")
        return HouseholdCard(raw_markdown=raw_text)

    # ── Level 1: JSON 解析 ──

    def _parse_json(self, text: str) -> HouseholdCard | None:
        """尝试从文本中提取 JSON 并转换为 HouseholdCard"""
        data = self._extract_json_object(text)
        if data is None:
            return None
        return self._dict_to_card(data)

    def _extract_json_object(self, text: str) -> dict | None:
        """从文本中提取 JSON 对象（多种格式尝试）"""
        # 直接解析
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Markdown 代码块中的 JSON
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # 裸 JSON 对象
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    # ── Level 2: Markdown 表格解析 ──

    def _parse_markdown_table(self, text: str) -> HouseholdCard | None:
        """解析 Markdown 表格格式"""
        fields_dict: dict[str, str] = {}
        in_table = False

        for line in text.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                # 过滤空元素和分隔行
                parts = [p for p in parts if p]
                if len(parts) >= 2:
                    # 跳过分隔行 (---|---)
                    if all(set(p) <= {'-', ':', ' '} for p in parts):
                        continue
                    key = parts[0]
                    value = parts[1]
                    if key and value and key not in self.SKIP_KEYS:
                        if key not in fields_dict:
                            fields_dict[key] = value

        if fields_dict:
            return self._dict_to_card(fields_dict)
        return None

    # ── Level 3: 描述性文本提取 ──

    def _extract_fields_from_text(self, text: str) -> dict[str, str]:
        """从 OCR 描述性输出中提取字段-值对

        支持以下格式:
        - Markdown: **字段名**: 值 / **字段名**：值
        - 纯文本: 字段名: 值 / 字段名：值
        - 列表项: - 字段名: 值
        - 表格行: 字段名 | 值
        """
        fields: dict[str, str] = {}

        for line in text.split('\n'):
            line = line.strip().lstrip(',-').strip()
            if not line:
                continue

            # 格式1: **字段名**: 值 (Markdown 加粗)
            m = re.match(r'\*\*(.+?)\*\*\s*[:：]\s*(.+)', line)
            if not m:
                # 格式2: 字段名: 值 / 字段名：值 (纯文本键值对)
                m = re.match(r'([^:：]{2,20})\s*[:：]\s*(.+)', line)
            if not m:
                # 格式3: 字段名 | 值 (表格行)
                m = re.match(r'([^|]{2,20})\|\s*(.+)', line)

            if not m:
                continue

            key = m.group(1).strip().strip('*').strip()
            value = m.group(2).strip()
            # 清理 value
            value = value.strip(',').strip('*').strip()
            # 过滤
            if not key or key in self.SKIP_KEYS:
                continue
            if not value or value in ('无', '-', '—'):
                continue
            # 保留首次出现的值（更可靠）
            if key not in fields:
                fields[key] = value

        return fields

    # ── 字典 → HouseholdCard 转换 ──

    def _dict_to_card(self, data: dict[str, str]) -> HouseholdCard:
        """将扁平字典转换为 HouseholdCard

        支持中文字段名和嵌套字段路径（如 "birth.date"）
        """
        card = HouseholdCard()

        for key, value in data.items():
            if not value:
                continue
            # 查找字段映射
            attr_path = FIELD_TO_ATTR.get(key)
            if attr_path is None:
                # 尝试模糊匹配
                attr_path = self._fuzzy_match_field(key)
                if attr_path is None:
                    continue

            self._set_card_field(card, attr_path, str(value))

        return card

    def _set_card_field(self, card: HouseholdCard, attr_path: str, value: str) -> None:
        """设置 HouseholdCard 的字段值（支持嵌套路径如 birth.date）"""
        parts = attr_path.split('.')
        obj = card

        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return

        final_attr = parts[-1]
        if hasattr(obj, final_attr):
            setattr(obj, final_attr, value)

    def _fuzzy_match_field(self, key: str) -> str | None:
        """模糊匹配字段名"""
        key_lower = key.lower().strip()
        for known_key, attr in FIELD_TO_ATTR.items():
            if key_lower in known_key.lower() or known_key.lower() in key_lower:
                return attr
        return None
