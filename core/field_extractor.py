"""字段提取器 — 从 test_ocr.py:223-322 迁移并增强

四级降级策略:
1. JSON 解析 → HouseholdCard
2. Markdown 表格解析 → HouseholdCard
3. HTML 表格解析 → HouseholdCard
4. 描述性文本正则提取 → HouseholdCard（当前最常用）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import fields as dataclass_fields

from core.models import (
    ATTR_TO_LABEL,
    FIELD_TO_ATTR,
    ChangeRecord,
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
        """从 OCR 输出中提取结构化字段（四级降级）

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

        # Level 3: HTML 表格解析
        card = self._parse_html_table(raw_text)
        if card:
            logger.info("HTML 表格解析成功")
            card.raw_markdown = raw_text
            return card

        # Level 4: 描述性文本正则提取
        fields_dict = self._extract_fields_from_text(raw_text)
        if fields_dict:
            card = self._dict_to_card(fields_dict)
            logger.info("描述性文本提取成功，提取 %d 个字段", len(fields_dict))
            card.raw_markdown = raw_text
            return card

        # 全部失败，保留原始文本
        logger.warning("所有提取策略均失败，保留原始文本")
        return HouseholdCard(raw_markdown=raw_text)

    def extract_multi(self, raw_text: str) -> list[HouseholdCard]:
        """从 OCR 输出中提取多个 HouseholdCard（支持多列宽表）

        当 OCR 输出包含户籍卡多列（如户主+妻+子）的宽表时，
        自动按列拆分生成对应数量的 HouseholdCard。

        Args:
            raw_text: OCR 原始输出文本

        Returns:
            HouseholdCard 列表，多列时多个，单列/非表格时一个
        """
        if not raw_text or not raw_text.strip():
            return [HouseholdCard(raw_markdown=raw_text or "")]

        # 尝试多列 Markdown 宽表解析
        cards = self._parse_multi_column_table(raw_text)
        if cards:
            for c in cards:
                c.raw_markdown = raw_text
            logger.info("多列宽表解析成功，共 %d 列", len(cards))
            return cards

        # 尝试从描述性文本中拆分多成员
        multi_dicts = self._extract_multi_from_text(raw_text)
        if len(multi_dicts) > 1:
            cards = [self._dict_to_card(d) for d in multi_dicts]
            for c in cards:
                c.raw_markdown = raw_text
            logger.info("多成员拆分成功，共 %d 位成员", len(cards))
            return cards

        # 回退：单列常规提取
        card = self.extract(raw_text)
        return [card]

    # ── 多列宽表解析 ──

    def _parse_multi_column_table(self, text: str) -> list[HouseholdCard] | None:
        """解析多列 Markdown 宽表

        表格格式示例 (3列):
            | 字段名  | 户主   | 妻     |
            |---------|--------|--------|
            | 姓名    | 王盛祥 | 马妇女 |
            | 性别    | 男     | 女     |

        首列为字段名，其余各列对应一个成员。
        表头第一行的非字段列标题将用作 relation_to_household_head。

        Returns:
            list[HouseholdCard] 或 None（不是宽表时）
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        table_lines = [l for l in lines if '|' in l]

        if len(table_lines) < 2:
            return None

        # 解析所有行
        rows: list[list[str]] = []
        for line in table_lines:
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]
            if not parts:
                continue
            # 跳过分隔行
            if all(set(p) <= {'-', ':', ' '} for p in parts):
                continue
            rows.append(parts)

        if not rows:
            return None

        # 第一行为表头：[字段名列, 成员1标题, 成员2标题, ...]
        header = rows[0]
        num_cols = len(header) - 1  # 去掉字段名列

        if num_cols < 1:
            return None

        # 单列情况交给普通解析（avoid double-parsing）
        # 但如果列标题不是通用"值"/"内容"，说明是显式多成员表
        member_headers = header[1:]

        # 构建每个成员的字段字典
        member_dicts: list[dict[str, str]] = [{} for _ in range(num_cols)]

        # 将成员标题作为 relation_to_household_head 预填
        for i, h in enumerate(member_headers):
            if h and h not in ("值", "内容", "识别值", "value"):
                member_dicts[i]["户主或与户主关系"] = h

        # 遍历数据行，填充字段值
        for row in rows[1:]:
            if not row:
                continue
            field_name = row[0]
            if field_name in self.SKIP_KEYS:
                continue
            for col_idx in range(num_cols):
                value_idx = col_idx + 1
                if value_idx < len(row):
                    value = row[value_idx].strip()
                    if value and value not in ('-', '—', '无'):
                        if field_name not in member_dicts[col_idx]:
                            member_dicts[col_idx][field_name] = value

        # 检查解析是否有意义（至少有一列有字段命中）
        known_hits = sum(
            1 for d in member_dicts
            for k in d if k in FIELD_TO_ATTR or self._fuzzy_match_field(k) is not None
        )
        if known_hits == 0:
            return None

        # 转换为 HouseholdCard
        cards = [self._dict_to_card(d) for d in member_dicts]
        return cards if cards else None

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
        """解析 Markdown 表格格式

        支持两种模式:
        1. 两列表格: | 字段名 | 值 |（每行一个字段）
        2. 宽表交替: | 字段1 | 值1 | 字段2 | 值2 | ...（所有字段在一行）
        """
        fields_dict: dict[str, str] = {}
        all_cells_by_row: list[list[str]] = []

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
                    # 模式1: 两列表格 — parts[0] 为 key, parts[1] 为 value
                    key = parts[0]
                    value = parts[1]
                    if key and value and key not in self.SKIP_KEYS:
                        if key not in fields_dict:
                            fields_dict[key] = value
                    # 收集所有单元格用于模式2
                    all_cells_by_row.append(parts)

        # 模式2: 宽表扫描 — 遍历所有单元格，遇到已知字段名则取下一个单元格为值
        # 仅在模式1提取较少字段时启用（避免覆盖已有结果）
        if len(fields_dict) < 3:
            for row in all_cells_by_row:
                i = 0
                while i < len(row) - 1:
                    key = row[i]
                    value = row[i + 1]
                    is_field = key in FIELD_TO_ATTR or self._fuzzy_match_field(key) is not None
                    if is_field and key not in self.SKIP_KEYS:
                        if key not in fields_dict and value:
                            value_is_field = value in FIELD_TO_ATTR or self._fuzzy_match_field(value) is not None
                            if not value_is_field:
                                fields_dict[key] = value
                                i += 2
                                continue
                    i += 1

        if fields_dict:
            return self._dict_to_card(fields_dict)
        return None

    # ── Level 3: HTML 表格解析 ──

    def _parse_html_table(self, text: str) -> HouseholdCard | None:
        """解析 HTML 表格格式（<table><tr><td>...）

        处理两种常见模式:
        1. 交替单元格: <td>字段名</td><td>值</td><td>字段名</td><td>值</td>
        2. 表头/数据行: 第一行为字段名，第二行为值
        """
        if '<td>' not in text and '<th>' not in text:
            return None

        # 提取所有行
        rows: list[list[str]] = []
        for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL | re.IGNORECASE):
            row_html = tr_match.group(1)
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
            # 清理单元格内容（移除内嵌 HTML 标签）
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if cells:
                rows.append(cells)

        if not rows:
            return None

        fields_dict: dict[str, str] = {}

        if len(rows) >= 2:
            # 模式2: 表头行 + 数据行
            headers = rows[0]
            for data_row in rows[1:]:
                for i, header in enumerate(headers):
                    if i < len(data_row):
                        value = data_row[i].strip()
                        if header and value and header not in self.SKIP_KEYS:
                            if header not in fields_dict:
                                fields_dict[header] = value

        # 模式1: 交替单元格（单行或补充多行）
        for row in rows:
            i = 0
            while i < len(row) - 1:
                key = row[i]
                value = row[i + 1]
                is_field = key in FIELD_TO_ATTR or self._fuzzy_match_field(key) is not None
                if is_field and key not in self.SKIP_KEYS:
                    if key not in fields_dict and value:
                        value_is_field = value in FIELD_TO_ATTR or self._fuzzy_match_field(value) is not None
                        if not value_is_field:
                            fields_dict[key] = value
                            i += 2
                            continue
                i += 1

        if fields_dict:
            return self._dict_to_card(fields_dict)
        return None

    # ── 多成员描述性文本拆分 ──

    # 成员边界标记字段：当这些字段重复出现时，表示新成员开始
    _MEMBER_BOUNDARY_FIELDS = frozenset({"姓名", "户主或与户主关系"})

    def _extract_multi_from_text(self, text: str) -> list[dict[str, str]]:
        """从描述性文本中提取多成员字段

        策略：收集所有「字段: 值」对，当关键字段（如「姓名」）重复出现时，
        自动拆分为新成员。

        Returns:
            字段字典列表，每个字典对应一位成员；单成员时返回 [{}]
        """
        # Step 1: 收集所有字段-值对（保留重复，有序）
        all_pairs: list[tuple[str, str]] = []

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
            value = m.group(2).strip().strip(',').strip('*').strip()

            if not key or key in self.SKIP_KEYS:
                continue
            if not value or value in ('无', '-', '—'):
                continue

            # 只保留已知字段
            is_known = key in FIELD_TO_ATTR or self._fuzzy_match_field(key) is not None
            if is_known:
                all_pairs.append((key, value))

        if not all_pairs:
            return [{}]

        # Step 2: 按成员边界拆分
        members: list[dict[str, str]] = [{}]

        for key, value in all_pairs:
            # 姓名重复 → 新成员
            if key in self._MEMBER_BOUNDARY_FIELDS and key in members[-1]:
                members.append({})

            # 非边界字段重复 → 也视为新成员（如第二个"性别"说明有第二人）
            elif key in members[-1] and key not in self._MEMBER_BOUNDARY_FIELDS:
                # 只有当前成员已有该字段时才开新成员
                # 避免因 OCR 重复输出同一人而误拆
                pass  # 覆盖当前成员的同字段值

            members[-1][key] = value

        # 过滤空成员
        members = [m for m in members if m]

        return members if members else [{}]

    # ── Level 4: 描述性文本提取 ──

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

    # 列表类型字段集合
    _LIST_FIELDS = frozenset({"migration_in", "migration_local", "cancellation", "changes"})

    def _set_card_field(self, card: HouseholdCard, attr_path: str, value: str) -> None:
        """设置 HouseholdCard 的字段值（支持嵌套路径和列表类型字段）"""
        # 列表类型字段：解析为 ChangeRecord 列表
        if attr_path in self._LIST_FIELDS:
            records = self._parse_change_records(value)
            setattr(card, attr_path, records)
            return

        parts = attr_path.split('.')
        obj = card

        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return

        final_attr = parts[-1]
        if hasattr(obj, final_attr):
            setattr(obj, final_attr, value)

    @staticmethod
    def _parse_change_records(value: str) -> list[ChangeRecord]:
        """将文本解析为 ChangeRecord 列表

        支持格式:
        - "1953年3月 由北京迁入" → [ChangeRecord(date="1953年3月", content="由北京迁入")]
        - "1953年3月由北京迁入；1960年8月由上海迁入" → 2 条记录
        """
        if not value or not value.strip():
            return []

        records: list[ChangeRecord] = []

        # 按分号/换行分割多条记录
        parts = re.split(r'[；;\n]', value)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 尝试分离日期和内容
            m = re.match(r'(\d{2,4}年\s*\d{1,2}月?\s*\d{0,2}日?)\s*(.*)', part)
            if m:
                records.append(ChangeRecord(date=m.group(1).strip(), content=m.group(2).strip()))
            else:
                records.append(ChangeRecord(content=part))

        return records

    def _fuzzy_match_field(self, key: str) -> str | None:
        """模糊匹配字段名"""
        key_lower = key.lower().strip()
        for known_key, attr in FIELD_TO_ATTR.items():
            if key_lower in known_key.lower() or known_key.lower() in key_lower:
                return attr
        return None
