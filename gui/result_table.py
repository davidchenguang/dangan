"""识别结果可编辑表格 — 字段名 + 多成员列

支持多列（多成员）显示，每列对应一个 HouseholdCard。
点击编辑，修改后标记为"已校正"。
"""

from __future__ import annotations

import logging
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
)

from core.models import ATTR_TO_LABEL, FIELD_TO_ATTR, HouseholdCard

logger = logging.getLogger(__name__)


class ResultTable(QWidget):
    """识别结果面板

    显示 HouseholdCard 的所有字段，支持多列（多成员）和编辑。

    信号:
        field_edited: 字段被编辑 (col_index, field_attr, new_value)
    """

    field_edited = Signal(int, str, str)

    # 要显示的字段列表（按顺序）
    DISPLAY_FIELDS: list[tuple[str, str]] = [
        ("relation_to_household_head", "户主或与户主关系"),
        ("name", "姓名"),
        ("alias", "别名"),
        ("gender", "性别"),
        ("birth.date", "出生日期"),
        ("birth.address", "出生地址"),
        ("native_place", "籍贯"),
        ("ethnicity", "民族"),
        ("religion", "宗教信仰"),
        ("marital_status", "婚姻状况"),
        ("education", "文化程度"),
        ("occupation.occupation", "职业"),
        ("occupation.workplace", "服务处所"),
        ("other_residence", "本市其他住所"),
        ("citizen_cert.code_number", "公民证代号号码"),
        ("citizen_cert.issuing_authority", "签发机关"),
        ("citizen_cert.issue_date", "签发日期"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        self._title = QLabel("识别结果")
        self._title.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 4px; color: #333333;"
        )
        layout.addWidget(self._title)

        # 表格（列数动态调整）
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["字段名", "识别值"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self._table)

        # 原始文本降级视图
        self._raw_title = QLabel("原始识别文本 (结构化失败时的回退)")
        self._raw_title.setStyleSheet("font-weight: bold; color: #d32f2f; margin-top: 10px;")
        layout.addWidget(self._raw_title)

        self._raw_text_edit = QTextEdit()
        self._raw_text_edit.setReadOnly(True)
        self._raw_text_edit.setStyleSheet(
            "background-color: #fff3e0; color: #333; font-family: monospace;"
        )
        layout.addWidget(self._raw_text_edit)

        self._raw_title.hide()
        self._raw_text_edit.hide()

        self._cards: list[HouseholdCard] = []
        self._updating = False  # 防止编辑信号循环

    # ── 向后兼容接口 ──

    def display_card(self, card: HouseholdCard) -> None:
        """向后兼容：显示单个 HouseholdCard"""
        self.display_cards([card])

    def get_card(self) -> HouseholdCard:
        """向后兼容：获取第一个卡片"""
        cards = self.get_cards()
        return cards[0] if cards else HouseholdCard()

    # ── 主要接口 ──

    def display_cards(self, cards: list[HouseholdCard]) -> None:
        """显示多个 HouseholdCard（多列宽表模式）"""
        if not cards:
            self.clear()
            return

        self._updating = True
        self._cards = list(cards)

        num_members = len(cards)

        # 设置列数：1（字段名）+ N（每个成员）
        self._table.setColumnCount(1 + num_members)

        # 动态表头
        headers = ["字段名"]
        for i, card in enumerate(cards):
            relation = card.relation_to_household_head
            if relation:
                headers.append(relation)
            elif num_members == 1:
                headers.append("识别值")
            else:
                headers.append(f"成员{i + 1}")
        self._table.setHorizontalHeaderLabels(headers)

        # 设置列宽策略
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        for col in range(1, 1 + num_members):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch
            )

        row_count = len(self.DISPLAY_FIELDS)
        self._table.setRowCount(row_count)

        # 检查任一 card 是否含有有效字段
        has_values = any(
            any(self._get_field_value(card, attr_path) for attr_path, _ in self.DISPLAY_FIELDS)
            for card in cards
        )

        for row, (attr_path, label) in enumerate(self.DISPLAY_FIELDS):
            # 字段名列（不可编辑）
            key_item = QTableWidgetItem(label)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, key_item)

            # 各成员值列（可编辑）
            for col_idx, card in enumerate(cards):
                value = self._get_field_value(card, attr_path)
                value_item = QTableWidgetItem(value or "")
                self._table.setItem(row, col_idx + 1, value_item)

        # 标题提示
        if has_values:
            first_name = cards[0].name or "未识别"
            if num_members > 1:
                self._title.setText(f"识别结果 — {num_members} 位成员（首位: {first_name}）")
            else:
                self._title.setText(f"识别结果 — {first_name}")
            self._title.setStyleSheet(
                "font-weight: bold; font-size: 14px; padding: 4px; color: #333333;"
            )
        elif any(c.raw_markdown for c in cards):
            self._title.setText("识别结果 — 字段提取失败（原始文本如下）")
            self._title.setStyleSheet(
                "font-weight: bold; font-size: 14px; padding: 4px; color: #d32f2f;"
            )
            # 显示原始文本
            raw = next((c.raw_markdown for c in cards if c.raw_markdown), "")
            self._raw_title.show()
            self._raw_text_edit.show()
            self._raw_text_edit.setPlainText(raw)
        else:
            self._title.setText("识别结果 — 无输出")
            self._title.setStyleSheet(
                "font-weight: bold; font-size: 14px; padding: 4px; color: #e65100;"
            )

        self._updating = False

    def get_cards(self) -> list[HouseholdCard]:
        """从表格获取当前编辑后的所有 HouseholdCard"""
        if not self._cards:
            return [HouseholdCard()]

        # 深拷贝，避免 _set_field_value 污染 _cards 原始数据
        import copy
        cards = copy.deepcopy(self._cards)
        num_members = len(cards)

        for col_idx, card in enumerate(cards):
            for row, (attr_path, _label) in enumerate(self.DISPLAY_FIELDS):
                value_item = self._table.item(row, col_idx + 1)
                if value_item:
                    self._set_field_value(card, attr_path, value_item.text())
            card.reviewed = True

        return cards

    def clear(self) -> None:
        """清空表格"""
        self._updating = True
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["字段名", "识别值"])
        self._table.setRowCount(0)
        self._cards = []
        self._title.setText("识别结果")
        self._title.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 4px; color: #333333;"
        )
        if hasattr(self, "_raw_title"):
            self._raw_title.hide()
            self._raw_text_edit.hide()
            self._raw_text_edit.clear()
        self._updating = False

    def _get_field_value(self, card: HouseholdCard, attr_path: str) -> str:
        """从 HouseholdCard 获取嵌套字段值"""
        parts = attr_path.split(".")
        obj = card
        for part in parts:
            obj = getattr(obj, part, "")
            if obj is None:
                return ""
        return str(obj)

    def _set_field_value(self, card: HouseholdCard, attr_path: str, value: str) -> None:
        """设置 HouseholdCard 的嵌套字段值"""
        parts = attr_path.split(".")
        obj = card
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return
        setattr(obj, parts[-1], value)

    def _on_cell_changed(self, row: int, col: int) -> None:
        """单元格内容改变"""
        if self._updating or col < 1:
            return

        if row < len(self.DISPLAY_FIELDS):
            attr_path = self.DISPLAY_FIELDS[row][0]
            value_item = self._table.item(row, col)
            if value_item:
                card_idx = col - 1
                self.field_edited.emit(card_idx, attr_path, value_item.text())
