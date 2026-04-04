"""识别结果可编辑表格 — 字段名 + 识别值两列

点击编辑，修改后标记为"已校正"。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
    QLabel,
)

from core.models import ATTR_TO_LABEL, FIELD_TO_ATTR, HouseholdCard

logger = logging.getLogger(__name__)


class ResultTable(QWidget):
    """识别结果面板

    显示 HouseholdCard 的所有字段，支持编辑。

    信号:
        field_edited: 字段被编辑 (field_attr, new_value)
    """

    field_edited = Signal(str, str)

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

        # 表格
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["字段名", "识别值"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)

        self._table.cellChanged.connect(self._on_cell_changed)

        layout.addWidget(self._table)

        self._card: HouseholdCard | None = None
        self._updating = False  # 防止编辑信号循环

    def display_card(self, card: HouseholdCard) -> None:
        """显示 HouseholdCard 的识别结果"""
        self._updating = True
        self._card = card

        # 检查是否有有效字段值
        has_values = any(
            self._get_field_value(card, attr_path)
            for attr_path, _label in self.DISPLAY_FIELDS
        )

        row_count = len(self.DISPLAY_FIELDS)

        # 如果所有字段为空但有 raw_markdown，额外显示一行
        show_raw = not has_values and card.raw_markdown
        if show_raw:
            row_count += 1

        self._table.setRowCount(row_count)

        for row, (attr_path, label) in enumerate(self.DISPLAY_FIELDS):
            # 字段名列（不可编辑）
            key_item = QTableWidgetItem(label)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, key_item)

            # 值列（可编辑）
            value = self._get_field_value(card, attr_path)
            value_item = QTableWidgetItem(value or "")
            self._table.setItem(row, 1, value_item)

        # 显示原始 OCR 文本
        if show_raw:
            raw_row = len(self.DISPLAY_FIELDS)
            key_item = QTableWidgetItem("原始OCR文本")
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            key_item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(raw_row, 0, key_item)

            value_item = QTableWidgetItem(card.raw_markdown[:500])
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item.setForeground(Qt.GlobalColor.darkRed)
            self._table.setItem(raw_row, 1, value_item)

        # 标题提示
        if has_values:
            self._title.setText(f"识别结果 — {card.name or '未识别'}")
        elif card.raw_markdown:
            self._title.setText("识别结果 — 字段提取失败（原始文本如下）")
            self._title.setStyleSheet(
                "font-weight: bold; font-size: 14px; padding: 4px; color: #d32f2f;"
            )
        else:
            self._title.setText("识别结果 — 无输出")
            self._title.setStyleSheet(
                "font-weight: bold; font-size: 14px; padding: 4px; color: #e65100;"
            )

        self._updating = False

    def get_card(self) -> HouseholdCard:
        """从表格获取当前编辑后的 HouseholdCard"""
        if self._card is None:
            return HouseholdCard()

        card = self._card
        for row, (attr_path, _label) in enumerate(self.DISPLAY_FIELDS):
            value_item = self._table.item(row, 1)
            if value_item:
                self._set_field_value(card, attr_path, value_item.text())

        card.reviewed = True
        return card

    def clear(self) -> None:
        """清空表格"""
        self._updating = True
        self._table.setRowCount(0)
        self._card = None
        self._title.setText("识别结果")
        self._title.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 4px; color: #333333;"
        )
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
        if self._updating or col != 1:
            return

        if row < len(self.DISPLAY_FIELDS):
            attr_path = self.DISPLAY_FIELDS[row][0]
            value_item = self._table.item(row, 1)
            if value_item:
                self.field_edited.emit(attr_path, value_item.text())
