"""缩略图列表面板 — 图片缩略图列表

支持多选、拖拽导入、显示处理状态图标。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QContextMenuEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMenu,
)

from core.models import ProcessingStatus

logger = logging.getLogger(__name__)

# 支持的图片格式
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})


class ThumbnailPanel(QListWidget):
    """缩略图列表面板

    信号:
        image_selected: 选中图片改变 (image_id, filepath)
        images_imported: 新图片导入 (list of (image_id, filepath))
    """

    image_selected = Signal(str, str)  # image_id, filepath
    images_imported = Signal(list)      # [(image_id, filepath), ...]

    def __init__(self, thumbnail_size: int = 120, parent=None) -> None:
        super().__init__(parent)

        self._thumbnail_size = thumbnail_size
        self._items: dict[str, dict] = {}  # image_id → {item, filepath, status}

        # 视图模式
        self.setViewMode(self.ViewMode.IconMode)
        self.setIconSize(QSize(thumbnail_size, thumbnail_size))
        self.setResizeMode(self.ResizeMode.Adjust)
        self.setSpacing(4)
        self.setMovement(self.Movement.Static)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)

        # 上下文菜单
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

        # 信号
        self.currentItemChanged.connect(self._on_current_changed)

    def add_images(self, filepaths: list[str]) -> list[str]:
        """添加图片到列表

        Returns:
            新添加的 image_id 列表
        """
        new_ids: list[str] = []
        new_items: list[tuple[str, str]] = []

        for filepath in filepaths:
            path = Path(filepath)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if not path.exists():
                continue

            image_id = str(uuid.uuid4())

            # 创建缩略图
            pixmap = QPixmap(filepath)
            if pixmap.isNull():
                continue

            thumbnail = pixmap.scaled(
                self._thumbnail_size,
                self._thumbnail_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            item = QListWidgetItem(QIcon(thumbnail), path.name)
            item.setData(256, image_id)      # Qt.ItemDataRole.UserRole
            item.setData(257, filepath)      # Qt.ItemDataRole.UserRole + 1
            item.setToolTip(filepath)
            item.setStatusTip("待处理")

            self.addItem(item)
            self._items[image_id] = {
                "item": item,
                "filepath": filepath,
                "status": ProcessingStatus.PENDING,
            }

            new_ids.append(image_id)
            new_items.append((image_id, filepath))

        if new_items:
            self.images_imported.emit(new_items)

        return new_ids

    def update_status(self, image_id: str, status: ProcessingStatus) -> None:
        """更新图片处理状态"""
        info = self._items.get(image_id)
        if info is None:
            return

        info["status"] = status
        item = info["item"]

        status_text = {
            ProcessingStatus.PENDING: "待处理",
            ProcessingStatus.PROCESSING: "处理中...",
            ProcessingStatus.DONE: "已完成",
            ProcessingStatus.FAILED: "失败",
        }.get(status, "")

        item.setStatusTip(status_text)

        # 通过文本后缀显示状态
        name = Path(info["filepath"]).name
        suffix_map = {
            ProcessingStatus.PENDING: name,
            ProcessingStatus.PROCESSING: f"[...] {name}",
            ProcessingStatus.DONE: f"[OK] {name}",
            ProcessingStatus.FAILED: f"[X] {name}",
        }
        item.setText(suffix_map.get(status, name))

    def get_filepath(self, image_id: str) -> str | None:
        """根据 image_id 获取文件路径"""
        info = self._items.get(image_id)
        return info["filepath"] if info else None

    def get_status(self, image_id: str) -> ProcessingStatus:
        """根据 image_id 获取处理状态"""
        info = self._items.get(image_id)
        return info["status"] if info else ProcessingStatus.PENDING

    def get_all_image_ids(self) -> list[str]:
        """获取所有图片 ID"""
        return list(self._items.keys())

    def get_selected_image_ids(self) -> list[str]:
        """获取选中的图片 ID"""
        return [
            item.data(256)
            for item in self.selectedItems()
        ]

    def get_current_image_id(self) -> str | None:
        """获取当前选中的图片 ID"""
        item = self.currentItem()
        return item.data(256) if item else None

    def remove_image(self, image_id: str) -> None:
        """移除图片"""
        info = self._items.pop(image_id, None)
        if info:
            row = self.row(info["item"])
            self.takeItem(row)

    def clear_all(self) -> None:
        """清空所有图片"""
        self.clear()
        self._items.clear()

    def _on_current_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        """选中项改变"""
        if current:
            image_id = current.data(256)
            filepath = current.data(257)
            if image_id and filepath:
                self.image_selected.emit(image_id, filepath)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """右键菜单"""
        menu = QMenu(self)
        menu.addAction("移除选中", self._remove_selected)
        menu.addAction("清空列表", self.clear_all)
        menu.exec(event.globalPos())

    def _remove_selected(self) -> None:
        """移除选中的图片"""
        for item in self.selectedItems():
            image_id = item.data(256)
            if image_id:
                self.remove_image(image_id)
