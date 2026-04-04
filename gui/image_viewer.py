"""图片查看器 — QGraphicsView + 字段区域高亮

支持缩放、平移、字段区域叠加显示。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen, QPainter, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QAbstractItemView,
)

logger = logging.getLogger(__name__)


class FieldRegionItem(QGraphicsRectItem):
    """字段区域高亮矩形

    悬停时显示字段名和识别文本。
    颜色按置信度区分:
    - 绿色: > 0.9
    - 黄色: 0.7 ~ 0.9
    - 红色: < 0.7
    - 蓝色: 无置信度信息（默认）
    """

    def __init__(
        self,
        rect: QRectF,
        field_name: str = "",
        field_value: str = "",
        confidence: float = 0.0,
    ) -> None:
        super().__init__(rect)

        self.field_name = field_name
        self.field_value = field_value
        self.confidence = confidence

        # 根据置信度选择颜色
        if confidence > 0.9:
            color = QColor(0, 200, 0, 60)
            border_color = QColor(0, 200, 0, 180)
        elif confidence > 0.7:
            color = QColor(255, 200, 0, 60)
            border_color = QColor(255, 200, 0, 180)
        elif confidence > 0:
            color = QColor(255, 0, 0, 60)
            border_color = QColor(255, 0, 0, 180)
        else:
            color = QColor(0, 120, 255, 40)
            border_color = QColor(0, 120, 255, 120)

        self.setBrush(color)
        self.setPen(QPen(border_color, 2))
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        # 标签
        self._label: QGraphicsTextItem | None = None

    def hoverEnterEvent(self, event) -> None:
        """鼠标悬停时显示标签"""
        if self._label is None:
            text = f"<b>{self.field_name}</b>"
            if self.field_value:
                text += f": {self.field_value}"
            if self.confidence > 0:
                text += f" <i>({self.confidence:.0%})</i>"

            self._label = QGraphicsTextItem(self)
            self._label.setHtml(text)
            self._label.setDefaultTextColor(QColor(255, 255, 255))
            self._label.setPos(self.rect().x(), self.rect().y() - 25)

            # 背景
            from PySide6.QtGui import QTextDocument
            doc = self._label.document()
            doc.setTextWidth(300)

        self._label.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """鼠标离开时隐藏标签"""
        if self._label:
            self._label.setVisible(False)
        super().hoverLeaveEvent(event)


class ImageViewer(QGraphicsView):
    """自定义图片查看器

    支持:
    - 鼠标滚轮缩放
    - 鼠标拖拽平移
    - 字段区域高亮叠加
    - 自适应窗口大小
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 渲染优化
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # 拖拽模式
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._zoom_level = 0
        self._min_zoom = -10
        self._max_zoom = 20
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._field_items: list[FieldRegionItem] = []

    def load_image(self, filepath: str) -> None:
        """加载图片"""
        from PySide6.QtGui import QPixmap

        self.clear_all()

        pixmap = QPixmap(filepath)
        if pixmap.isNull():
            logger.warning("无法加载图片: %s", filepath)
            return

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = 0

    def clear_all(self) -> None:
        """清空所有内容"""
        self._scene.clear()
        self._pixmap_item = None
        self._field_items.clear()
        self._zoom_level = 0

    def add_field_region(
        self,
        x: float, y: float, w: float, h: float,
        field_name: str = "",
        field_value: str = "",
        confidence: float = 0.0,
    ) -> None:
        """添加字段区域高亮"""
        item = FieldRegionItem(
            QRectF(x, y, w, h),
            field_name=field_name,
            field_value=field_value,
            confidence=confidence,
        )
        self._field_items.append(item)
        self._scene.addItem(item)

    def clear_field_regions(self) -> None:
        """清除所有字段区域高亮"""
        for item in self._field_items:
            self._scene.removeItem(item)
        self._field_items.clear()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """鼠标滚轮缩放"""
        delta = event.angleDelta().y()
        if delta > 0:
            if self._zoom_level < self._max_zoom:
                self.scale(1.15, 1.15)
                self._zoom_level += 1
        else:
            if self._zoom_level > self._min_zoom:
                self.scale(1 / 1.15, 1 / 1.15)
                self._zoom_level -= 1

    def fit_to_window(self) -> None:
        """自适应窗口大小"""
        if self._scene.sceneRect().isValid():
            self.fitInView(
                self._scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self._zoom_level = 0

    def reset_zoom(self) -> None:
        """重置缩放"""
        self.resetTransform()
        self._zoom_level = 0
        self.fit_to_window()
