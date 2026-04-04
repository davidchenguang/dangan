"""导出对话框 — 多格式选择导出

支持同时导出多种格式，选择导出范围。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)


class ExportDialog(QDialog):
    """导出设置对话框

    使用方式:
        dialog = ExportDialog(cards, parent)
        if dialog.exec():
            filepaths = dialog.get_export_paths()
    """

    def __init__(self, total_count: int, recognized_count: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出识别结果")
        self.setMinimumWidth(400)

        self._export_paths: list[str] = []
        self._total_count = total_count
        self._recognized_count = recognized_count

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 格式选择
        format_group = QGroupBox("导出格式")
        format_layout = QVBoxLayout(format_group)

        self._cb_xlsx = QCheckBox("Excel (.xlsx)")
        self._cb_xlsx.setChecked(True)
        self._cb_csv = QCheckBox("CSV (.csv)")
        self._cb_json = QCheckBox("JSON (.json)")
        self._cb_pdf = QCheckBox("PDF (.pdf)")
        self._cb_docx = QCheckBox("Word (.docx)")

        format_layout.addWidget(self._cb_xlsx)
        format_layout.addWidget(self._cb_csv)
        format_layout.addWidget(self._cb_json)
        format_layout.addWidget(self._cb_pdf)
        format_layout.addWidget(self._cb_docx)

        layout.addWidget(format_group)

        # 导出范围
        range_group = QGroupBox("导出范围")
        range_layout = QVBoxLayout(range_group)

        self._rb_all = QRadioButton(f"全部 ({self._total_count} 张)")
        self._rb_recognized = QRadioButton(f"仅已识别 ({self._recognized_count} 张)")
        self._rb_recognized.setChecked(True)

        range_layout.addWidget(self._rb_all)
        range_layout.addWidget(self._rb_recognized)

        layout.addWidget(range_group)

        # 输出目录
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("输出目录:"))

        self._dir_edit = QLineEdit("./output")
        dir_layout.addWidget(self._dir_edit)

        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(browse_btn)

        layout.addLayout(dir_layout)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @Slot()
    def _browse_dir(self) -> None:
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self._dir_edit.setText(dir_path)

    @Slot()
    def _on_accept(self) -> None:
        """确认导出"""
        output_dir = self._dir_edit.text()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._export_paths = []
        formats = [
            (self._cb_xlsx, "results.xlsx"),
            (self._cb_csv, "results.csv"),
            (self._cb_json, "results.json"),
            (self._cb_pdf, "results.pdf"),
            (self._cb_docx, "results.docx"),
        ]

        for checkbox, filename in formats:
            if checkbox.isChecked():
                self._export_paths.append(str(Path(output_dir) / filename))

        if not self._export_paths:
            return  # 至少选择一种格式

        self.accept()

    def get_export_paths(self) -> list[str]:
        """获取导出文件路径列表"""
        return self._export_paths

    def is_export_all(self) -> bool:
        """是否导出全部（否则仅已识别）"""
        return self._rb_all.isChecked()
