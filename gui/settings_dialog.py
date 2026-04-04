"""设置对话框 — 预处理参数配置、模型路径配置

修改后自动保存到 config.yaml。
"""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QDoubleSpinBox,
)

from core.config import AppConfig
from core.models import PreprocessConfig


class SettingsDialog(QDialog):
    """设置对话框

    使用方式:
        dialog = SettingsDialog(config, parent)
        if dialog.exec():
            config.save()
    """

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(500)

        self._config = config

        self._init_ui()
        self._load_values()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)

        self._model_path = QLineEdit()
        model_layout.addRow("模型路径:", self._model_path)

        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_model)
        model_layout.addRow("", browse_btn)

        self._prompt_combo = QComboBox()
        self._prompt_combo.addItems(["verbatim", "markdown", "json"])
        model_layout.addRow("默认 Prompt:", self._prompt_combo)

        layout.addWidget(model_group)

        # 预处理设置
        prep_group = QGroupBox("预处理设置")
        prep_layout = QFormLayout(prep_group)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.0, 4.0)
        self._scale_spin.setSingleStep(0.5)
        self._scale_spin.setValue(2.0)
        prep_layout.addRow("放大倍数:", self._scale_spin)

        self._clahe_spin = QDoubleSpinBox()
        self._clahe_spin.setRange(0.5, 10.0)
        self._clahe_spin.setSingleStep(0.5)
        self._clahe_spin.setValue(3.0)
        prep_layout.addRow("CLAHE 对比度:", self._clahe_spin)

        self._cb_enhance = QCheckBox("对比度增强 (CLAHE)")
        self._cb_enhance.setChecked(True)
        prep_layout.addRow("", self._cb_enhance)

        self._cb_auto_rotate = QCheckBox("倾斜校正")
        prep_layout.addRow("", self._cb_auto_rotate)

        self._cb_repair = QCheckBox("边缘修补")
        prep_layout.addRow("", self._cb_repair)

        self._cb_normalize = QCheckBox("背景归一化（去黄）")
        prep_layout.addRow("", self._cb_normalize)

        self._cb_remove_stamps = QCheckBox("印章去除")
        prep_layout.addRow("", self._cb_remove_stamps)

        self._cb_denoise = QCheckBox("去噪（默认关闭，可能破坏笔画）")
        prep_layout.addRow("", self._cb_denoise)

        self._cb_binarize = QCheckBox("二值化（默认关闭）")
        prep_layout.addRow("", self._cb_binarize)

        layout.addWidget(prep_group)

        # GUI 设置
        gui_group = QGroupBox("界面设置")
        gui_layout = QFormLayout(gui_group)

        self._thumbnail_spin = QSpinBox()
        self._thumbnail_spin.setRange(60, 300)
        self._thumbnail_spin.setValue(120)
        gui_layout.addRow("缩略图大小:", self._thumbnail_spin)

        layout.addWidget(gui_group)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_values(self) -> None:
        """从配置加载当前值"""
        cfg = self._config
        prep = cfg.preprocess

        self._model_path.setText(cfg.model_path)
        self._prompt_combo.setCurrentText(cfg.ocr_prompt)
        self._scale_spin.setValue(prep.scale)
        self._clahe_spin.setValue(prep.clahe_clip)
        self._cb_enhance.setChecked(prep.enhance_contrast)
        self._cb_auto_rotate.setChecked(prep.auto_rotate)
        self._cb_repair.setChecked(prep.repair_edges)
        self._cb_normalize.setChecked(prep.normalize_background)
        self._cb_remove_stamps.setChecked(prep.remove_stamps)
        self._cb_denoise.setChecked(prep.denoise)
        self._cb_binarize.setChecked(prep.binarize)
        self._thumbnail_spin.setValue(cfg.thumbnail_size)

    @Slot()
    def _browse_model(self) -> None:
        """浏览模型路径"""
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self._model_path.setText(dir_path)

    @Slot()
    def _on_accept(self) -> None:
        """保存设置"""
        cfg = self._config
        # 更新配置数据
        cfg._data.setdefault("model", {})["path"] = self._model_path.text()
        cfg._data.setdefault("ocr", {})["prompt"] = self._prompt_combo.currentText()
        cfg._data.setdefault("gui", {})["thumbnail_size"] = self._thumbnail_spin.value()

        # 预处理配置
        cfg._data["preprocess"] = {
            "scale": self._scale_spin.value(),
            "clahe_clip": self._clahe_spin.value(),
            "enhance_contrast": self._cb_enhance.isChecked(),
            "auto_rotate": self._cb_auto_rotate.isChecked(),
            "repair_edges": self._cb_repair.isChecked(),
            "normalize_background": self._cb_normalize.isChecked(),
            "remove_stamps": self._cb_remove_stamps.isChecked(),
            "denoise": self._cb_denoise.isChecked(),
            "binarize": self._cb_binarize.isChecked(),
        }

        # 重建预处理配置对象
        cfg._build_preprocess()
        cfg.save()

        self.accept()
