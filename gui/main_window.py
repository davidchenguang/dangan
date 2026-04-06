"""主窗口 — QMainWindow 三栏布局

布局:
├── menuBar: 文件 | 编辑 | 处理 | 导出 | 设置
├── toolBar: [导入] [识别] [全部识别] [停止] | [导出] [设置]
├── centralWidget: QSplitter (水平三栏)
│   ├── 左栏(200px): ThumbnailPanel
│   ├── 中栏(弹性): ImageViewer + 进度条
│   └── 右栏(300px): ResultTable
├── statusBar: 模型状态 | GPU显存 | 进度
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThreadPool, Qt, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.config import AppConfig
from core.models import HouseholdCard, ProcessingStatus
from core.pipeline import OcrPipeline, PipelineResult
from gui.image_viewer import ImageViewer
from gui.result_table import ResultTable
from gui.thumbnail_panel import ThumbnailPanel
from gui.workers import ModelLoadWorker, OcrWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("50年代户籍卡 OCR 识别系统")
        self.setMinimumSize(1200, 700)

        # 配置
        self._config = AppConfig()
        self._config.load()

        # 数据存储: image_id → HouseholdCard
        self._results: dict[str, HouseholdCard] = {}

        # 流水线（懒加载）
        self._pipeline: OcrPipeline | None = None

        # 任务计数器
        self._completed_count: int = 0
        self._total_count: int = 0

        # 活跃 Worker 引用（防止 GC 回收导致信号源丢失）
        self._active_workers: list[OcrWorker] = []

        # UI
        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._init_connections()

        # 延迟加载模型
        self._load_model_async()

    def _init_ui(self) -> None:
        """初始化 UI 布局"""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # 三栏分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左栏: 缩略图面板
        self._thumbnail_panel = ThumbnailPanel(
            thumbnail_size=self._config.thumbnail_size,
        )
        self._thumbnail_panel.setMinimumWidth(180)
        self._thumbnail_panel.setMaximumWidth(300)
        splitter.addWidget(self._thumbnail_panel)

        # 中栏: 图片查看器 + 进度条
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        self._image_viewer = ImageViewer()
        self._image_viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        middle_layout.addWidget(self._image_viewer)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        middle_layout.addWidget(self._progress_bar)

        splitter.addWidget(middle)

        # 右栏: 结果面板
        self._result_table = ResultTable()
        self._result_table.setMinimumWidth(250)
        self._result_table.setMaximumWidth(450)
        splitter.addWidget(self._result_table)

        # 分割比例
        splitter.setStretchFactor(0, 0)  # 左栏不拉伸
        splitter.setStretchFactor(1, 1)  # 中栏弹性
        splitter.setStretchFactor(2, 0)  # 右栏不拉伸
        splitter.setSizes([200, 600, 300])

        layout.addWidget(splitter)

    def _init_menu(self) -> None:
        """初始化菜单栏"""
        menu_bar = self.menuBar()

        # 文件
        file_menu = menu_bar.addMenu("文件(&F)")
        file_menu.addAction("导入图片(&I)", self._import_images, QKeySequence("Ctrl+I"))
        file_menu.addAction("导入文件夹(&D)", self._import_folder)
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, QKeySequence("Ctrl+Q"))

        # 编辑
        edit_menu = menu_bar.addMenu("编辑(&E)")
        edit_menu.addAction("清空列表", self._clear_all)

        # 处理
        process_menu = menu_bar.addMenu("处理(&P)")
        process_menu.addAction("识别当前(&R)", self._recognize_current, QKeySequence("F5"))
        process_menu.addAction("识别选中(&S)", self._recognize_selected)
        process_menu.addAction("识别全部(&A)", self._recognize_all, QKeySequence("Ctrl+F5"))

        # 导出
        export_menu = menu_bar.addMenu("导出(&X)")
        export_menu.addAction("导出当前(&C)", self._export_current)
        export_menu.addAction("导出全部(&E)", self._export_all)

        # 设置
        settings_menu = menu_bar.addMenu("设置(&T)")
        settings_menu.addAction("首选项(&P)", self._show_settings)

    def _init_toolbar(self) -> None:
        """初始化工具栏"""
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)

        toolbar.addAction("导入", self._import_images)
        toolbar.addSeparator()

        self._btn_recognize = toolbar.addAction("识别", self._recognize_current)
        self._btn_recognize_all = toolbar.addAction("全部识别", self._recognize_all)
        self._btn_stop = toolbar.addAction("停止", self._stop_processing)
        self._btn_stop.setEnabled(False)

        toolbar.addSeparator()
        toolbar.addAction("导出", self._export_all)
        toolbar.addAction("设置", self._show_settings)

    def _init_statusbar(self) -> None:
        """初始化状态栏"""
        status_bar = self.statusBar()

        self._status_model = QLabel("模型: 未加载")
        self._status_vram = QLabel("GPU 显存: --")
        self._status_progress = QLabel("就绪")

        status_bar.addWidget(self._status_model)
        status_bar.addWidget(self._status_vram, 1)
        status_bar.addPermanentWidget(self._status_progress)

    def _init_connections(self) -> None:
        """初始化信号连接"""
        self._thumbnail_panel.image_selected.connect(self._on_image_selected)

    # ── 模型加载 ──

    def _load_model_async(self) -> None:
        """异步加载模型"""
        self._status_model.setText("模型: 正在加载...")
        self._set_controls_enabled(False)

        config = self._config
        self._pipeline = OcrPipeline(
            model_path=config.model_path,
            preprocess_config=config.preprocess,
            prompt=config.ocr_prompt,
            base_size=config.ocr_base_size,
            image_size=config.ocr_image_size,
            crop_mode=config.ocr_crop_mode,
            voting_rounds=config.ocr_voting_rounds,
        )

        self._model_worker = ModelLoadWorker(self._pipeline)
        self._model_worker.signals.status.connect(self._on_model_status)
        self._model_worker.signals.loaded.connect(self._on_model_loaded)
        QThreadPool.globalInstance().start(self._model_worker)

    @Slot(str)
    def _on_model_status(self, msg: str) -> None:
        self._status_model.setText(f"模型: {msg}")

    @Slot(bool)
    def _on_model_loaded(self, success: bool) -> None:
        if success and self._pipeline:
            vram = self._pipeline.get_vram_usage_mb()
            self._status_model.setText("模型: 已就绪")
            self._status_vram.setText(f"GPU 显存: {vram:.0f} MB")
            self._set_controls_enabled(True)
        else:
            self._status_model.setText("模型: 加载失败")
            QMessageBox.critical(
                self, "模型加载失败",
                "无法加载 DeepSeek-OCR-2 模型。\n"
                "请检查模型路径和 CUDA 环境。",
            )

    # ── 图片操作 ──

    @Slot()
    def _import_images(self) -> None:
        """导入图片"""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;所有文件 (*.*)",
        )
        if filepaths:
            self._thumbnail_panel.add_images(filepaths)

    @Slot()
    def _import_folder(self) -> None:
        """导入文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"):
                paths.extend(str(p) for p in Path(folder).glob(ext))
            if paths:
                self._thumbnail_panel.add_images(sorted(paths))
            else:
                QMessageBox.information(self, "提示", "该文件夹中没有图片文件。")

    @Slot(str, str)
    def _on_image_selected(self, image_id: str, filepath: str) -> None:
        """选中图片改变"""
        self._image_viewer.load_image(filepath)

        # 如果已有识别结果，显示
        card = self._results.get(image_id)
        if card:
            self._result_table.display_card(card)
        else:
            self._result_table.clear()

    @Slot()
    def _clear_all(self) -> None:
        """清空所有"""
        self._thumbnail_panel.clear_all()
        self._results.clear()
        self._image_viewer.clear_all()
        self._result_table.clear()

    # ── OCR 处理 ──

    @Slot()
    def _recognize_current(self) -> None:
        """识别当前选中图片"""
        image_id = self._thumbnail_panel.get_current_image_id()
        if not image_id:
            QMessageBox.information(self, "提示", "请先选择一张图片。")
            return
        self._submit_ocr_tasks([image_id])

    @Slot()
    def _recognize_selected(self) -> None:
        """识别选中图片"""
        ids = self._thumbnail_panel.get_selected_image_ids()
        if not ids:
            QMessageBox.information(self, "提示", "请先选择图片。")
            return
        self._submit_ocr_tasks(ids)

    @Slot()
    def _recognize_all(self) -> None:
        """识别全部待处理图片"""
        all_ids = self._thumbnail_panel.get_all_image_ids()
        pending_ids = [
            id_ for id_ in all_ids
            if self._thumbnail_panel.get_status(id_) == ProcessingStatus.PENDING
        ]
        if not pending_ids:
            QMessageBox.information(self, "提示", "没有待处理的图片。")
            return
        self._submit_ocr_tasks(pending_ids)

    def _submit_ocr_tasks(self, image_ids: list[str]) -> None:
        """提交 OCR 任务"""
        if not self._pipeline or not self._pipeline.is_initialized:
            QMessageBox.warning(self, "提示", "模型尚未加载完成，请稍候。")
            return

        self._set_controls_enabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, len(image_ids))
        self._progress_bar.setValue(0)
        self._completed_count = 0
        self._total_count = len(image_ids)

        for image_id in image_ids:
            filepath = self._thumbnail_panel.get_filepath(image_id)
            if not filepath:
                continue

            self._thumbnail_panel.update_status(image_id, ProcessingStatus.PROCESSING)

            worker = OcrWorker(
                fn=self._pipeline.process,
                image_id=image_id,
                kwargs={"image_path": filepath},
            )
            worker.signals.result.connect(self._on_ocr_result)
            worker.signals.error.connect(self._on_ocr_error)
            worker.signals.finished.connect(self._on_ocr_finished)
            self._active_workers.append(worker)
            QThreadPool.globalInstance().start(worker)

    @Slot(str, object)
    def _on_ocr_result(self, image_id: str, result: PipelineResult) -> None:
        """OCR 识别完成"""
        if result.success:
            self._results[image_id] = result.card
            self._thumbnail_panel.update_status(image_id, ProcessingStatus.DONE)

            # 如果是当前选中的图片，更新显示
            current_id = self._thumbnail_panel.get_current_image_id()
            if image_id == current_id:
                self._result_table.display_card(result.card)
        else:
            self._thumbnail_panel.update_status(image_id, ProcessingStatus.FAILED)
            logger.error("OCR 失败 [%s]: %s", image_id, result.error)

    @Slot(str, str)
    def _on_ocr_error(self, image_id: str, error: str) -> None:
        """OCR 识别失败"""
        self._thumbnail_panel.update_status(image_id, ProcessingStatus.FAILED)
        self._status_progress.setText(f"错误: {error[:50]}")

    @Slot(str)
    def _on_ocr_finished(self, image_id: str) -> None:
        """单个 OCR 任务完成"""
        # 清理已完成的 worker 引用
        self._active_workers = [
            w for w in self._active_workers if w.image_id != image_id
        ]

        self._completed_count += 1
        total = self._total_count
        self._progress_bar.setValue(self._completed_count)
        self._status_progress.setText(f"进度: {self._completed_count}/{total}")

        if self._completed_count >= total:
            self._active_workers.clear()
            self._set_controls_enabled(True)
            self._progress_bar.setVisible(False)
            vram = self._pipeline.get_vram_usage_mb() if self._pipeline else 0
            self._status_progress.setText(
                f"完成 {total} 张图片识别，GPU 显存: {vram:.0f} MB"
            )

    @Slot()
    def _stop_processing(self) -> None:
        """停止处理（注意：正在推理的任务无法中断）"""
        # QThreadPool 无法取消正在运行的任务，只能等待
        QMessageBox.information(
            self, "提示",
            "正在推理中的任务无法中断。\n"
            "队列中的任务将在当前任务完成后停止。",
        )

    # ── 导出 ──

    @Slot()
    def _export_current(self) -> None:
        """导出当前结果"""
        image_id = self._thumbnail_panel.get_current_image_id()
        if not image_id or image_id not in self._results:
            QMessageBox.information(self, "提示", "请先识别一张图片。")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出识别结果", "output/result.xlsx",
            "Excel (*.xlsx);;CSV (*.csv);;JSON (*.json);;PDF (*.pdf);;Word (*.docx)",
        )
        if filepath:
            from core.exporter import ResultExporter
            exporter = ResultExporter()
            card = self._result_table.get_card()
            exporter.export([card], filepath)
            QMessageBox.information(self, "成功", f"已导出到:\n{filepath}")

    @Slot()
    def _export_all(self) -> None:
        """导出全部结果"""
        # 收集已识别的结果
        recognized_cards = [
            self._results[id_]
            for id_ in self._thumbnail_panel.get_all_image_ids()
            if id_ in self._results
        ]
        total = len(self._thumbnail_panel.get_all_image_ids())
        recognized = len(recognized_cards)

        if not recognized:
            QMessageBox.information(self, "提示", "没有已识别的结果可导出。")
            return

        # 弹出导出对话框
        from gui.export_dialog import ExportDialog
        dialog = ExportDialog(total, recognized, self)
        if dialog.exec():
            filepaths = dialog.get_export_paths()
            if not filepaths:
                return

            cards = recognized_cards

            from core.exporter import ResultExporter
            exporter = ResultExporter()
            exported = exporter.export_multi(cards, filepaths)

            if exported:
                msg = "已导出:\n" + "\n".join(f"  • {p}" for p in exported)
                QMessageBox.information(self, "导出成功", msg)

    # ── 设置 ──

    @Slot()
    def _show_settings(self) -> None:
        """显示设置对话框"""
        from gui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self._config, self)
        if dialog.exec():
            # 配置已更新，可能需要重新加载模型
            self._config.save()
            self._status_progress.setText("设置已保存，部分设置需重启生效。")

    # ── 辅助 ──

    def _set_controls_enabled(self, enabled: bool) -> None:
        """启用/禁用处理相关控件"""
        self._btn_recognize.setEnabled(enabled)
        self._btn_recognize_all.setEnabled(enabled)
        self._btn_stop.setEnabled(not enabled)

    def closeEvent(self, event) -> None:
        """窗口关闭时清理资源"""
        if self._pipeline and self._pipeline.is_initialized:
            self._pipeline.engine.unload()
        event.accept()
