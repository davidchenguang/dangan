"""异步 OCR 任务 — QRunnable 实现

每张图片的 OCR 在独立 QRunnable 中执行，避免阻塞 UI 线程。
注意：模型推理本身无法中断，GUI 需明确提示用户。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

logger = logging.getLogger(__name__)


class OcrWorkerSignals(QObject):
    """OCR Worker 信号定义

    注意：Signals 必须定义在 QObject 子类上，不能直接挂在 QRunnable 上。
    """
    # 识别完成: (image_id, result_dict)
    result = Signal(str, object)
    # 识别失败: (image_id, error_message)
    error = Signal(str, str)
    # 进度更新: (image_id, percentage 0-100)
    progress = Signal(str, int)
    # 任务完成: (image_id)
    finished = Signal(str)


class OcrWorker(QRunnable):
    """单张图片 OCR 异步任务

    使用方式:
        worker = OcrWorker(fn=pipeline.process, image_id=item.id,
                           kwargs={"image_path": item.filepath})
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(self._on_finished)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        image_id: str,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.image_id = image_id
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.signals = OcrWorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """执行 OCR 任务"""
        try:
            self._safe_emit("progress", self.image_id, 10)
            result = self.fn(*self.args, **self.kwargs)
            self._safe_emit("progress", self.image_id, 90)
            self._safe_emit("result", self.image_id, result)
        except Exception as e:
            logger.error("OCR 任务失败 [%s]: %s", self.image_id, e)
            self._safe_emit("error", self.image_id, str(e))
        finally:
            self._safe_emit("finished", self.image_id)

    def _safe_emit(self, signal_name: str, *args: Any) -> None:
        """安全发射信号，防止信号源被提前删除时崩溃"""
        try:
            signal = getattr(self.signals, signal_name, None)
            if signal is not None:
                signal.emit(*args)
        except RuntimeError:
            logger.debug("信号 %s 发射失败（对象已销毁）: %s", signal_name, self.image_id)


class ModelLoadWorker(QRunnable):
    """模型加载异步任务"""

    def __init__(self, pipeline: Any) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.signals = ModelLoadSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            self._safe_emit("status", "正在加载模型，请耐心等待...")
            self.pipeline.initialize()
            vram = self.pipeline.get_vram_usage_mb()
            self._safe_emit("status", f"模型加载完成 (GPU 显存: {vram:.0f} MB)")
            self._safe_emit("loaded", True)
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            self._safe_emit("status", f"模型加载失败: {e}")
            self._safe_emit("loaded", False)

    def _safe_emit(self, signal_name: str, *args: Any) -> None:
        """安全发射信号，防止信号源被提前删除时崩溃"""
        try:
            signal = getattr(self.signals, signal_name, None)
            if signal is not None:
                signal.emit(*args)
        except RuntimeError:
            logger.debug("信号 %s 发射失败（对象已销毁）", signal_name)


class ModelLoadSignals(QObject):
    """模型加载信号"""
    status = Signal(str)
    loaded = Signal(bool)
