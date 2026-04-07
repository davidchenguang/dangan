"""50年代老旧户籍卡 OCR 识别系统 — 应用入口

运行方式:
    python main.py
"""

import logging
import sys

from PySide6.QtWidgets import QApplication


def setup_logging() -> None:
    """配置日志"""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 文件 handler — 覆盖写入
    file_handler = logging.FileHandler(
        "ocr_app.log", encoding="utf-8", mode="w",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 控制台 handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)


def main() -> int:
    """应用入口"""
    # UTF-8 兼容（Windows 必须在所有 IO 之前设置）
    if sys.platform == "win32":
        import os
        os.environ.setdefault("PYTHONUTF8", "1")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    # 单实例锁：防止多进程竞争 GPU 显存
    import msvcrt
    lock_path = os.path.join(os.environ.get("TEMP", "."), "dangan_ocr.lock")
    try:
        _lock_fd = open(lock_path, "w")
        msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        print("已有一个户籍卡 OCR 实例正在运行，不可重复启动。", file=sys.stderr)
        return 1

    setup_logging()
    logger = logging.getLogger("main")
    logger.info("启动户籍卡 OCR 识别系统")

    app = QApplication(sys.argv)
    app.setApplicationName("户籍卡 OCR 识别系统")
    app.setOrganizationName("dangan")

    # 加载样式表
    from pathlib import Path
    qss_path = Path(__file__).parent / "resources" / "styles.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    # 创建主窗口
    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
