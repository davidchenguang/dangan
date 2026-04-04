"""图像预处理 — 从 test_ocr.py:98-147 迁移并增强

核心优化策略（经实验验证）：
1. 放大 2x — 使手写文字达到模型可识别的分辨率阈值
2. CLAHE 对比度增强 — 增强褪色手写文字与背景的对比度
3. 不做去噪 — 去噪反而破坏笔画细节（经测试验证）

注意：必须配合 crop_mode=True 使用，放大后的图片会触发 crop 机制
产生多个 patches，每个 patch 分辨率更高，手写文字更容易识别。
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as e:
    raise ImportError(f"图像处理依赖未安装: {e}，请运行: pip install opencv-python numpy")

from core.models import PreprocessConfig

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """户籍卡图像预处理器"""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def process(self, image_path: str, output_dir: str | None = None) -> str:
        """预处理户籍卡扫描件图片

        Args:
            image_path: 输入图片路径
            output_dir: 预处理后图片保存目录 (None则保存到 output/)

        Returns:
            预处理后图片路径

        Raises:
            FileNotFoundError: 图片文件不存在
            ValueError: 图片格式不支持或无法解码
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
            raise ValueError(f"不支持的图片格式: {path.suffix}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片（文件可能已损坏）: {image_path}")

        h, w = img.shape[:2]
        logger.info("原始尺寸: %dx%d", w, h)

        cfg = self.config

        # Step 1: 放大 — 关键步骤，使手写文字达到模型可识别阈值
        if cfg.scale > 1.0:
            new_w, new_h = int(w * cfg.scale), int(h * cfg.scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info("放大后: %dx%d", new_w, new_h)

        # Step 2: 倾斜校正（可选）
        if cfg.auto_rotate:
            img = self._auto_rotate(img)

        # Step 3: 边缘修补（可选）
        if cfg.repair_edges:
            img = self._repair_edges(img)

        # Step 4: 背景归一化（可选，去黄）
        if cfg.normalize_background:
            img = self._normalize_background(img)

        # Step 5: 印章去除（可选）
        if cfg.remove_stamps:
            img = self._remove_stamps(img)

        # Step 6: 去噪（可选，默认关闭）
        if cfg.denoise:
            img = self._denoise(img)

        # Step 7: 对比度增强 (CLAHE)
        if cfg.enhance_contrast:
            img = self._enhance_contrast(img, cfg.clahe_clip)

        # Step 8: 二值化（可选，默认关闭）
        if cfg.binarize:
            img = self._binarize(img)

        # 保存
        save_dir = output_dir or "./output"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        suffix = Path(image_path).suffix
        out_path = str(Path(save_dir) / f"{stem}_preprocessed{suffix}")
        cv2.imwrite(out_path, img)
        logger.info("预处理图片已保存: %s", out_path)
        return out_path

    def _enhance_contrast(self, img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
        """CLAHE 对比度增强"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """非局部均值去噪"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def _auto_rotate(self, img: np.ndarray) -> np.ndarray:
        """基于霍夫变换的倾斜校正"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

        if lines is None:
            return img

        # 取水平线计算平均角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            # 接近水平的线（theta 接近 0 或 pi）
            angle = np.degrees(theta) - 90
            if abs(angle) < 15:  # 倾斜不超过15度
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:  # 小于0.5度不校正
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.info("倾斜校正: %.2f 度", median_angle)
        return rotated

    def _repair_edges(self, img: np.ndarray) -> np.ndarray:
        """边缘修补 — 去除装订痕迹和破损边缘"""
        h, w = img.shape[:2]
        # 通过投影检测内容区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )

        # 水平投影
        h_proj = np.sum(thresh, axis=1)
        # 垂直投影
        v_proj = np.sum(thresh, axis=0)

        # 找到内容边界（跳过边缘空白区域）
        margin = 10
        top = max(0, np.argmax(h_proj > np.mean(h_proj) * 0.1) - margin)
        bottom = min(h, len(h_proj) - np.argmax(h_proj[::-1] > np.mean(h_proj) * 0.1) + margin)
        left = max(0, np.argmax(v_proj > np.mean(v_proj) * 0.1) - margin)
        right = min(w, len(v_proj) - np.argmax(v_proj[::-1] > np.mean(v_proj) * 0.1) + margin)

        return img[top:bottom, left:right]

    def _normalize_background(self, img: np.ndarray) -> np.ndarray:
        """背景归一化 — 去除黄色底色"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # 形态学闭运算估计背景
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
        background = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)

        # 除法归一化
        normalized = cv2.divide(l_channel, background, scale=255)
        lab = cv2.merge([normalized, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _remove_stamps(self, img: np.ndarray) -> np.ndarray:
        """印章去除 — HSV 色彩空间过滤红色区域"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 红色在 HSV 中分布在两个区间
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 用邻域像素填充（inpainting）
        result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return result

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """二值化 — 默认关闭"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
