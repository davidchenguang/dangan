"""图像增强工具 — 为投票机制提供每轮不同的微小变换

通过轻微调整亮度、对比度和微量模糊，使同一张图片在不同轮次中
产生微小差异，配合 temperature > 0，让确定性模型产生不同输出。
"""

from __future__ import annotations

import cv2
import numpy as np


def augment_for_voting(image: np.ndarray, seed: int = 0) -> np.ndarray:
    """对图像施加微小随机变换（用于投票轮次间的输入差异化）

    变换范围（经调试不会破坏文字识别）:
    - 亮度: ±8 (0-255 范围)
    - 对比度: 0.95x ~ 1.05x
    - 微小高斯模糊: sigma 0.3 ~ 0.8

    Args:
        image: BGR 格式图像 (OpenCV)
        seed: 随机种子，不同值产生不同变换

    Returns:
        增强后的图像（相同尺寸）
    """
    rng = np.random.RandomState(seed)
    result = image.copy().astype(np.float32)

    # 1. 亮度调整: ±8
    brightness_delta = rng.uniform(-8, 8)
    result = result + brightness_delta

    # 2. 对比度调整: 0.95 ~ 1.05
    contrast_factor = rng.uniform(0.95, 1.05)
    mean = result.mean()
    result = (result - mean) * contrast_factor + mean

    # 3. 微小高斯模糊 (sigma 0.3 ~ 0.8)
    sigma = rng.uniform(0.3, 0.8)
    if sigma > 0.4:
        ksize = int(sigma * 6) | 1  # 确保 ksize 为奇数
        result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    return np.clip(result, 0, 255).astype(np.uint8)
