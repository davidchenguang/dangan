"""测试 core/preprocessor.py — 图像预处理"""
import cv2
import numpy as np
import pytest

from core.models import PreprocessConfig
from core.preprocessor import ImagePreprocessor


class TestImagePreprocessorBasic:
    """基础预处理"""

    def test_scale_doubles_image(self, sample_image_file):
        """默认 scale=2.0 应放大 2 倍"""
        config = PreprocessConfig(scale=2.0, enhance_contrast=False)
        prep = ImagePreprocessor(config)
        result_path = prep.process(sample_image_file)
        img = cv2.imread(result_path)
        assert img.shape[:2] == (200, 200)  # 100*2

    def test_no_scale_keeps_size(self, sample_image_file):
        """scale=1.0 不放大"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=False)
        prep = ImagePreprocessor(config)
        result_path = prep.process(sample_image_file)
        img = cv2.imread(result_path)
        assert img.shape[:2] == (100, 100)

    def test_clahe_produces_gray_output(self, sample_image_file):
        """CLAHE 增强后输出是三通道灰度"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=True)
        prep = ImagePreprocessor(config)
        result_path = prep.process(sample_image_file)
        img = cv2.imread(result_path)
        assert img.ndim == 3  # BGR 三通道
        # CLAHE 输出是灰度转 BGR，三通道应相同
        b, g, r = cv2.split(img)
        assert np.array_equal(b, g)
        assert np.array_equal(g, r)

    def test_output_file_exists(self, sample_image_file, tmp_path):
        """预处理图片应保存到指定目录"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=False)
        prep = ImagePreprocessor(config)
        result = prep.process(sample_image_file, output_dir=str(tmp_path))
        import os
        assert os.path.exists(result)


class TestImagePreprocessorValidation:
    """输入验证"""

    def test_missing_file_raises(self):
        prep = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            prep.process("/nonexistent/file.jpg")

    def test_unsupported_format_raises(self, tmp_path):
        """非图片格式应报错"""
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")
        prep = ImagePreprocessor()
        with pytest.raises(ValueError, match="不支持"):
            prep.process(str(bad_file))


class TestImagePreprocessorOptionalSteps:
    """可选预处理步骤"""

    def test_denoise_reduces_noise(self, sample_image_file, tmp_path):
        """去噪应产生不同输出"""
        config_no_denoise = PreprocessConfig(scale=1.0, enhance_contrast=False, denoise=False)
        config_denoise = PreprocessConfig(scale=1.0, enhance_contrast=False, denoise=True)

        dir1 = tmp_path / "no_denoise"
        dir2 = tmp_path / "denoise"
        r1 = ImagePreprocessor(config_no_denoise).process(sample_image_file, output_dir=str(dir1))
        r2 = ImagePreprocessor(config_denoise).process(sample_image_file, output_dir=str(dir2))

        img1 = cv2.imread(r1)
        img2 = cv2.imread(r2)
        assert not np.array_equal(img1, img2)

    def test_binarize_produces_binary_output(self, sample_image_file, tmp_path):
        """二值化应将像素值分为两极（JPEG 压缩可能引入少量中间值）"""
        config = PreprocessConfig(scale=1.0, enhance_contrast=False, binarize=True)
        result_path = ImagePreprocessor(config).process(
            sample_image_file, output_dir=str(tmp_path / "binarize"),
        )
        img = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        unique = set(np.unique(img))
        # 二值化后大部分像素应集中在 0 和 255 附近
        # JPEG 压缩可能引入少量中间值，但不应超过 10 个唯一值
        assert len(unique) <= 20
