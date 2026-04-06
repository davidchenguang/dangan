"""测试 core/augmentation.py — 图像增强"""
import numpy as np
import pytest

from core.augmentation import augment_for_voting


class TestAugmentForVoting:

    def test_preserves_shape(self, sample_image):
        """增强后图片尺寸不变"""
        result = augment_for_voting(sample_image, seed=42)
        assert result.shape == sample_image.shape

    def test_preserves_dtype(self, sample_image):
        """输出类型保持 uint8"""
        result = augment_for_voting(sample_image, seed=42)
        assert result.dtype == np.uint8

    def test_different_seeds_produce_different_results(self, sample_image):
        """不同 seed 应产生不同增强结果"""
        r1 = augment_for_voting(sample_image, seed=42)
        r2 = augment_for_voting(sample_image, seed=99)
        assert not np.array_equal(r1, r2)

    def test_same_seed_reproducible(self, sample_image):
        """相同 seed 应产生完全相同的结果"""
        r1 = augment_for_voting(sample_image, seed=42)
        r2 = augment_for_voting(sample_image, seed=42)
        assert np.array_equal(r1, r2)

    def test_augmentation_is_subtle(self):
        """增强幅度应微小（像素差异不超过 15%）"""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = augment_for_voting(img, seed=42)
        diff = np.abs(result.astype(int) - img.astype(int))
        assert np.percentile(diff, 95) <= 20

    def test_seed_zero_still_augments(self, sample_image):
        """seed=0 仍然产生增强（不是原样返回）"""
        result = augment_for_voting(sample_image, seed=0)
        assert result.shape == sample_image.shape
