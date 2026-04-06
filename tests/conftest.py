"""共享测试 fixtures"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.models import (
    BirthInfo,
    ChangeRecord,
    HouseholdCard,
    OccupationInfo,
    PreprocessConfig,
)


@pytest.fixture
def sample_card():
    """标准测试用 HouseholdCard"""
    return HouseholdCard(
        name="张三",
        gender="男",
        birth=BirthInfo(date="1950年1月1日", address="北京市"),
        ethnicity="汉",
        education="小学",
        occupation=OccupationInfo(occupation="工人", workplace="北京钢厂"),
        native_place="河北",
        marital_status="已婚",
        source_image="test.jpg",
    )


@pytest.fixture
def sample_card_list():
    """多张卡片（用于投票测试）"""
    return [
        HouseholdCard(name="张三", gender="男", ethnicity="汉", education="小学"),
        HouseholdCard(name="张三", gender="男", ethnicity="汉", education="初中"),
        HouseholdCard(name="李四", gender="男", ethnicity="汉", education="小学"),
    ]


@pytest.fixture
def sample_image():
    """100x100 BGR 测试图像"""
    np.random.seed(42)
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_file(sample_image, tmp_path):
    """写入临时文件的测试图像"""
    import cv2
    path = str(tmp_path / "test_image.jpg")
    cv2.imwrite(path, sample_image)
    return path


@pytest.fixture
def default_preprocess_config():
    """默认预处理配置"""
    return PreprocessConfig()
