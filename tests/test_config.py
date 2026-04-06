"""测试 core/config.py — 配置管理"""

import os
import tempfile

import pytest
import yaml

from core.config import AppConfig
from core.models import PreprocessConfig


@pytest.fixture(autouse=True)
def reset_config():
    """每个测试前重置单例"""
    AppConfig.reset()
    yield
    AppConfig.reset()


class TestAppConfig:
    """AppConfig 测试"""

    def test_default_values(self):
        """默认配置文件应正确加载"""
        config = AppConfig()
        config.load()

        # 模型路径不应为空
        assert config.model_path
        assert "deepseek-ocr-2" in config.model_path.lower()

        # 预处理默认值
        prep = config.preprocess
        assert isinstance(prep, PreprocessConfig)
        assert prep.scale == 2.0
        assert prep.clahe_clip == 3.0
        assert prep.denoise is False
        assert prep.enhance_contrast is True

    def test_ocr_defaults(self):
        config = AppConfig()
        config.load()

        assert config.ocr_base_size == 1024
        assert config.ocr_image_size == 768
        assert config.ocr_crop_mode is True
        assert config.ocr_prompt == "verbatim"
        assert config.thumbnail_size == 120

    def test_export_defaults(self):
        config = AppConfig()
        config.load()

        assert config.export_dir == "./output"

    def test_missing_config_file(self):
        """配置文件不存在时使用默认值"""
        config = AppConfig()
        config.load("/nonexistent/path/config.yaml")
        assert config.model_path  # 仍有默认值

    def test_save_and_reload(self):
        """保存后重新加载应保持一致"""
        config = AppConfig()
        config.load()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_config.yaml")
            config.save(save_path)

            # 验证文件存在
            assert os.path.exists(save_path)

            # 重新加载
            AppConfig.reset()
            config2 = AppConfig()
            config2.load(save_path)

            assert config2.model_path == config.model_path
            assert config2.ocr_prompt == config.ocr_prompt

    def test_custom_config(self):
        """自定义配置文件"""
        custom_config = {
            "model": {"path": "/custom/model/path"},
            "ocr": {"prompt": "json"},
            "gui": {"thumbnail_size": 200},
            "preprocess": {"scale": 3.0, "denoise": True},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "custom.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(custom_config, f, allow_unicode=True)

            config = AppConfig()
            config.load(config_path)

            assert config.model_path == "/custom/model/path"
            assert config.ocr_prompt == "json"  # 自定义配置指定了 json
            assert config.thumbnail_size == 200
            assert config.preprocess.scale == 3.0
            assert config.preprocess.denoise is True

    def test_singleton(self):
        """单例模式验证"""
        config1 = AppConfig()
        config1.load()
        config2 = AppConfig()
        assert config1 is config2

    def test_reset(self):
        """重置后应产生新实例"""
        config1 = AppConfig()
        config1.load()
        AppConfig.reset()
        config2 = AppConfig()
        config2.load()
        assert config1 is not config2

    def test_invalid_yaml(self):
        """无效 YAML 应优雅降级"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "bad.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("invalid: yaml: content:\n  - [broken")

            config = AppConfig()
            config.load(config_path)  # 不应崩溃
            # 应有默认值
            assert config.ocr_prompt == "structured"
