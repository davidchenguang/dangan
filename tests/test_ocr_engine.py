"""测试 core/ocr_engine.py — OCR 引擎接口"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestDeepSeekOCREngineInit:
    """初始化与属性"""

    def test_default_params(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake/model")
        assert engine._model_path == "/fake/model"
        assert engine._base_size == 1024
        assert engine._image_size == 768
        assert engine.is_loaded() is False

    def test_custom_params(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(
            model_path="/fake",
            base_size=512,
            image_size=384,
            crop_mode=False,
        )
        assert engine._base_size == 512
        assert engine._crop_mode is False


class TestDeepSeekOCREngineRecognize:
    """recognize() 方法"""

    def _make_loaded_engine(self, return_value="test output"):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._model.infer.return_value = return_value
        return engine

    def test_basic_recognize(self):
        engine = self._make_loaded_engine()
        result = engine.recognize("fake.jpg", "prompt")
        assert result == "test output"
        engine._model.infer.assert_called_once()

    def test_dict_result_extracts_text(self):
        engine = self._make_loaded_engine(return_value={"text": "hello"})
        result = engine.recognize("fake.jpg", "prompt")
        assert result == "hello"

    def test_unloaded_raises(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        with pytest.raises(RuntimeError, match="模型未加载"):
            engine.recognize("fake.jpg", "prompt")

    def test_temperature_zero_no_patch(self):
        """temperature=0.0 不应 monkey-patch generate"""
        engine = self._make_loaded_engine()
        original_generate = engine._model.generate
        engine.recognize("fake.jpg", "prompt", temperature=0.0)
        # generate 应未被替换
        assert engine._model.generate == original_generate

    def test_temperature_positive_patches_generate(self):
        """temperature > 0 应临时替换 generate 并恢复"""
        engine = self._make_loaded_engine()
        original_generate = engine._model.generate
        engine.recognize("fake.jpg", "prompt", temperature=0.5)
        # generate 应被恢复
        assert engine._model.generate == original_generate

    def test_temperature_passed_to_infer(self):
        """temperature 应影响 generate 调用参数"""
        engine = self._make_loaded_engine()
        engine.recognize("fake.jpg", "prompt", temperature=0.3)
        engine._model.infer.assert_called_once()


class TestDeepSeekOCREngineUnload:
    """unload() 方法"""

    def test_unload_clears_references(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine.unload()
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine.is_loaded() is False


class TestDeepSeekOCREngineVRam:
    """VRAM 使用量"""

    def test_vram_without_cuda(self):
        from core.ocr_engine import DeepSeekOCREngine
        engine = DeepSeekOCREngine(model_path="/fake")
        with patch("core.ocr_engine.torch.cuda.is_available", return_value=False):
            assert engine.get_vram_usage_mb() == 0.0
class TestDeepSeekOCREngineExtractResult:
    """_extract_result() 清洗逻辑测试"""

    def test_basic_cleaning(self):
        from core.ocr_engine import DeepSeekOCREngine
        # 测试同时包含 <image> 和 <|...|>
        text = "<image>\n<|ocr_text|>识别内容<|endoftext|>"
        cleaned = DeepSeekOCREngine._extract_result(text, "")
        assert cleaned == "识别内容"

    def test_dict_cleaning(self):
        from core.ocr_engine import DeepSeekOCREngine
        result = {"text": "  <image>  姓名：张三  "}
        cleaned = DeepSeekOCREngine._extract_result(result, "")
        assert cleaned == "姓名：张三"

    def test_stdout_fallback_cleaning(self):
        from core.ocr_engine import DeepSeekOCREngine
        # stdout 包含调试行和 <image>
        stdout_text = "image: path/to/img\nPATCHES: 1\n<image>识别内容\n====\n"
        cleaned = DeepSeekOCREngine._extract_result(None, stdout_text)
        assert cleaned == "识别内容"

    def test_multiple_tags_cleaning(self):
        from core.ocr_engine import DeepSeekOCREngine
        text = "<image><|v1|>第一行<|v2|>\n<image>第二行"
        cleaned = DeepSeekOCREngine._extract_result(text, "")
        assert cleaned == "第一行\n第二行"
