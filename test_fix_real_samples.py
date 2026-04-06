import os
import sys
import torch
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.getcwd())

from core.ocr_engine import DeepSeekOCREngine
from core.models import CARD_OCR_PROMPT
from core.compat import apply_patches

def test_samples():
    # 必须在加载模型前应用补丁
    apply_patches()
    
    logging.basicConfig(level=logging.INFO)
    
    model_path = r"C:\Users\cheng\.lmstudio\models\forkjoin-ai\deepseek-ocr-2"
    engine = DeepSeekOCREngine(model_path=model_path)
    
    print("[1/3] 正在加载模型...")
    engine.load()
    
    samples = ["resources/1.jpg", "resources/2.jpg"]
    
    for img_path in samples:
        print(f"\n[2/3] 正在识别: {img_path}")
        if not os.path.exists(img_path):
            print(f"！！！文件不存在: {img_path}")
            continue
            
        # 使用我们修复过的引擎进行识别
        # 它内部会调用 _extract_result 自动清洗 <image> 等标签
        text = engine.recognize(img_path, CARD_OCR_PROMPT)
        
        print(f"\n--- {img_path} 识别结果 ---")
        print(text)
        print("-" * 30)
        
        # 验证清洗效果
        if "<image>" in text:
            print("❌ 错误: 结果中仍包含 <image> 标签")
        else:
            print("✅ 成功: 结果中不含 <image> 标签")
            
        if "<|" in text:
            print("❌ 错误: 结果中仍包含 <|...|> 标签")
        else:
            print("✅ 成功: 结果中不含 VLM 内部标签")

    print("\n[3/3] 测试完成")

if __name__ == "__main__":
    test_samples()
