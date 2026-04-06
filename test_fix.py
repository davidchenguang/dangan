"""快速验证 OCR 修复效果"""
import sys
import os
import time
import logging

os.environ.setdefault("PYTHONUTF8", "1")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)

from core.compat import apply_patches
apply_patches()

from core.config import AppConfig
from core.pipeline import OcrPipeline
from pathlib import Path

config = AppConfig()
config.load()
print(f"Config: prompt={config.ocr_prompt}, voting_rounds={config.ocr_voting_rounds}")

# 创建 pipeline
pipeline = OcrPipeline(
    model_path=config.model_path,
    preprocess_config=config.preprocess,
    prompt=config.ocr_prompt,
    base_size=config.ocr_base_size,
    image_size=config.ocr_image_size,
    crop_mode=config.ocr_crop_mode,
    voting_rounds=config.ocr_voting_rounds,
)
print("Initializing pipeline (loading model)...")
pipeline.initialize()
print("Pipeline initialized")

# 优先使用已预处理的图片
preprocessed = list(Path("output").glob("*_preprocessed*"))
if preprocessed:
    image_path = str(preprocessed[0])
    print(f"Testing with preprocessed: {image_path}")
else:
    images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
    if not images:
        print("No test images found")
        sys.exit(1)
    image_path = str(images[0])
    print(f"Testing with: {image_path}")

print(f"Voting rounds: {config.ocr_voting_rounds}")
start = time.time()
result = pipeline.process(image_path, preprocess=True)
elapsed = time.time() - start
print(f"Result: success={result.success}, error={result.error}, time={elapsed:.1f}s")
card = result.card
print(f"Name: {card.name!r}")
print(f"Gender: {card.gender!r}")
print(f"Birth date: {card.birth.date!r}")
print(f"Native place: {card.native_place!r}")
print(f"Ethnicity: {card.ethnicity!r}")
print(f"Education: {card.education!r}")
print(f"Occupation: {card.occupation.occupation!r}")
print(f"Raw markdown length: {len(card.raw_markdown)}")
if card.raw_markdown:
    print(f"Raw markdown preview: {card.raw_markdown[:300]}")
