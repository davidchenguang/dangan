"""技术验证脚本：测试 DeepSeek-OCR-2 对50年代户籍卡的识别效果
用法: python test_ocr.py [--model MODEL_PATH] --image IMAGE_PATH [--prompt VERBATIM|markdown|json|english] [--no-preprocess] [--scale N]
"""

import argparse
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ===== 兼容性修复: transformers 4.57+ =====

# 修复 1: LlamaFlashAttention2 / LlamaSdpaAttention 已移除
import transformers.models.llama.modeling_llama as llama_module
if not hasattr(llama_module, "LlamaFlashAttention2"):
    if hasattr(llama_module, "LlamaAttention"):
        llama_module.LlamaFlashAttention2 = llama_module.LlamaAttention
        llama_module.LlamaSdpaAttention = llama_module.LlamaAttention
if not hasattr(llama_module, "LlamaSdpaAttention"):
    if hasattr(llama_module, "LlamaAttention"):
        llama_module.LlamaSdpaAttention = llama_module.LlamaAttention

# 修复 2: DynamicCache seen_tokens / get_usable_length / get_max_length 已移除
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, 'seen_tokens'):
    DynamicCache.seen_tokens = property(
        lambda self: self._seen_tokens
        if hasattr(self, '_seen_tokens')
        else self.get_seq_length()
    )
if not hasattr(DynamicCache, 'get_usable_length'):
    DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

# 修复 3: LlamaAttention.forward() 签名变更 (4.57+ 要求 position_embeddings 而非 position_ids)
_orig_llama_attn_forward = llama_module.LlamaAttention.forward
_orig_params = list(inspect.signature(_orig_llama_attn_forward).parameters.keys())

if "position_embeddings" in _orig_params:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

    def _compat_forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        hidden_shape = (bsz, q_len, self.config.num_attention_heads, head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if hasattr(self, 'rotary_emb') and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = None, None

        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        n_rep = self.config.num_attention_heads // self.config.num_key_value_heads
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    llama_module.LlamaAttention.forward = _compat_forward

from transformers import AutoModel, AutoTokenizer


# ===== 图像预处理 =====

def preprocess_image(image_path: str, output_dir: str = None, scale: float = 2.0,
                     clahe_clip: float = 3.0) -> str:
    """预处理户籍卡扫描件图片

    核心优化策略（经实验验证）：
    1. 放大 2x — 使手写文字达到模型可识别的分辨率阈值
    2. CLAHE 对比度增强 — 增强褪色手写文字与背景的对比度
    3. 不做去噪 — 去噪反而破坏笔画细节（经测试验证）

    注意：必须配合 crop_mode=True 使用，放大后的图片会触发 crop 机制
    产生多个 patches，每个 patch 分辨率更高，手写文字更容易识别。

    Args:
        image_path: 输入图片路径
        output_dir: 预处理后图片保存目录 (None则保存到 output/)
        scale: 放大倍数 (默认2.0，经测试2x效果最佳)
        clahe_clip: CLAHE 对比度限制 (默认3.0)

    Returns:
        预处理后图片路径
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    print(f"  原始尺寸: {w}x{h}")

    # Step 1: 放大 — 关键步骤，使手写文字达到模型可识别阈值
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    print(f"  放大后: {new_w}x{new_h}")

    # Step 2: CLAHE 对比度增强
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # 不做去噪 — fastNlMeansDenoisingColored 会破坏笔画细节

    # 保存
    save_dir = output_dir or str(Path("output"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    suffix = Path(image_path).suffix
    out_path = str(Path(save_dir) / f"{stem}_preprocessed{suffix}")
    cv2.imwrite(out_path, img)
    print(f"  预处理图片已保存: {out_path}")
    return out_path


def load_model(model_path: str):
    """加载 DeepSeek-OCR-2 模型"""
    print(f"[1/4] 正在加载模型: {model_path}")
    print("      首次加载可能需要几分钟，请耐心等待...")

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    # 修复 3 补充：给使用 LlamaAttention 的层注入 rotary_emb
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention) and not hasattr(module, 'rotary_emb'):
            module.rotary_emb = LlamaRotaryEmbedding(config=module.config, device=model.device)

    elapsed = time.time() - start
    print(f"      模型加载完成，耗时 {elapsed:.1f}s")

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"      GPU 显存占用: {vram_mb:.0f} MB")

    return tokenizer, model


def run_ocr(tokenizer, model, image_path: str, prompt: str,
            preprocess: bool = True, scale: float = 2.0):
    """执行 OCR 识别

    Args:
        preprocess: 是否预处理图片 (默认True)
        scale: 预处理放大倍数 (默认2.0)
    """
    actual_image = image_path

    if preprocess:
        print(f"\n[2/4] 正在预处理图片...")
        actual_image = preprocess_image(image_path, scale=scale)
        print(f"  使用预处理图片: {actual_image}")
    else:
        print(f"\n[2/4] 跳过预处理，使用原始图片")

    print(f"  正在识别图片...")
    start = time.time()

    output_dir = str(Path("output"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=actual_image,
        output_path=output_dir,
        base_size=1024,
        image_size=768,
        crop_mode=True,  # 必须开启 — 放大后的图片需要 crop 产生多 patch
        eval_mode=True,
    )
    elapsed = time.time() - start

    if isinstance(result, dict):
        text = result.get("text", str(result))
    else:
        text = str(result)

    print(f"      识别完成，耗时 {elapsed:.1f}s")
    return text


def extract_json(text: str):
    """尝试从 OCR 输出中提取 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def deduplicate_output(text: str, min_repeat: int = 3) -> str:
    """去除模型输出中的重复循环内容

    检测连续重复的行模式并截断。例如模型可能反复输出：
    - 何时由本市迁来本市
    - 何时由本市迁来本市
    - ...

    Args:
        text: 原始输出文本
        min_repeat: 最小重复次数阈值 (默认3)

    Returns:
        去重后的文本
    """
    lines = text.split('\n')
    if len(lines) < min_repeat:
        return text

    # 检测从某一行开始出现连续重复的模式
    for i in range(len(lines)):
        pattern = lines[i].strip()
        if not pattern or len(pattern) < 4:
            continue
        repeat_count = 0
        for j in range(i, len(lines)):
            if lines[j].strip() == pattern:
                repeat_count += 1
            else:
                break
        if repeat_count >= min_repeat:
            # 找到重复起点，截断并标注
            kept = lines[:i + 1]
            return '\n'.join(kept) + f'\n... (已截断 {repeat_count - 1} 行重复内容)'

    return text


def extract_fields_from_text(text: str) -> dict:
    """从 OCR 描述性输出中提取字段-值对

    支持以下格式：
    - Markdown: **字段名**: 值 / **字段名**：值
    - 键值对: 字段名: 值 / 字段名：值
    - 列表项: - 字段名: 值

    Returns:
        字段名 -> 值 的字典
    """
    fields = {}

    # 跳过的字段名黑名单（描述性/非字段内容）
    skip_keys = {'标题', '表格结构', '图片内容描述', '左侧部分', '右侧部分'}

    # 模式1: Markdown 加粗字段名 **字段名**：值
    # 逐行处理以避免跨行贪婪匹配
    for line in text.split('\n'):
        line = line.strip().lstrip(',').strip()
        m = re.match(r'\*\*(.+?)\*\*\s*[:：]\s*(.+)', line)
        if not m:
            continue
        key = m.group(1).strip()
        value = m.group(2).strip()
        # 清理 value: 去掉首尾逗号、星号、空白
        value = value.strip(',').strip('*').strip()
        # 跳过黑名单、空值、"无"
        if not key or key in skip_keys:
            continue
        if not value or value == '无':
            continue
        # 跳过已存在的字段（保留首次出现的值，更可靠）
        if key in fields:
            continue
        fields[key] = value

    return fields


def export_to_csv(fields: dict, output_path: str):
    """导出字段到 CSV 文件"""
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['字段名', '值'])
        for key, value in fields.items():
            writer.writerow([key, value])
    print(f"  CSV 已导出: {output_path}")


def export_to_excel(fields: dict, output_path: str):
    """导出字段到 Excel 文件"""
    try:
        import openpyxl
    except ImportError:
        print("  警告: openpyxl 未安装，跳过 Excel 导出 (pip install openpyxl)")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "户籍卡识别结果"

    # 表头
    ws['A1'] = '字段名'
    ws['B1'] = '值'
    ws['A1'].font = openpyxl.styles.Font(bold=True)
    ws['B1'].font = openpyxl.styles.Font(bold=True)

    # 数据
    for i, (key, value) in enumerate(fields.items(), start=2):
        ws[f'A{i}'] = key
        ws[f'B{i}'] = value

    # 自动列宽
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 40

    wb.save(output_path)
    print(f"  Excel 已导出: {output_path}")


def validate_result(raw_text: str, export: bool = False, image_name: str = ""):
    """验证识别结果并可选导出

    Args:
        raw_text: OCR 原始输出
        export: 是否导出结构化文件
        image_name: 图片文件名 (用于导出文件命名)
    """
    print("\n[3/4] 验证识别结果...")

    # 去重
    clean_text = deduplicate_output(raw_text)

    print("\n── 原始输出 (去重后) ──")
    print(clean_text)

    # 尝试 JSON 解析
    parsed = extract_json(clean_text)
    if parsed:
        print("\n── JSON 解析成功 ──")
        for key, value in parsed.items():
            status = "[OK]" if value else "[  ]"
            print(f"  {status} {key}: {value}")

        if export:
            stem = Path(image_name).stem if image_name else "result"
            export_to_csv(parsed, f"output/{stem}_result.csv")
            export_to_excel(parsed, f"output/{stem}_result.xlsx")
        return parsed

    # 尝试从描述性文本提取字段
    fields = extract_fields_from_text(clean_text)
    if fields:
        print("\n── 字段提取成功 ──")
        for key, value in fields.items():
            print(f"  [OK] {key}: {value}")

        if export:
            stem = Path(image_name).stem if image_name else "result"
            export_to_csv(fields, f"output/{stem}_result.csv")
            export_to_excel(fields, f"output/{stem}_result.xlsx")
        return fields

    print("\n-- 自动结构化失败，请手动处理原始输出 --")
    return None


# ===== Prompt 模板集合 =====

# 最佳 prompt (经测试验证)：简洁直接，避免过长导致模型重复或输出空)
# 关键发现：prompt 越短越好，长 prompt 会触发模型重复循环或空输出
PROMPT_VERBATIM = "<image>\n请逐字精确识别这张户籍登记表扫描件中的所有文字内容，包括手写和印刷文字。"

# Markdown 表格 prompt
PROMPT_MARKDOWN = """<image>
请将这张户籍登记表的内容转换为 markdown 表格。请仔细识别每一个手写填写的文字，不要遗漏。如果某个字段看不清，请标注[模糊]。"""

# JSON prompt
CARD_OCR_PROMPT = """<image>
<|grounding|>这是50年代常住人口登记表的扫描件，黄色底色,包含手写和印刷文字。
请按以下 JSON 格式输出所有可见字段,字段为空则填空字符串：
{
  "户主或与户主关系": "",
  "姓名": "", "别名": "", "性别": "",
  "出生日期": "", "出生地址": "",
  "籍贯": "", "民族": "", "宗教信仰": "",
  "婚姻状况": "", "文化程度": "",
  "职业": "", "服务处所": "",
  "本市其他住所": "",
  "公民证代号号码": "", "签发机关": "", "签发日期": "",
  "何时由何地迁来本市": "",
  "何时由本市何处迁来本地": "",
  "注销户口日期和原因": "",
  "户口登记事项变更更正记载": ""
}
忽略印章、污渍和装订痕迹,只识别正式填写的文字内容。"""

# 英文 prompt
PROMPT_ENGLISH = """<image>
Please perform OCR on this Chinese household registration card from the 1950s. The card has yellow background with both printed headers and handwritten content.
Please transcribe ALL text you can see, both printed and handwritten. Output each field name and its value. Focus especially on the handwritten text in each field."""


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 户籍卡识别测试")
    parser.add_argument(
        "--model",
        default=r"C:\Users\cheng\.lmstudio\models\forkjoin-ai\deepseek-ocr-2",
        help="模型路径",
    )
    parser.add_argument(
        "--image",
        default=r"C:\Users\cheng\Desktop\1.jpg",
        help="测试图片路径",
    )
    parser.add_argument(
        "--prompt",
        choices=["all", "verbatim", "markdown", "json", "english"],
        default="verbatim",
        help="测试哪个 Prompt (默认 verbatim)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="禁用图片预处理",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="预处理放大倍数 (默认2.0)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="导出结构化表格文件 (CSV + Excel)",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"错误: 图片不存在: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is not None:
        h, w = img.shape[:2]
        print(f"图片尺寸: {w}x{h}")
    else:
        print("警告: OpenCV 无法读取图片，但继续尝试 OCR")

    tokenizer, model = load_model(args.model)

    prompts = {
        "逐字识别": PROMPT_VERBATIM,
        "Markdown 表格": PROMPT_MARKDOWN,
        "JSON 输出": CARD_OCR_PROMPT,
        "英文 OCR": PROMPT_ENGLISH,
    }

    if args.prompt == "all":
        for name, prompt in prompts.items():
            print(f"\n{'='*60}")
            print(f"测试 Prompt: {name} (预处理: {'关' if args.no_preprocess else '开'})")
            print(f"{'='*60}")
            text = run_ocr(
                tokenizer, model, args.image, prompt,
                preprocess=not args.no_preprocess,
                scale=args.scale,
            )
            validate_result(text, export=args.export, image_name=args.image)
    else:
        prompt_key = {
            "verbatim": "逐字识别",
            "markdown": "Markdown 表格",
            "json": "JSON 输出",
            "english": "英文 OCR",
        }
        name = prompt_key[args.prompt]
        prompt = prompts[name]
        print(f"\n{'='*60}")
        print(f"测试 Prompt: {name} (预处理: {'关' if args.no_preprocess else '开'})")
        print(f"{'='*60}")
        text = run_ocr(
            tokenizer, model, args.image, prompt,
            preprocess=not args.no_preprocess,
            scale=args.scale,
        )
        validate_result(text, export=args.export, image_name=args.image)

    print(f"\n{'='*60}")
    print("[4/4] 技术验证完成")


if __name__ == "__main__":
    main()
