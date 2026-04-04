# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**50年代老旧户籍卡 OCR 识别系统** — 基于 DeepSeek-OCR-2 本地大模型的桌面 GUI 应用，用于读取50年代户籍卡扫描图片中的手写文字信息，支持单张查看编辑和批量处理导出。

- **运行环境：** Windows + RTX 5090D，全本地运行，不联网
- **GUI 框架：** PySide6/Qt
- **OCR 引擎：** DeepSeek-OCR-2（本地模型）
- **辅助工具：** OpenCV（预处理）、RapidLayout（版面分析）

## 技术栈

- Python 3.12+（推荐通过 uv 管理：`C:\Users\cheng\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe`）
- PySide6 >= 6.6, PyTorch >= 2.6.0, transformers >= 4.46.3, flash-attn >= 2.7.3
- OpenCV, numpy, Pillow — 图像处理
- openpyxl, python-docx, reportlab — 多格式导出
- 模型路径默认：`C:\Users\cheng\.lmstudio\models\forkjoin-ai\deepseek-ocr-2`

## 项目架构

```
dangan/
├── main.py                    # 应用入口（规划中）
├── test_ocr.py                # DeepSeek-OCR-2 技术验证脚本
├── config.yaml                # 配置文件（规划中）
├── docs/plans/                # 设计文档
├── core/                      # 核心业务逻辑（零 Qt 依赖）
│   ├── pipeline.py            # 流水线编排：预处理 → 版面分析 → OCR → 结构化
│   ├── preprocessor.py        # 图像预处理（倾斜校正、去噪、印章去除、对比度增强）
│   ├── layout_analyzer.py     # 版面分析，定位表格字段区域
│   ├── ocr_engine.py          # OCR 引擎抽象接口 + DeepSeek-OCR-2 实现
│   ├── field_extractor.py     # OCR 输出 → 结构化字段（JSON/Markdown/文本三级降级）
│   ├── exporter.py            # 多格式导出（Excel/CSV/JSON/PDF/Word）
│   └── models.py              # 数据结构定义（HouseholdCard 等）
├── gui/                       # PySide6 界面层
│   ├── main_window.py         # 主窗口（三栏布局）
│   ├── image_viewer.py        # QGraphicsView 图片查看器 + 字段区域高亮
│   ├── workers.py             # QRunnable 异步 OCR 任务
│   └── ...
└── tests/                     # 测试
```

### 分层原则

- **`core/` 零 Qt 依赖** — 可独立测试和复用
- **`gui/` 只负责展示和交互调度** — 不含业务逻辑
- **流水线数据流：** `图片 → ImagePreprocessor → LayoutAnalyzer → DeepSeekOCREngine → FieldExtractor → HouseholdCard → ResultExporter`

### 核心数据结构

- `HouseholdCard` — 一张户籍卡的结构化识别结果，包含简单字段（姓名、性别等）、复合字段（出生、公民证）、记录型字段（迁移/注销记录）
- `FieldRegion` — 版面分析中字段位置区域（x, y, w, h）
- `ProcessingStatus` — 枚举：pending / processing / done / failed
- `PreprocessConfig` — 预处理参数配置，默认关闭二值化（DeepSeek-OCR-2 对灰度图效果更好）

## 常用命令

```bash
# 运行 OCR 技术验证（测试所有 Prompt 策略）
python test_ocr.py --image <图片路径> --model <模型路径>

# 仅测试 JSON 输出 Prompt
python test_ocr.py --image <图片路径> --prompt json

# 仅测试 Markdown 表格 Prompt
python test_ocr.py --image <图片路径> --prompt markdown

# 仅测试纯文本 OCR Prompt
python test_ocr.py --image <图片路径> --prompt text
```

## 关键实现细节

### transformers 兼容性修复

`test_ocr.py` 包含针对 transformers 4.57+ 的兼容性补丁：
- `LlamaFlashAttention2` / `LlamaSdpaAttention` 已移除 → 回退到 `LlamaAttention`
- `DynamicCache.seen_tokens` / `get_usable_length` / `get_max_length` 已移除 → 属性注入兼容

迁移到正式代码时需保留这些补丁。

### 户籍卡 OCR Prompt 策略

使用 `<|grounding|>` 标记配合 JSON 格式要求，要求模型输出结构化字段。字段提取器采用三级降级：JSON 解析 → Markdown 表格解析 → 原始文本保留。

### 预处理流水线顺序

倾斜校正 → 边缘修补 → 背景归一化（去黄） → 印章去除 → 去噪 → 对比度增强（CLAHE）

## Spec Workflow 文档体系

项目使用 `.spec-workflow/` 管理规范文档流程。开发新功能前先创建指导文档（steering/），再创建功能规范（specs/）。

- 创建规范前调用 `spec-workflow-guide` MCP 工具获取完整流程
- 任务模板遵循从底层到顶层：类型定义 → 模型 → 服务 → API → 前端 → 集成测试
- 详细设计文档见 `docs/plans/2026-04-02-household-card-ocr-design.md`
