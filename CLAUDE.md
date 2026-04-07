# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**50年代老旧户籍卡 OCR 识别系统** — 基于 DeepSeek-OCR-2 本地大模型的桌面 GUI 应用，用于读取50年代户籍卡扫描图片中的手写文字信息，支持单张查看编辑和批量处理导出。项目已处于功能完善的准 Beta 阶段。

- **运行环境：** Windows，全本地运行，支持无独立显卡环境柔性降级运行（自动适配 CUDA/CPU 和 bf16/fp16/fp32）
- **GUI 框架：** PySide6/Qt
- **OCR 引擎：** DeepSeek-OCR-2（本地模型）
- **辅助工具：** OpenCV（预处理）、RapidLayout（版面分析）

## 技术栈

- Python 3.12+（推荐通过 uv 管理环境）
- PySide6 >= 6.6, PyTorch >= 2.6.0, transformers >= 4.46.3, flash-attn >= 2.7.3
- OpenCV, numpy, Pillow — 图像处理
- openpyxl, python-docx, reportlab — 多格式导出

## 项目架构

```
dangan/
├── main.py                    # 应用入口（已实现）
├── test_ocr.py                # DeepSeek-OCR-2 技术验证脚本
├── config.yaml                # 配置文件（已实现，支持 DANGAN_MODEL_PATH 环境变量覆盖）
├── docs/plans/                # 设计文档
├── core/                      # 核心业务逻辑（零 Qt 依赖，测试覆盖率高）
│   ├── pipeline.py            # 流水线编排：预处理 → 版面分析 → OCR → 结构化
│   ├── preprocessor.py        # 图像预处理（倾斜校正、去噪、印章去除、对比度增强）
│   ├── layout_analyzer.py     # 版面分析，定位表格字段区域
│   ├── ocr_engine.py          # OCR 引擎抽象接口 + DeepSeek-OCR-2 实现
│   ├── field_extractor.py     # OCR 输出 → 结构化字段（JSON/Markdown/文本三级降级）
│   ├── exporter.py            # 多格式导出（Excel/CSV/JSON/PDF/Word）
│   ├── models.py              # 数据结构定义（HouseholdCard 等）
│   ├── augmentation.py        # 图像增强机制
│   ├── voting.py              # 多轮 OCR 投票纠错机制
│   ├── deduplicator.py        # 识别结果去重/清洗策略
│   └── patches/               # 第三方库猴子补丁隔离区
│       └── transformers_fix.py # 针对 transformers 4.57+ 的兼容断言修复
├── gui/                       # PySide6 界面层
│   ├── main_window.py         # 主窗口（三栏布局）
│   ├── image_viewer.py        # QGraphicsView 图片查看器 + 字段区域高亮
│   ├── result_table.py        # 识别结果表格呈现
│   ├── export_dialog.py       # 导出选项弹窗
│   ├── settings_dialog.py     # 配置弹窗
│   ├── thumbnail_panel.py     # 侧边缩略图树
│   └── workers.py             # QRunnable 异步 OCR 任务
└── tests/                     # 完整的单元与集成测试（包含针对各个 core 模块的测试用例）
```

### 分层原则

- **`core/` 零 Qt 依赖** — 可独立测试和复用
- **`gui/` 只负责展示和交互调度** — 不含业务逻辑
- **流水线数据流：** `图片 → 预处理/增强 → LayoutAnalyzer → DeepSeekOCREngine (含多轮 Voting 机制) → FieldExtractor → HouseholdCard → ResultExporter`

### 核心数据结构与配置

- `HouseholdCard` — 一张户籍卡的结构化识别结果，包含简单字段（姓名、性别等）、复合字段（出生、公民证）、记录型字段（迁移/注销记录）
- `FieldRegion` — 版面分析中字段位置区域（x, y, w, h）
- `ProcessingStatus` — 枚举：pending / processing / done / failed
- `AppConfig` (`core/config.py`) — 单例配置管理器，支持模型路径后备 (`./models/` 或环境设定)，并提供设备柔性检测。

## 关键实现细节

### 容错与防御性设计
- **设备降级策略**：硬件检测从硬编码 `cuda/bfloat16` 调整为了探测环境动态挂载（fallback 到 CPU / float16 或 float32）。
- **相对路径优先机制**：消除强制关联任意特定的 Windows 本地绝对目录（如特定用户的桌面或`.lmstudio`路径）。

### transformers 兼容性修复
项目中提取了对 transformers 的猴子补丁到 `core/patches/transformers_fix.py`中：
- 修复 `LlamaFlashAttention2` / `LlamaSdpaAttention` 未找到时回退 `LlamaAttention`。
- 注入 `DynamicCache.seen_tokens` 等丢失属性。
- 覆写了基于 4.57+ 签名（要求 `position_embeddings`）的 `LlamaAttention.forward`。

### OCR 增强策略
引入了 `voting.py` 和在配置中的 `voting_rounds`（默认为3次）。该设计通过改变微弱的 Prompt 温度系数或裁剪方式形成多出路识别，然后使用众数投票解决历史文档中极为模糊的字迹识别。

## 常用命令

```bash
# 运行主界面
python main.py

# 运行基础 OCR 技术验证
python test_ocr.py --image <图片路径> --prompt json

# 运行针对 core/ 层的所有测试
pytest tests/
```

## Spec Workflow 文档体系

项目使用 `.spec-workflow/` 管理规范文档流程。开发新功能前先创建指导文档（steering/），再创建功能规范（specs/）。

- 创建规范前调用 `spec-workflow-guide` MCP 工具获取完整流程
- 任务模板遵循从底层到顶层：类型定义 → 模型 → 服务 → API → 前端 → 集成测试
- 详细设计文档见 `docs/plans/2026-04-02-household-card-ocr-design.md`
