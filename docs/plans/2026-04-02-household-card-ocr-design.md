# 50年代老旧户籍卡 OCR 识别系统 — 设计文档

## 概述

基于 DeepSeek-OCR-2 本地大模型，开发一个桌面 GUI 应用，用于读取50年代老旧户籍卡（图片方式）中的手写文字信息，支持单张查看编辑和批量处理导出。

- **运行环境：** Windows + RTX 5090D，全本地运行，不联网
- **GUI 框架：** PySide6/Qt
- **OCR 引擎：** DeepSeek-OCR-2 (`deepseek-ai/DeepSeek-OCR-2`)
- **辅助工具：** OpenCV（预处理）、RapidLayout（版面分析）
- **导出格式：** Excel/CSV、JSON、PDF 报告、Word 文档

---

## 一、整体架构

### 分层流水线架构

```
┌─────────────────────────────────────────┐
│           UI 层 (PySide6)               │
│  主窗口 / 图片查看器 / 结果面板 / 导出    │
├─────────────────────────────────────────┤
│         任务调度层                       │
│  QThreadPool + QRunnable 异步处理        │
├─────────────────────────────────────────┤
│         业务流水线层                     │
│  预处理 → 版面分析 → OCR → 结构化提取    │
├─────────────────────────────────────────┤
│         引擎抽象层                       │
│  OCREngine 接口 / ImageProcessor 接口    │
├─────────────────────────────────────────┤
│         基础设施层                       │
│  文件 I/O / 模型加载 / 配置管理          │
└─────────────────────────────────────────┘
```

### 数据流

```
图片文件 → ImagePreprocessor → LayoutAnalyzer → DeepSeekOCREngine → FieldExtractor → HouseholdCard
                                                                                      ↓
                                                                              ResultExporter → Excel/JSON/PDF/Word
```

### 核心模块

| 模块 | 职责 | 关键类 |
|------|------|--------|
| `core/pipeline.py` | 流水线编排，串联各阶段 | `OcrPipeline` |
| `core/preprocessor.py` | 图像预处理（去噪、倾斜校正、对比度增强） | `ImagePreprocessor` |
| `core/layout_analyzer.py` | 版面分析，定位表格字段区域 | `LayoutAnalyzer` |
| `core/ocr_engine.py` | DeepSeek-OCR-2 封装 | `DeepSeekOCREngine` |
| `core/field_extractor.py` | 从 OCR 结果提取结构化字段 | `FieldExtractor` |
| `core/exporter.py` | 多格式导出 | `ResultExporter` |
| `core/models.py` | 所有数据结构 | `HouseholdCard` 等 |

---

## 二、核心数据结构

### 基于50年代常住人口登记表的实际字段

字段分为三类：

**简单字段（单值）：**
- 姓名、别名、性别、籍贯、民族、宗教信仰、婚姻状况、文化程度

**复合字段（含子字段）：**
- 出生 → 日期 + 地址
- 公民证 → 代号号码 + 签发机关 + 签发日期
- 职业及服务处所 → 职业 + 服务处所

**记录型字段（可变条目、多行）：**
- 何时由何地迁来本市
- 何时由本市何处迁来本地
- 注销户口日期和原因
- 户口登记事项变更更正记载

### 数据结构定义

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QPixmap

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

# ── 基础字段区域（版面分析用）──

@dataclass
class FieldRegion:
    """表格中一个字段/子字段的位置区域"""
    x: int; y: int; w: int; h: int
    label: str

# ── 复合字段结构 ──

@dataclass
class BirthInfo:
    """出生"""
    date: str = ""          # 日期
    address: str = ""       # 地址

@dataclass
class CitizenCertificate:
    """公民证"""
    code_number: str = ""           # 代号号码
    issuing_authority: str = ""     # 签发机关
    issue_date: str = ""            # 签发日期

@dataclass
class OccupationInfo:
    """职业及服务处所"""
    occupation: str = ""    # 职业
    workplace: str = ""     # 服务处所

@dataclass
class ChangeRecord:
    """变更/迁移/注销记录（通用结构）"""
    date: str = ""          # 日期
    content: str = ""       # 内容描述

# ── 完整户籍卡 ──

@dataclass
class HouseholdCard:
    """一张户籍卡的结构化识别结果"""

    # 简单字段
    name: str = ""                          # 姓名
    alias: str = ""                         # 别名
    gender: str = ""                        # 性别
    native_place: str = ""                  # 籍贯
    ethnicity: str = ""                     # 民族
    religion: str = ""                      # 宗教信仰
    marital_status: str = ""                # 婚姻状况
    education: str = ""                     # 文化程度
    other_residence: str = ""               # 本市其他住所
    relation_to_household_head: str = ""    # 户主或与户主关系

    # 复合字段
    birth: BirthInfo = field(default_factory=BirthInfo)
    occupation: OccupationInfo = field(default_factory=OccupationInfo)
    citizen_cert: CitizenCertificate = field(default_factory=CitizenCertificate)

    # 记录型字段（可多条）
    migration_in: list[ChangeRecord] = field(default_factory=list)      # 何时由何地迁来本市
    migration_local: list[ChangeRecord] = field(default_factory=list)   # 何时由本市何处迁来本地
    cancellation: list[ChangeRecord] = field(default_factory=list)      # 注销户口日期和原因
    changes: list[ChangeRecord] = field(default_factory=list)           # 户口登记事项变更更正记载

    # 元数据
    source_image: str = ""            # 源图片路径
    raw_markdown: str = ""            # DeepSeek-OCR-2 原始输出
    confidence: dict[str, float] = field(default_factory=dict)
    reviewed: bool = False

# ── 流水线中间数据 ──

@dataclass
class ImageItem:
    """一张待处理的户籍卡图片"""
    id: str                          # UUID
    filepath: Path                   # 原始文件路径
    status: ProcessingStatus         # pending/processing/done/failed
    thumbnail: QPixmap | None        # UI 缩略图缓存

@dataclass
class ProcessedImage:
    """预处理后的图像"""
    source: ImageItem
    enhanced: np.ndarray             # 增强后的图像（OpenCV 格式）
    rotations_applied: float         # 倾斜校正角度

@dataclass
class LayoutResult:
    """版面分析结果"""
    source: ProcessedImage
    fields: dict[str, FieldRegion]   # 字段名 → 区域坐标

# ── 版面分析字段定义 ──

HOUSEHOLD_CARD_SCHEMA = {
    # 简单字段
    "户主或与户主关系": True,
    "姓名": True,
    "别名": False,
    "性别": True,
    "籍贯": True,
    "民族": True,
    "宗教信仰": False,
    "婚姻状况": False,
    "文化程度": False,
    "本市其他住所": False,
    # 复合字段
    "出生": ["日期", "地址"],
    "职业及服务处所": ["职业", "服务处所"],
    "公民证": ["代号号码", "签发机关", "签发日期"],
    # 记录型字段
    "何时由何地迁来本市": "records",
    "何时由本市何处迁来本地": "records",
    "注销户口日期和原因": "records",
    "户口登记事项变更更正记载": "records",
}
```

---

## 三、图像预处理流水线

50年代户籍卡的典型退化：泛黄褪色、墨迹模糊、污渍遮挡、折痕破损、倾斜变形、印章覆盖、装订痕迹。

### 处理流程（按顺序）

```
倾斜校正 → 边缘修补（裁切+inpainting） → 背景归一化（去黄） → 印章去除 → 去噪 → 对比度增强
```

### 各步骤算法

| 步骤 | 算法 | 说明 |
|------|------|------|
| 倾斜校正 | OpenCV 霍夫变换检测表格线 → 计算旋转角度 | 户籍卡有明确的表格线，检测可靠 |
| 边缘修补 | 水平/垂直投影检测内容边界 → inpainting 修复装订孔 → 裁切 | 去掉装订痕迹和破损边缘 |
| 背景归一化 | LAB 色彩空间 + 形态学闭运算估计背景 → 除法归一化 | 均匀黄色底色校正为白色 |
| 印章去除 | HSV 色彩空间过滤红色区域 → 邻域像素填充 | 红色印章干扰手写文字 |
| 去噪 | 非局部均值去噪（`cv2.fastNlMeansDenoising`） | 保留笔画细节，去除斑点噪声 |
| 对比度增强 | CLAHE（自适应直方图均衡化） | 局部增强，避免泛黄背景过曝 |

### 配置

```python
@dataclass
class PreprocessConfig:
    auto_rotate: bool = True
    repair_edges: bool = True
    normalize_background: bool = True
    remove_stamps: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    binarize: bool = False          # 默认关闭，DeepSeek-OCR-2 对灰度图效果更好
```

---

## 四、OCR 引擎与字段提取

### 双通道识别策略

```
通道 1（整卡）：整张图 → DeepSeek-OCR-2 → JSON/Markdown → 字段提取
通道 2（补识别）：对置信度低的字段 → 裁切该区域 → 单独送 OCR → 覆盖结果
```

整卡识别优先，因为 DeepSeek-OCR-2 能利用字段间上下文关系提高准确率。仅低置信度字段触发补识别。

### 引擎抽象

```python
from abc import ABC, abstractmethod

class OCREngine(ABC):
    """OCR 引擎抽象接口"""
    @abstractmethod
    def recognize(self, image: np.ndarray, prompt: str) -> str: ...

    @abstractmethod
    def recognize_region(self, image: np.ndarray, region: FieldRegion) -> str: ...

class DeepSeekOCREngine(OCREngine):
    """DeepSeek-OCR-2 封装"""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        ).eval().cuda().to(torch.bfloat16)

    def recognize(self, image: np.ndarray, prompt: str) -> str:
        """整卡识别"""
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            cv2.imwrite(tmp.name, image)
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=tmp.name,
                base_size=1024,
                image_size=768,
                crop_mode=True,
            )
        return result["text"] if isinstance(result, dict) else str(result)

    def recognize_region(self, image: np.ndarray, region: FieldRegion) -> str:
        """裁切字段区域后单独识别"""
        crop = image[region.y:region.y+region.h, region.x:region.x+region.w]
        prompt = "<image>\n请精确识别这张图片中的手写文字，只输出文字内容。"
        return self.recognize(crop, prompt)
```

### 户籍卡专用 Prompt

```python
CARD_OCR_PROMPT = """<image>
<|grounding|>这是50年代常住人口登记表的扫描件，黄色底色，包含手写和印刷文字。
请按以下 JSON 格式输出所有可见字段，字段为空则填空字符串：
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
忽略印章、污渍和装订痕迹，只识别正式填写的文字内容。"""
```

### 字段提取器（三级降级）

```python
class FieldExtractor:
    """从 OCR 原始输出中提取结构化字段"""

    def extract(self, raw_text: str) -> HouseholdCard:
        # 1. 尝试解析 JSON（优先）
        card = self._parse_json(raw_text)
        if card:
            return card

        # 2. 降级：解析 Markdown 表格
        card = self._parse_markdown_table(raw_text)
        if card:
            return card

        # 3. 最终降级：整段文本作为 raw_markdown 返回
        return HouseholdCard(raw_markdown=raw_text)
```

---

## 五、PySide6 GUI 界面

### 主窗口三栏布局

```
┌──────────────────────────────────────────────────────────────────┐
│  菜单栏：文件 | 编辑 | 处理 | 导出 | 设置                         │
├──────────┬───────────────────────────────────┬──────────────────┤
│          │                                   │                  │
│  缩略图  │        图片查看器                  │   识别结果面板    │
│  列表    │    （QGraphicsView）              │                  │
│          │    图片 + 字段区域高亮             │  字段名 | 识别值  │
│  滚动    │    缩放/平移                       │  ──────────────  │
│  列表    │                                   │  姓名   | 张三   │
│          │                                   │  性别   | 男     │
│  ◻ 图片1 │                                   │  出生日期| ...    │
│  ◻ 图片2 │                                   │  籍贯   | ...    │
│  ◻ ...   │                                   │  ...    | ...    │
│          ├───────────────────────────────────┤                  │
│          │  进度条 / 状态栏                    │                  │
├──────────┴───────────────────────────────────┴──────────────────┤
│  状态栏：模型状态 | GPU 显存 | 当前进度                           │
└──────────────────────────────────────────────────────────────────┘
```

### 三栏分工

| 面板 | 组件 | 职责 |
|------|------|------|
| 左栏 | `QListWidget` (IconMode) | 图片缩略图，支持多选，显示处理状态图标 |
| 中栏 | 自定义 `QGraphicsView` | 原图/预处理后图像，叠加字段区域高亮框，缩放平移 |
| 右栏 | `QTableWidget`（可编辑） | 字段名+识别值两列，点击编辑，修改后标记为"已校正" |

### 字段高亮

图片查看器中叠加 `QGraphicsRectItem`，颜色按置信度区分：
- 绿色：置信度 > 0.9
- 黄色：置信度 0.7 ~ 0.9
- 红色：置信度 < 0.7

悬停时显示字段名和识别文本。

### 核心交互流程

```
导入图片（拖拽或菜单）
  → 左栏生成缩略图（状态：待处理）
  → 点击缩略图 → 中栏显示原图
  → 点击"识别"按钮
  → QThreadPool 提交 OcrWorker
  → 进度条更新
  → 完成后：
    → 中栏叠加字段区域高亮
    → 右栏填充识别结果
    → 用户可点击右栏字段直接编辑
  → 编辑后自动标记 reviewed = True
```

### 异步任务调度

```python
class OcrWorker(QRunnable):
    """每张图片的 OCR 在独立 QRunnable 中执行"""

    def __init__(self, fn, image_id, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.image_id = image_id
        self.signals = OcrWorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(self.image_id, result)
        except Exception:
            self.signals.error.emit(...)
        finally:
            self.signals.finished.emit(self.image_id)
```

### 工具栏快捷操作

| 按钮 | 功能 |
|------|------|
| 导入图片 | 文件选择对话框（支持多选、文件夹） |
| 全部识别 | 批量提交所有待处理图片 |
| 识别当前 | 仅识别当前选中的图片 |
| 切换视图 | 原图 / 预处理后 / 识别标注 三种视图切换 |
| 导出 | 弹出导出对话框选择格式和范围 |
| 设置 | 预处理参数配置、模型路径配置 |

---

## 六、多格式导出

### 架构

统一数据源 + 独立格式渲染器：

```python
class ResultExporter:
    def __init__(self):
        self._renderers = {
            ".xlsx": ExcelRenderer(),
            ".csv":  CsvRenderer(),
            ".json": JsonRenderer(),
            ".pdf":  PdfRenderer(),
            ".docx": WordRenderer(),
        }

    def export(self, cards: list[HouseholdCard], filepath: str):
        ext = Path(filepath).suffix
        self._renderers[ext].render(cards, filepath)
```

### 各格式特点

| 格式 | 库 | 内容 |
|------|-----|------|
| Excel | openpyxl | 表头冻结、自动列宽，每行一张卡 |
| CSV | 标准库 | 逗号分隔，兼容性强 |
| JSON | 标准库 | dataclass 序列化，保留完整嵌套结构 |
| PDF | reportlab | 原图 + 识别结果对照表格，A4 排版 |
| Word | python-docx | 可编辑文档，原图 + 字段表格 |

### 导出对话框

- 4 种格式可同时勾选，一次操作生成多种文件
- 导出范围：全部 / 仅已识别 / 仅已校正 / 当前选中
- PDF/Word 包含原图 + 识别结果对照，方便人工审核

---

## 七、项目结构

```
dangan/
├── main.py                          # 应用入口
├── requirements.txt                 # 依赖清单
├── config.yaml                      # 默认配置文件
│
├── core/                            # 核心业务逻辑（不依赖 Qt）
│   ├── __init__.py
│   ├── pipeline.py                  # 流水线编排
│   ├── preprocessor.py              # 图像预处理
│   ├── layout_analyzer.py           # 版面分析
│   ├── ocr_engine.py                # OCR 引擎抽象 + DeepSeek 实现
│   ├── field_extractor.py           # 字段提取与结构化
│   ├── exporter.py                  # 多格式导出
│   └── models.py                    # 所有数据结构
│
├── gui/                             # PySide6 界面层
│   ├── __init__.py
│   ├── main_window.py               # QMainWindow 主窗口
│   ├── image_viewer.py              # QGraphicsView 图片查看器 + 高亮
│   ├── thumbnail_panel.py           # 缩略图列表
│   ├── result_table.py              # 识别结果可编辑表格
│   ├── export_dialog.py             # 导出设置对话框
│   ├── settings_dialog.py           # 设置对话框
│   └── workers.py                   # QRunnable 异步任务
│
├── resources/                       # 静态资源
│   └── styles.qss                   # Qt 样式表
│
└── tests/                           # 测试
    ├── test_preprocessor.py
    ├── test_extractor.py
    ├── test_exporter.py
    └── fixtures/                    # 测试用图片
```

**分层原则：** `core/` 零 Qt 依赖，可独立测试和复用；`gui/` 只负责展示和交互调度。

---

## 八、配置管理

### config.yaml

```yaml
model:
  deepseek_ocr_path: "deepseek-ai/DeepSeek-OCR-2"
  device: "cuda"
  dtype: "bfloat16"
  flash_attention: true

preprocess:
  auto_rotate: true
  repair_edges: true
  normalize_background: true
  remove_stamps: true
  denoise: true
  enhance_contrast: true
  binarize: false

export:
  default_dir: "./output"
  excel_freeze_header: true

gui:
  thumbnail_size: 120
  default_zoom_fit: true
```

GUI 设置对话框修改后自动保存到 YAML，下次启动生效。

---

## 九、错误处理

### 错误分级

| 级别 | 场景 | 处理方式 |
|------|------|----------|
| 致命 | 模型加载失败、CUDA 不可用、内存不足 | 弹窗提示，应用不可用，引导检查环境 |
| 严重 | 单张图片 OCR 崩溃、推理超时 | 标记 failed，跳过继续处理下一张 |
| 警告 | 字段提取失败、JSON 解析出错 | 保存 raw_markdown，用户手动填写 |
| 信息 | 预处理某步骤跳过 | 状态栏提示，不影响流程 |

### 边界情况

| 情况 | 处理策略 |
|------|----------|
| 图片分辨率过低（<300dpi） | 预处理阶段检测并警告，尝试超分辨率放大 |
| 手写字迹完全无法辨认 | 字段留空，置信度标 0，不猜测 |
| 一张卡上有多个人的信息 | 暂不支持，标记为"需人工处理" |
| 非户籍卡图片误导入 | 提示用户确认，不强制处理 |
| 批量处理中途取消 | QThreadPool 取消待执行任务，已完成结果保留 |
| 模型首次加载耗时 | 启动时后台加载，显示进度条 |

### 模型懒加载

单例模式管理模型实例，整个应用共享一份 GPU 显存。懒加载避免启动时阻塞界面。

---

## 十、依赖清单

```
PySide6>=6.6
torch>=2.6.0
transformers>=4.46.3
accelerate
sentencepiece
flash-attn>=2.7.3
opencv-python>=4.8
numpy
Pillow
openpyxl
python-docx
reportlab
PyYAML
```
