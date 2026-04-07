"""数据结构定义 — 户籍卡 OCR 系统所有核心数据模型

数据流: 图片 → HouseholdCard → ResultExporter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProcessingStatus(Enum):
    """图片处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


# ── 复合字段结构 ──


@dataclass
class BirthInfo:
    """出生信息"""
    date: str = ""          # 日期
    address: str = ""       # 地址


@dataclass
class CitizenCertificate:
    """公民证信息"""
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
    raw_markdown: str = ""            # OCR 原始输出
    confidence: dict[str, float] = field(default_factory=dict)
    reviewed: bool = False


# ── 预处理配置 ──


@dataclass
class PreprocessConfig:
    """预处理参数配置

    经实验验证的最佳策略:
    - 2x放大 + CLAHE(clipLimit=3.0) + 不去噪 + crop_mode=True
    - 去噪会破坏笔画细节，默认关闭
    - 二值化对 DeepSeek-OCR-2 效果不佳，默认关闭
    """
    scale: float = 2.0               # 放大倍数
    clahe_clip: float = 3.0          # CLAHE 对比度限制
    auto_rotate: bool = False        # 倾斜校正
    repair_edges: bool = False       # 边缘修补
    normalize_background: bool = False  # 背景归一化（去黄）
    remove_stamps: bool = False      # 印章去除
    denoise: bool = False            # 去噪（默认关闭，破坏笔画）
    enhance_contrast: bool = True    # 对比度增强 (CLAHE)
    binarize: bool = False           # 二值化（默认关闭）


# ── 字段区域（版面分析用）──


@dataclass
class FieldRegion:
    """表格中一个字段的位置区域"""
    x: int
    y: int
    w: int
    h: int
    label: str = ""


# ── Prompt 模板 ──

# 最佳 prompt（参考 DeepSeek-OCR-2 Demo "Free OCR" 模式）
# 简洁直接，避免模型输出 HTML/结构化格式
# 注意：必须包含 <image> 标签，模型的 infer() 依赖此标签定位图像 token 插入位置
PROMPT_VERBATIM = "<image>\n请逐字精确识别这张户籍登记表扫描件中的所有文字内容，包括手写和印刷文字。"

# 结构化字段提取 prompt（推荐）— 显式列出字段名，要求逐行输出
# 不使用 <|grounding|>，避免模型按物理布局输出
# 注意：必须包含 <image> 标签，模型的 infer() 依赖此标签定位图像 token 插入位置
PROMPT_STRUCTURED = (
    "<image>\n"
    "请识别这张户籍登记表中每个手写字段的内容。\n"
    "严格按以下格式逐行输出，每行一个「字段名：值」。看不清的字段留空。\n"
    "户主或与户主关系：\n"
    "姓名：\n"
    "别名：\n"
    "性别：\n"
    "出生日期：\n"
    "出生地址：\n"
    "籍贯：\n"
    "民族：\n"
    "宗教信仰：\n"
    "婚姻状况：\n"
    "文化程度：\n"
    "职业：\n"
    "服务处所：\n"
    "本市其他住所：\n"
    "公民证代号号码：\n"
    "签发机关：\n"
    "签发日期：\n"
    "何时由何地迁来本市：\n"
    "何时由本市何处迁来本地：\n"
    "注销户口日期和原因：\n"
    "户口登记事项变更更正记载：\n"
)

# Markdown 表格 prompt — 使用 <|grounding|> 提升结构化输出质量
PROMPT_MARKDOWN = (
    "<image>\n<|grounding|>请将这张户籍登记表的内容转换为 markdown 表格。"
    "只使用 markdown 表格格式（| 字段 | 值 |），不要使用 HTML 标签。"
    "请仔细识别每一个手写填写的文字，不要遗漏。如果某个字段看不清，请标注[模糊]。"
)

# JSON prompt（强制结构化输出）— <|grounding|> 已启用
CARD_OCR_PROMPT = (
    "<image>\n"
    "<|grounding|>这是50年代常住人口登记表的扫描件，黄色底色，包含手写和印刷文字。\n"
    "请仔细识别表格中每个字段的手写内容，严格按以下 JSON 格式输出。\n"
    "要求：\n"
    "1. 只输出 JSON，不要输出任何其他文字说明\n"
    "2. 字段为空或看不清则填空字符串\n"
    "3. 不要重复输出同一个字段\n"
    "{\n"
    '  "户主或与户主关系": "",\n'
    '  "姓名": "", "别名": "", "性别": "",\n'
    '  "出生日期": "", "出生地址": "",\n'
    '  "籍贯": "", "民族": "", "宗教信仰": "",\n'
    '  "婚姻状况": "", "文化程度": "",\n'
    '  "职业": "", "服务处所": "",\n'
    '  "本市其他住所": "",\n'
    '  "公民证代号号码": "", "签发机关": "", "签发日期": "",\n'
    '  "何时由何地迁来本市": "",\n'
    '  "何时由本市何处迁来本地": "",\n'
    '  "注销户口日期和原因": "",\n'
    '  "户口登记事项变更更正记载": ""\n'
    "}\n"
    "忽略印章、污渍和装订痕迹，只识别正式填写的文字内容。"
)

# 英文 prompt
PROMPT_ENGLISH = (
    "<image>\n"
    "Please perform OCR on this Chinese household registration card from the 1950s. "
    "The card has yellow background with both printed headers and handwritten content.\n"
    "Please transcribe ALL text you can see, both printed and handwritten. "
    "Output each field name and its value. Focus especially on the handwritten text in each field."
)


# ── 投票结果 ──


@dataclass
class VotingResult:
    """多轮投票识别结果"""
    cards: list["HouseholdCard"] = field(default_factory=list)   # 投票后的合并结果（多列支持）
    confidence: dict[str, float] = field(default_factory=dict)  # 字段名→置信度
    raw_results: list["HouseholdCard"] = field(default_factory=list)  # 各轮结果
    rounds: int = 0                              # 实际执行轮数

    @property
    def card(self) -> "HouseholdCard":
        """向后兼容：返回第一个卡片（单列模式）"""
        return self.cards[0] if self.cards else HouseholdCard()


# ── 字段名映射（中文 → HouseholdCard 属性）──

FIELD_TO_ATTR: dict[str, str] = {
    "户主或与户主关系": "relation_to_household_head",
    "姓名": "name",
    "别名": "alias",
    "性别": "gender",
    "出生日期": "birth.date",
    "出生地址": "birth.address",
    "籍贯": "native_place",
    "民族": "ethnicity",
    "宗教信仰": "religion",
    "婚姻状况": "marital_status",
    "文化程度": "education",
    "职业": "occupation.occupation",
    "服务处所": "occupation.workplace",
    "本市其他住所": "other_residence",
    "公民证代号号码": "citizen_cert.code_number",
    "签发机关": "citizen_cert.issuing_authority",
    "签发日期": "citizen_cert.issue_date",
    "何时由何地迁来本市": "migration_in",
    "何时由本市何处迁来本地": "migration_local",
    "注销户口日期和原因": "cancellation",
    "户口登记事项变更更正记载": "changes",
}

# 反向映射（属性 → 中文标签）
ATTR_TO_LABEL: dict[str, str] = {v: k for k, v in FIELD_TO_ATTR.items()}
