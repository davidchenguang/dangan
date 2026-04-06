"""多轮 OCR 投票引擎 — 每轮使用不同图像增强 + 温度采样，对每个字段取众数

通过统计方法缓解 DeepSeek-OCR-2 的手写识别误差：
- 每轮对输入图像施加微小随机增强（亮度、对比度、模糊）
- 每轮使用递增 temperature（第1轮0.0、第2轮0.3、第3轮0.5...）
- 简单字段（str）：取 N 轮中出现次数最多的值
- 复合字段（BirthInfo 等）：对每个子属性独立投票
- 列表字段（ChangeRecord）：取所有轮次唯一记录的并集
"""

from __future__ import annotations

import logging
import tempfile
from collections import Counter
from dataclasses import fields as dataclass_fields
from pathlib import Path

from core.deduplicator import deduplicate_output
from core.field_extractor import FieldExtractor
from core.models import (
    ATTR_TO_LABEL,
    BirthInfo,
    ChangeRecord,
    CitizenCertificate,
    HouseholdCard,
    OccupationInfo,
    VotingResult,
)
from core.ocr_engine import OCREngine

logger = logging.getLogger(__name__)

# 简单字符串字段列表（直接投票）
_SIMPLE_FIELDS = [
    "name", "alias", "gender", "native_place", "ethnicity",
    "religion", "marital_status", "education", "other_residence",
    "relation_to_household_head",
]

# 复合字段列表（对子属性独立投票）
_COMPOSITE_FIELDS = {
    "birth": BirthInfo,
    "occupation": OccupationInfo,
    "citizen_cert": CitizenCertificate,
}

# 列表字段列表（取并集）
_LIST_FIELDS = [
    "migration_in", "migration_local", "cancellation", "changes",
]

# 每轮 temperature 模板：第1轮确定性，后续递增
_TEMPERATURE_SCHEDULE = [0.0, 0.3, 0.5, 0.6, 0.7]


class OcrVotingEngine:
    """多轮 OCR 投票引擎"""

    def __init__(
        self,
        engine: OCREngine,
        extractor: FieldExtractor | None = None,
        rounds: int = 3,
    ) -> None:
        self._engine = engine
        self._extractor = extractor or FieldExtractor()
        self._rounds = max(1, rounds)

    def recognize_with_voting(
        self,
        image_path: str,
        prompt: str,
    ) -> VotingResult:
        """执行 N 轮 OCR，对每个字段投票选出最可靠值

        每轮使用:
        - 不同的图像增强（seed=round_index）
        - 递增的 temperature（0.0 → 0.3 → 0.5）

        Args:
            image_path: 预处理后的图片路径
            prompt: OCR Prompt

        Returns:
            VotingResult 包含合并后的 HouseholdCard + 各字段置信度
        """
        raw_results: list[HouseholdCard] = []
        temp_files: list[str] = []  # 需要清理的临时增强图片

        try:
            for i in range(self._rounds):
                logger.info("投票轮次 %d/%d", i + 1, self._rounds)

                # 1. 图像增强（每轮不同 seed）
                aug_path = self._prepare_augmented_image(image_path, seed=i)
                if aug_path != image_path:
                    temp_files.append(aug_path)

                # 2. 递增 temperature
                temp = _TEMPERATURE_SCHEDULE[min(i, len(_TEMPERATURE_SCHEDULE) - 1)]

                # 3. OCR 识别
                raw_text = self._engine.recognize(aug_path, prompt, temperature=temp)

                if not raw_text or not raw_text.strip():
                    logger.warning("轮次 %d 输出为空，跳过", i + 1)
                    continue

                clean_text = deduplicate_output(raw_text)
                card = self._extractor.extract(clean_text)
                raw_results.append(card)

        finally:
            # 清理临时文件
            for f in temp_files:
                Path(f).unlink(missing_ok=True)

        if not raw_results:
            logger.warning("所有轮次均无有效输出")
            return VotingResult(
                card=HouseholdCard(),
                rounds=self._rounds,
            )

        # 单轮直接返回
        if len(raw_results) == 1:
            return VotingResult(
                card=raw_results[0],
                confidence={k: 1.0 for k in _SIMPLE_FIELDS},
                raw_results=raw_results,
                rounds=1,
            )

        # 多轮投票
        merged_card, confidence = self._merge_by_voting(raw_results)
        logger.info(
            "投票完成，%d 轮有效，平均置信度: %.2f",
            len(raw_results),
            sum(confidence.values()) / max(len(confidence), 1),
        )

        return VotingResult(
            card=merged_card,
            confidence=confidence,
            raw_results=raw_results,
            rounds=len(raw_results),
        )

    @staticmethod
    def _prepare_augmented_image(image_path: str, seed: int = 0) -> str:
        """为投票轮次准备增强图像

        Args:
            image_path: 原始图片路径
            seed: 随机种子（通常为轮次索引）

        Returns:
            增强后图片的临时文件路径（seed=0 时返回原路径）
        """
        if seed == 0:
            return image_path  # 第1轮不做增强

        import cv2
        from core.augmentation import augment_for_voting

        img = cv2.imread(image_path)
        if img is None:
            return image_path

        augmented = augment_for_voting(img, seed=seed)

        suffix = Path(image_path).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        cv2.imwrite(tmp.name, augmented)
        tmp.close()
        return tmp.name

    def _merge_by_voting(
        self,
        cards: list[HouseholdCard],
    ) -> tuple[HouseholdCard, dict[str, float]]:
        """对多轮结果执行字段级投票

        Returns:
            (merged_card, confidence_dict)
        """
        merged = HouseholdCard()
        confidence: dict[str, float] = {}
        n = len(cards)

        # 1. 简单字段投票
        for field_name in _SIMPLE_FIELDS:
            values = [getattr(c, field_name, "") for c in cards]
            values = [v for v in values if v]  # 过滤空值
            winner, conf = _vote(values, n)
            if winner:
                setattr(merged, field_name, winner)
            confidence[field_name] = conf

        # 2. 复合字段投票（对每个子属性独立投票）
        for attr_name, cls in _COMPOSITE_FIELDS.items():
            composite = cls()
            for sub_field in dataclass_fields(cls):
                values = []
                for c in cards:
                    composite_obj = getattr(c, attr_name, None)
                    if composite_obj:
                        val = getattr(composite_obj, sub_field.name, "")
                        if val:
                            values.append(val)

                winner, conf = _vote(values, n)
                if winner:
                    setattr(composite, sub_field.name, winner)

                full_key = f"{attr_name}.{sub_field.name}"
                confidence[full_key] = conf

            setattr(merged, attr_name, composite)

        # 3. 列表字段：取并集（去重）
        for list_field in _LIST_FIELDS:
            all_records: list[ChangeRecord] = []
            for c in cards:
                records = getattr(c, list_field, [])
                all_records.extend(records)

            unique_records = _deduplicate_records(all_records)
            setattr(merged, list_field, unique_records)

            # 置信度：记录出现频率
            if unique_records:
                confidence[list_field] = len(unique_records) / max(len(all_records), 1)
            else:
                confidence[list_field] = 0.0

        # 4. 保留元数据
        merged.source_image = cards[0].source_image if cards else ""
        merged.raw_markdown = cards[0].raw_markdown if cards else ""

        return merged, confidence


def _vote(values: list[str], total_rounds: int) -> tuple[str, float]:
    """对一组值做多数投票

    Args:
        values: 非空值列表
        total_rounds: 总轮数

    Returns:
        (winner_value, confidence)
        - winner_value: 出现次数最多的值（空字符串表示无有效值）
        - confidence: 众数次数 / total_rounds
    """
    if not values:
        return "", 0.0

    counter = Counter(values)
    winner, count = counter.most_common(1)[0]
    return winner, count / total_rounds


def _deduplicate_records(records: list[ChangeRecord]) -> list[ChangeRecord]:
    """对 ChangeRecord 列表去重（基于 content 相等）"""
    seen: set[str] = set()
    unique: list[ChangeRecord] = []
    for r in records:
        key = f"{r.date}|{r.content}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique
