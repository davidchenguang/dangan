"""测试 core/voting.py — 多轮投票引擎"""
from unittest.mock import MagicMock

import pytest

from core.models import (
    BirthInfo,
    ChangeRecord,
    HouseholdCard,
    OccupationInfo,
    VotingResult,
)
from core.voting import OcrVotingEngine, _vote, _deduplicate_records


class TestVoteFunction:
    """_vote() 多数投票函数"""

    def test_majority_wins(self):
        winner, conf = _vote(["A", "A", "B"], 3)
        assert winner == "A"
        assert conf == pytest.approx(2 / 3, abs=0.01)

    def test_unanimous(self):
        winner, conf = _vote(["X", "X", "X"], 3)
        assert winner == "X"
        assert conf == 1.0

    def test_empty_values(self):
        winner, conf = _vote([], 3)
        assert winner == ""
        assert conf == 0.0

    def test_single_value(self):
        winner, conf = _vote(["only"], 3)
        assert winner == "only"
        assert conf == pytest.approx(1 / 3, abs=0.01)

    def test_tie_takes_first_encountered(self):
        """平票时取 Counter.most_common 第一个"""
        winner, conf = _vote(["A", "B"], 2)
        assert winner in ("A", "B")
        assert conf == 0.5


class TestDeduplicateRecords:
    """ChangeRecord 去重"""

    def test_removes_exact_duplicates(self):
        records = [
            ChangeRecord(date="1953年", content="迁入"),
            ChangeRecord(date="1953年", content="迁入"),
        ]
        result = _deduplicate_records(records)
        assert len(result) == 1

    def test_keeps_different_records(self):
        records = [
            ChangeRecord(date="1953年", content="迁入"),
            ChangeRecord(date="1960年", content="迁出"),
        ]
        result = _deduplicate_records(records)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate_records([]) == []


class TestOcrVotingEngineMerge:
    """_merge_by_voting 字段合并逻辑"""

    def _make_engine(self):
        mock_engine = MagicMock()
        return OcrVotingEngine(engine=mock_engine, rounds=3)

    def test_simple_field_voting(self):
        """简单字段取众数"""
        cards = [
            HouseholdCard(name="张三", gender="男"),
            HouseholdCard(name="张三", gender="男"),
            HouseholdCard(name="李四", gender="男"),
        ]
        engine = self._make_engine()
        merged, confidence = engine._merge_by_voting(cards)
        assert merged.name == "张三"
        assert confidence["name"] == pytest.approx(2 / 3, abs=0.01)

    def test_composite_field_voting(self):
        """复合字段对子属性独立投票"""
        cards = [
            HouseholdCard(birth=BirthInfo(date="1950年", address="北京")),
            HouseholdCard(birth=BirthInfo(date="1950年", address="上海")),
            HouseholdCard(birth=BirthInfo(date="1960年", address="北京")),
        ]
        engine = self._make_engine()
        merged, confidence = engine._merge_by_voting(cards)
        assert merged.birth.date == "1950年"
        assert merged.birth.address == "北京"

    def test_list_field_union(self):
        """列表字段取并集"""
        cards = [
            HouseholdCard(migration_in=[ChangeRecord(content="A")]),
            HouseholdCard(migration_in=[ChangeRecord(content="B")]),
        ]
        engine = self._make_engine()
        merged, _ = engine._merge_by_voting(cards)
        assert len(merged.migration_in) == 2

    def test_empty_card_list_handled(self):
        """空卡片列表不崩溃"""
        engine = self._make_engine()
        merged, confidence = engine._merge_by_voting([
            HouseholdCard(),
            HouseholdCard(),
        ])
        assert merged.name == ""


class TestOcrVotingEngineIntegration:
    """recognize_with_voting 集成测试"""

    def test_calls_recognize_n_times(self):
        """应调用 recognize N 次"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三\n**性别**：男"

        voter = OcrVotingEngine(engine=mock_engine, rounds=3)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert mock_engine.recognize.call_count == 3
        assert isinstance(result, VotingResult)

    def test_returns_merged_card(self):
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三\n**性别**：男"

        voter = OcrVotingEngine(engine=mock_engine, rounds=2)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.cards[0].name == "张三"
        assert result.rounds == 2

    def test_all_empty_outputs(self):
        """所有轮次空输出时返回空卡片"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = ""

        voter = OcrVotingEngine(engine=mock_engine, rounds=2)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.cards[0].name == ""
        assert result.rounds == 2

    def test_single_round_skips_voting(self):
        """单轮直接返回不做投票合并"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三"

        voter = OcrVotingEngine(engine=mock_engine, rounds=1)
        result = voter.recognize_with_voting("fake.jpg", "prompt")

        assert result.cards[0].name == "张三"
        assert result.rounds == 1
        assert all(v == 1.0 for v in result.confidence.values())

    def test_temperature_passed_per_round(self):
        """每轮应传递不同 temperature"""
        mock_engine = MagicMock()
        mock_engine.recognize.return_value = "**姓名**：张三"

        voter = OcrVotingEngine(engine=mock_engine, rounds=3)
        voter.recognize_with_voting("fake.jpg", "prompt")

        calls = mock_engine.recognize.call_args_list
        # 第1轮 temperature=0.0, 第2轮=0.3, 第3轮=0.5
        assert calls[0].kwargs.get("temperature") == 0.0
        assert calls[1].kwargs.get("temperature") == 0.3
        assert calls[2].kwargs.get("temperature") == 0.5
