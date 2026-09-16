"""MemoryFact 数据模型的单元测试。

覆盖：默认值生成、字段约束（confidence 边界 / content 非空）、
时区归一化、embedding 有限性校验、序列化往返。
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.memory.models import MemoryFact


class TestMemoryFactDefaults:
    """默认值行为测试。"""

    def test_default_id_is_valid_uuid(self) -> None:
        """未提供 id 时应自动生成合法 UUID4。"""
        fact = MemoryFact(content="测试记忆")
        import uuid

        # UUID 构造不抛异常即为合法
        uuid.UUID(fact.id)

    def test_default_ids_are_unique(self) -> None:
        """两次实例化的 id 必须不同。"""
        fact_a = MemoryFact(content="记忆甲")
        fact_b = MemoryFact(content="记忆乙")
        assert fact_a.id != fact_b.id

    def test_default_timestamp_is_utc_and_recent(self) -> None:
        """默认 timestamp 必须是 UTC 时区感知且接近当前时间。"""
        before = datetime.now(timezone.utc)
        fact = MemoryFact(content="测试记忆")
        after = datetime.now(timezone.utc)

        assert fact.timestamp.tzinfo is not None
        assert before <= fact.timestamp <= after

    def test_default_confidence_is_uninformative_prior(self) -> None:
        """默认置信度应为 0.5（最大熵 / 无信息先验）。"""
        fact = MemoryFact(content="测试记忆")
        assert fact.confidence == 0.5

    def test_default_embedding_is_none(self) -> None:
        """默认不携带向量。"""
        fact = MemoryFact(content="测试记忆")
        assert fact.embedding is None


class TestMemoryFactConstraints:
    """字段约束测试。"""

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_boundary_values_accepted(self, confidence: float) -> None:
        """confidence 边界值 0.0 与 1.0 应被接受（闭区间）。"""
        fact = MemoryFact(content="测试", confidence=confidence)
        assert fact.confidence == confidence

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -1.0])
    def test_confidence_out_of_range_rejected(self, confidence: float) -> None:
        """confidence 越界必须抛 ValidationError。"""
        with pytest.raises(ValidationError):
            MemoryFact(content="测试", confidence=confidence)

    def test_empty_content_rejected(self) -> None:
        """空内容必须被拒绝。"""
        with pytest.raises(ValidationError):
            MemoryFact(content="")

    def test_naive_timestamp_coerced_to_utc(self) -> None:
        """naive datetime 应被自动补 UTC 时区。"""
        naive = datetime(2026, 9, 15, 12, 0, 0)  # 无时区
        fact = MemoryFact(content="测试", timestamp=naive)
        assert fact.timestamp.tzinfo is not None
        assert fact.timestamp.utcoffset() == timedelta(0)

    def test_nan_in_embedding_rejected(self) -> None:
        """embedding 含 NaN 必须被拒绝（污染相似度计算）。"""
        with pytest.raises(ValidationError):
            MemoryFact(content="测试", embedding=[0.1, float("nan")])

    def test_inf_in_embedding_rejected(self) -> None:
        """embedding 含 Inf 必须被拒绝。"""
        with pytest.raises(ValidationError):
            MemoryFact(content="测试", embedding=[0.1, float("inf")])

    def test_valid_embedding_accepted(self) -> None:
        """合法有限向量应被接受。"""
        fact = MemoryFact(content="测试", embedding=[0.1, -0.2, 0.3])
        assert fact.embedding == [0.1, -0.2, 0.3]


class TestMemoryFactSerialization:
    """序列化行为测试。"""

    def test_roundtrip_preserves_all_fields(self) -> None:
        """model_dump -> 构造 的往返不应丢失或篡改字段。"""
        original = MemoryFact(
            content="用户偏好 pytest",
            confidence=0.85,
            embedding=[0.01, -0.02],
        )
        data = original.model_dump(mode="json")
        restored = MemoryFact.model_validate(data)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.confidence == original.confidence
        assert restored.timestamp == original.timestamp
        assert restored.embedding == original.embedding
