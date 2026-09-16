"""冲突消解模块的单元测试（FakeLLM 注入 + 固定时钟，全离线）。

覆盖：指数衰减公式、时区处理、三路决策（覆盖/保留/融合）、
阈值边界（经时间衰减构造，规避浮点减法陷阱）、降级路径。
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import math

import pytest

from app.memory.conflict_resolver import (
    DEFAULT_LAMBDA,
    FUSE_THRESHOLD,
    LLMNotConfiguredError,
    _ensure_utc,
    bayesian_update,
    effective_confidence,
    resolve_conflict,
)
from app.memory.models import MemoryFact

# 固定时钟：所有测试在 T_NOW 这一时刻进行，结果可复现
T_NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

FUSED_TEXT = "融合记忆：用户当前偏好的编辑器是 VS Code（此前为 PyCharm）"


class FakeLLM:
    """测试替身：记录调用次数与消息，返回固定融合文本。"""

    def __init__(self, response: str = FUSED_TEXT) -> None:
        self.response = response
        self.calls = 0
        self.last_messages = None

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        return SimpleNamespace(content=self.response)


def make_fact(content, confidence=0.5, age_days=0.0, tz=timezone.utc):
    """构造指定年龄的测试记忆。"""
    return MemoryFact(
        content=content,
        confidence=confidence,
        timestamp=T_NOW - timedelta(days=age_days),
    ) if tz is timezone.utc else MemoryFact(
        content=content,
        confidence=confidence,
        timestamp=(T_NOW - timedelta(days=age_days)).astimezone(tz),
    )


class TestEffectiveConfidence:
    """指数衰减公式测试。"""

    def test_fresh_memory_undecayed(self) -> None:
        """刚写入的记忆（年龄 0）有效置信度等于初始值。"""
        fact = make_fact("新记忆", confidence=0.8, age_days=0)
        assert effective_confidence(fact, DEFAULT_LAMBDA, now=T_NOW) == pytest.approx(0.8)

    def test_exponential_decay_value(self) -> None:
        """衰减公式：eff = c * exp(-lambda * days)。"""
        fact = make_fact("旧记忆", confidence=0.9, age_days=10)
        expected = 0.9 * math.exp(-0.01 * 10)
        assert effective_confidence(fact, 0.01, now=T_NOW) == pytest.approx(expected)

    def test_half_life_interpretation(self) -> None:
        """lambda=0.01 时约 69.3 天衰减一半（半衰期 = ln2/lambda）。"""
        fact = make_fact("半衰记忆", confidence=1.0, age_days=math.log(2) / 0.01)
        assert effective_confidence(fact, 0.01, now=T_NOW) == pytest.approx(0.5)

    def test_future_timestamp_clamped_to_zero_decay(self) -> None:
        """未来时间戳按 0 衰减处理，不得产生置信度增益。"""
        future = MemoryFact(
            content="来自未来的记忆",
            confidence=0.7,
            timestamp=T_NOW + timedelta(days=5),
        )
        assert effective_confidence(future, 0.01, now=T_NOW) == pytest.approx(0.7)

    def test_negative_lambda_rejected(self) -> None:
        """负衰减系数意味着时间越久越可信，公式上无意义。"""
        fact = make_fact("记忆")
        with pytest.raises(ValueError):
            effective_confidence(fact, -0.01, now=T_NOW)

    def test_non_memory_fact_rejected(self) -> None:
        with pytest.raises(TypeError):
            effective_confidence("不是记忆", 0.01, now=T_NOW)

    def test_naive_timestamp_treated_as_utc(self) -> None:
        """naive datetime 等价于同值 UTC（默认 0 衰减路径不抛错）。"""
        naive = MemoryFact(
            content="无时区记忆",
            confidence=0.6,
            timestamp=datetime(2026, 9, 15, 12, 0, 0),  # naive
        )
        assert effective_confidence(naive, 0.01, now=T_NOW) == pytest.approx(0.6)

    def test_non_utc_timezone_normalized(self) -> None:
        """UTC+8 时区与等效 UTC 时刻给出相同有效置信度。"""
        cst = timezone(timedelta(hours=8))
        aware = MemoryFact(
            content="东八区记忆",
            confidence=0.6,
            timestamp=T_NOW.astimezone(cst),
        )
        assert effective_confidence(aware, 0.01, now=T_NOW) == pytest.approx(0.6)

    def test_ensure_utc_handles_both_kinds(self) -> None:
        """_ensure_utc：naive 补 UTC，aware 转 UTC。"""
        naive = datetime(2026, 9, 15, 12, 0, 0)
        assert _ensure_utc(naive).tzinfo == timezone.utc

        cst = timezone(timedelta(hours=8))
        converted = _ensure_utc(datetime(2026, 9, 15, 20, 0, 0, tzinfo=cst))
        assert converted == datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestResolveThreePaths:
    """三路决策测试（用时间衰减构造差值，规避浮点边界）。"""

    def test_new_clearly_higher_replaces_old(self) -> None:
        """新记忆显著更强 -> replaced，旧记忆降级入历史。"""
        old = make_fact("旧记忆", confidence=0.4, age_days=0)
        new = make_fact("新记忆", confidence=0.9, age_days=0)  # 差 0.5 > 0.1
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.action == "replaced"
        assert result.primary.content == "新记忆"
        assert result.primary.id == new.id
        assert [h.content for h in result.history] == ["旧记忆"]

    def test_old_clearly_higher_keeps_with_history(self) -> None:
        """旧记忆显著更强 -> 保留旧记忆，新记忆入历史。"""
        old = make_fact("旧记忆", confidence=0.9, age_days=0)
        new = make_fact("新记忆", confidence=0.4, age_days=0)  # 差 0.5
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.action == "kept_with_history"
        assert result.primary.id == old.id
        assert [h.content for h in result.history] == ["新记忆"]

    def test_close_confidences_trigger_fusion(self) -> None:
        """有效置信度接近（差 < 0.1）-> LLM 融合。"""
        old = make_fact("偏好 PyCharm", confidence=0.9, age_days=0)
        new = make_fact("偏好 VS Code", confidence=0.85, age_days=0)  # 差 0.05
        fake = FakeLLM()
        result = resolve_conflict(old, new, llm_client=fake, now=T_NOW)
        assert result.action == "fused"
        assert fake.calls == 1
        assert result.primary.content == FUSED_TEXT
        # 融合记忆是新生成的事实
        assert result.primary.id not in {old.id, new.id}
        # 两条原始记忆都入历史（信息不灭）
        assert {h.id for h in result.history} == {old.id, new.id}

    def test_decay_flips_decision(self) -> None:
        """时间衰减逆转裁决：新记忆初始置信度高但年龄大 -> 旧记忆胜出。

        eff_new = 0.9 * exp(-0.01 * 200) ~ 0.122 < eff_old = 0.6（新）。
        """
        old = make_fact("旧记忆", confidence=0.6, age_days=0)
        new = make_fact("新记忆", confidence=0.9, age_days=200)
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.action == "kept_with_history"
        assert result.primary.id == old.id

    def test_boundary_via_decay_just_below_threshold_fuses(self) -> None:
        """衰减后差值 0.0906 < 0.1 -> 融合。"""
        old = make_fact("甲", confidence=0.5, age_days=0)
        new = make_fact("乙", confidence=0.5, age_days=20)  # 0.5*exp(-0.2)~0.4094
        assert 0.5 - 0.5 * math.exp(-0.2) < FUSE_THRESHOLD
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.action == "fused"

    def test_boundary_via_decay_just_above_threshold_keeps(self) -> None:
        """衰减后差值 0.1296 > 0.1 -> 旧记忆保留。"""
        old = make_fact("甲", confidence=0.5, age_days=0)
        new = make_fact("乙", confidence=0.5, age_days=30)  # 0.5*exp(-0.3)~0.3704
        assert 0.5 - 0.5 * math.exp(-0.3) > FUSE_THRESHOLD
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.action == "kept_with_history"


class TestFusionBehavior:
    """LLM 融合细节测试。"""

    def test_fused_confidence_is_mean_of_initials(self) -> None:
        """融合记忆 confidence = 两条初始置信度均值（不继承年龄惩罚）。"""
        old = make_fact("甲", confidence=0.9, age_days=0)
        new = make_fact("乙", confidence=0.85, age_days=0)
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.primary.confidence == pytest.approx(0.875)

    def test_fused_timestamp_is_fusion_time(self) -> None:
        """融合记忆 timestamp = 融合发生时刻（固定时钟 T_NOW）。"""
        old = make_fact("甲", confidence=0.9, age_days=0)
        new = make_fact("乙", confidence=0.85, age_days=0)
        result = resolve_conflict(old, new, llm_client=FakeLLM(), now=T_NOW)
        assert result.primary.timestamp == T_NOW

    def test_fuse_prompt_carries_both_contents(self) -> None:
        """融合 Prompt 必须包含两条记忆原文。"""
        old = make_fact("偏好 PyCharm", confidence=0.9, age_days=0)
        new = make_fact("偏好 VS Code", confidence=0.85, age_days=0)
        fake = FakeLLM()
        resolve_conflict(old, new, llm_client=fake, now=T_NOW)
        text = fake.last_messages[0]["content"]
        assert "偏好 PyCharm" in text
        assert "偏好 VS Code" in text

    def test_fusion_without_client_raises(self) -> None:
        """需要融合但无客户端 -> 明确报错（不可静默裁决）。"""
        old = make_fact("甲", confidence=0.9, age_days=0)
        new = make_fact("乙", confidence=0.85, age_days=0)
        with pytest.raises(LLMNotConfiguredError):
            resolve_conflict(old, new, llm_client=None, now=T_NOW)

    def test_empty_llm_response_degrades_to_stronger_memory(self) -> None:
        """LLM 返回空文本 -> 降级保留有效置信度较高一方。"""
        old = make_fact("甲", confidence=0.85, age_days=0)
        new = make_fact("乙", confidence=0.9, age_days=0)  # 差 0.05 -> 融合路径
        fake = FakeLLM(response="   ")
        result = resolve_conflict(old, new, llm_client=fake, now=T_NOW)
        assert result.action == "fused"  # 动作仍是融合（已尽力）
        assert result.primary.content == "乙"  # 保留较强一方
        assert [h.content for h in result.history] == ["甲"]

    def test_non_text_llm_response_degrades(self) -> None:
        """LLM 返回非文本同样触发降级。"""
        old = make_fact("甲", confidence=0.9, age_days=0)
        new = make_fact("乙", confidence=0.85, age_days=0)
        weird = FakeLLM()
        weird.invoke = lambda messages: {"oops": 1}
        result = resolve_conflict(old, new, llm_client=weird, now=T_NOW)
        assert result.primary.content == "甲"


class TestInputValidation:
    """输入校验测试。"""

    def test_non_memory_inputs_rejected(self) -> None:
        with pytest.raises(TypeError):
            resolve_conflict("旧", make_fact("新"), llm_client=FakeLLM())
        with pytest.raises(TypeError):
            resolve_conflict(make_fact("旧"), None, llm_client=FakeLLM())

    def test_negative_lambda_rejected(self) -> None:
        with pytest.raises(ValueError):
            resolve_conflict(
                make_fact("甲", 0.9), make_fact("乙", 0.85),
                lambda_val=-0.01, llm_client=FakeLLM(),
            )


class TestBayesianUpdateStillAvailable:
    """确认既有的贝叶斯更新函数未被破坏（回归保护）。"""

    def test_import_and_basic_property(self) -> None:
        assert bayesian_update(0.5, 4.0) > 0.5
        with pytest.raises(ValueError):
            bayesian_update(1.0, 2.0)
