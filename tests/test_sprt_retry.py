"""SPRT 重试止损单元测试（with_retry 的统计优化运行时路径）。"""

import pytest

from app.agent.self_correction import with_retry
from app.core.config import settings


@pytest.fixture
def sprt_on(monkeypatch):
    """基准实验同款 SPRT 参数：p0=0.5, p1=0.05（5 次连续失败触发止损）。"""
    monkeypatch.setattr(settings, "sprt_retry_enabled", True)
    monkeypatch.setattr(settings, "sprt_retry_p0", 0.5)
    monkeypatch.setattr(settings, "sprt_retry_p1", 0.05)


class TestSprtEarlyStop:
    def test_hopeless_tool_stops_early(self, sprt_on, tmp_logger) -> None:
        """无望工具：SPRT 在第 5 次失败后止损（而非烧满 11 次）。"""
        calls = {"n": 0}

        @with_retry(max_retries=10, task_id="t-hopeless", replan_llm=None, failure_logger=tmp_logger)
        def always_fail() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            always_fail()
        assert calls["n"] == 5  # log(18)/log(1.9) -> 5 次失败推过 Wald 上界
        records = tmp_logger.read_all()
        assert len(records) == 5
        assert all(r["final_success"] is False for r in records)

    def test_disabled_burns_all_retries(self, monkeypatch, tmp_logger) -> None:
        """SPRT 关闭（对照组 A）：烧满 max_retries。"""
        monkeypatch.setattr(settings, "sprt_retry_enabled", False)
        calls = {"n": 0}

        @with_retry(max_retries=10, task_id="t-fixed", replan_llm=None, failure_logger=tmp_logger)
        def always_fail() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            always_fail()
        assert calls["n"] == 11  # 1 次初始 + 10 次重试

    def test_never_exceeds_max_retries(self, sprt_on, tmp_logger) -> None:
        """SPRT 只会提前停：小预算下不越权（3 次尝试内停不下来就烧满）。"""
        calls = {"n": 0}

        @with_retry(max_retries=2, task_id="t-cap", replan_llm=None, failure_logger=tmp_logger)
        def always_fail() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            always_fail()
        assert calls["n"] == 3  # 2 次失败时 log_lr 仍未到上界 -> 正常烧满

    def test_transient_tool_unaffected(self, sprt_on, tmp_logger) -> None:
        """瞬态故障：第一次失败不足以触发止损，第二次恢复。"""
        calls = {"n": 0}

        @with_retry(max_retries=10, task_id="t-transient", replan_llm=None, failure_logger=tmp_logger)
        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("glitch")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 2

    def test_invalid_params_degrade_to_full_retry(self, monkeypatch, tmp_logger) -> None:
        """p1 >= p0 配置非法 -> 本次链禁用止损（fail-open），不炸不误停。"""
        monkeypatch.setattr(settings, "sprt_retry_enabled", True)
        monkeypatch.setattr(settings, "sprt_retry_p0", 0.2)
        monkeypatch.setattr(settings, "sprt_retry_p1", 0.5)  # 非法：p1 > p0
        calls = {"n": 0}

        @with_retry(max_retries=2, task_id="t-bad", replan_llm=None, failure_logger=tmp_logger)
        def always_fail() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            always_fail()
        assert calls["n"] == 3
