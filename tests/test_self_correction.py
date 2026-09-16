"""agent/self_correction.py 自纠错装饰器（with_retry + Replan）单元测试。

全程离线：Replan LLM 用 FakeLLM 注入，日志器注入 tmp_path 临时路径。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.self_correction import (  # noqa: E402
    FailureLogger,
    with_retry,
)


class ReplanFakeLLM:
    """Replan FakeLLM：按脚本逐次返回响应。"""

    def __init__(self, responses=None) -> None:
        self.responses = responses or []
        self.prompts: list[str] = []

    def invoke(self, messages):
        combined = " ".join(
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in messages
        )
        self.prompts.append(combined)
        if not self.responses:
            return SimpleNamespace(content="（无可用响应）")
        idx = min(len(self.prompts) - 1, len(self.responses) - 1)
        payload = self.responses[idx]
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return SimpleNamespace(content=payload)
        return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


@pytest.fixture
def tmp_logger(tmp_path):
    return FailureLogger(str(tmp_path / "failure_log.json"))


# ============================ 基本重试行为 ============================


class TestWithRetryBasics:

    def test_no_failure_no_log(self, tmp_logger) -> None:
        """一次成功：不调 Replan、不写日志。"""
        calls = []

        @with_retry(max_retries=3, task_id="t-ok", replan_llm=ReplanFakeLLM(),
                    failure_logger=tmp_logger)
        def ok() -> str:
            calls.append(1)
            return "fine"

        assert ok() == "fine"
        assert len(calls) == 1
        assert tmp_logger.read_all() == []

    def test_fail_then_succeed(self, tmp_logger) -> None:
        """失败 2 次第 3 次成功：日志 2 条且 final_success=True。"""
        attempts = {"n": 0}

        @with_retry(max_retries=3, task_id="t-retry", replan_llm=ReplanFakeLLM(),
                    failure_logger=tmp_logger)
        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError(f"第{attempts['n']}次抖动")
            return "finally"

        assert flaky() == "finally"
        records = tmp_logger.read_all()
        assert len(records) == 2
        assert [r["failed_step"] for r in records] == [1, 2]
        assert [r["retry_count"] for r in records] == [0, 1]
        assert all(r["final_success"] is True for r in records)
        assert all(r["task_id"] == "t-retry" for r in records)
        assert "第1次抖动" in records[0]["error_reason"]

    def test_exhausted_reraises_original(self, tmp_logger) -> None:
        """重试耗尽：重新抛出最后一次异常，日志 final_success=False。"""
        attempts = {"n": 0}

        @with_retry(max_retries=3, task_id="t-doom", replan_llm=ReplanFakeLLM(),
                    failure_logger=tmp_logger)
        def always_fail() -> None:
            attempts["n"] += 1
            raise RuntimeError(f"崩溃 #{attempts['n']}")

        with pytest.raises(RuntimeError, match="崩溃 #4"):
            always_fail()  # 1 次初始 + 3 次重试 = 4 次尝试

        assert attempts["n"] == 4
        records = tmp_logger.read_all()
        assert len(records) == 4
        assert all(r["final_success"] is False for r in records)
        assert [r["failed_step"] for r in records] == [1, 2, 3, 4]

    def test_wraps_preserves_metadata(self, tmp_logger) -> None:
        """functools.wraps 保留函数元数据。"""

        @with_retry(max_retries=2, failure_logger=tmp_logger)
        def documented_tool(x: int) -> int:
            """工具文档。"""
            return x

        assert documented_tool.__name__ == "documented_tool"
        assert documented_tool.__doc__ == "工具文档。"

    def test_invalid_max_retries_rejected(self) -> None:
        with pytest.raises(ValueError):
            with_retry(max_retries=0)

    def test_default_task_id_uses_func_qualname(self, tmp_logger) -> None:
        """未指定 task_id 时默认取「模块.函数名#短UUID」。"""

        @with_retry(max_retries=1, replan_llm=ReplanFakeLLM(), failure_logger=tmp_logger)
        def named_tool() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            named_tool()
        record = tmp_logger.read_all()[0]
        # qualname 对类内局部函数会带 <locals>，只断言关键组成部分
        assert "named_tool#" in record["task_id"]
        assert len(record["task_id"].split("#")[1]) == 8


# ============================ Replan 机制 ============================


class TestReplan:

    def test_replan_prompt_contains_error_and_history(self, tmp_logger) -> None:
        """Prompt 拼入错误信息与历史执行轨迹（用户契约）。"""
        llm = ReplanFakeLLM(responses=[{"analysis": "参数错", "revised_params": None, "reason": "无修正"}])
        attempts = {"n": 0}

        @with_retry(max_retries=2, task_id="t-prompt", replan_llm=llm, failure_logger=tmp_logger)
        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError(f"抖动{attempts['n']}")
            return "ok"

        assert flaky() == "ok"
        # 第一次 Replan：轨迹为「无」；第二次：包含第一次失败记录
        assert len(llm.prompts) == 2
        assert "抖动1" in llm.prompts[0]
        assert "无，本次为首次失败" in llm.prompts[0]
        assert "抖动2" in llm.prompts[1]
        assert "第1次尝试失败" in llm.prompts[1]

    def test_revised_params_change_outcome(self, tmp_logger) -> None:
        """LLM 修正参数后重试成功：修正覆盖原 kwargs。"""
        llm = ReplanFakeLLM(responses=[
            {"analysis": "除数为零", "revised_params": {"denominator": 4}, "reason": "换非零除数"},
        ])

        @with_retry(max_retries=2, task_id="t-fix", replan_llm=llm, failure_logger=tmp_logger)
        def divide(numerator: int, denominator: int) -> float:
            return numerator / denominator

        assert divide(8, denominator=0) == 2.0

    def test_action_snapshot_is_pre_replan(self, tmp_logger) -> None:
        """回归：失败记录的 action 必须是失败当时的调用（Replan 合并前）。

        此前 action 在合并修正参数之后才取 call_kwargs，导致偏好对的
        chosen 与 rejected 完全相同（DPO 纯噪声数据）。
        """
        llm = ReplanFakeLLM(responses=[
            {"analysis": "除数为零", "revised_params": {"denominator": 4}, "reason": "换非零除数"},
        ])

        @with_retry(max_retries=2, task_id="t-snap", replan_llm=llm, failure_logger=tmp_logger)
        def divide(numerator: int, denominator: int) -> float:
            return numerator / denominator

        assert divide(8, denominator=0) == 2.0
        record = tmp_logger.read_all()[0]
        assert "denominator': 0" in record["action"]      # Rejected：失败时的原参数
        assert "denominator': 4" in record["resolved_action"]  # Chosen：修正后成功调用
        assert record["action"] != record["resolved_action"]

    def test_llm_unavailable_still_retries(self, tmp_logger) -> None:
        """Replan LLM 不可用（None 且全局未配置）-> 退化为原样重试。"""
        attempts = {"n": 0}

        @with_retry(max_retries=2, task_id="t-nollm", replan_llm=None, failure_logger=tmp_logger)
        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("抖动")
            return "ok"

        assert flaky() == "ok"
        assert attempts["n"] == 2

    def test_llm_garbage_output_degrades_to_plain_retry(self, tmp_logger) -> None:
        """Replan 输出垃圾 -> 原样重试，不中断、不抛纠错层异常。"""
        llm = ReplanFakeLLM(responses=["抱歉，我无法输出 JSON"])

        @with_retry(max_retries=1, task_id="t-garbage", replan_llm=llm, failure_logger=tmp_logger)
        def always_fail() -> None:
            raise ValueError("业务错误")

        with pytest.raises(ValueError, match="业务错误"):
            always_fail()

    def test_llm_raising_degrades_to_plain_retry(self, tmp_logger) -> None:
        """Replan LLM 自身抛异常 -> fail-open 原样重试。"""
        llm = ReplanFakeLLM(responses=[ConnectionError("网络炸了")])

        @with_retry(max_retries=1, task_id="t-llm-crash", replan_llm=llm, failure_logger=tmp_logger)
        def always_fail() -> None:
            raise ValueError("业务错误")

        with pytest.raises(ValueError, match="业务错误"):
            always_fail()

    def test_non_dict_revised_params_ignored(self, tmp_logger) -> None:
        """revised_params 非字典（如字符串）-> 忽略修正，原样重试。"""
        llm = ReplanFakeLLM(responses=[{"analysis": "x", "revised_params": "不是字典", "reason": "y"}])
        seen_kwargs = []

        @with_retry(max_retries=1, task_id="t-badparams", replan_llm=llm, failure_logger=tmp_logger)
        def probe(**kwargs) -> dict:
            seen_kwargs.append(dict(kwargs))
            raise ValueError("总是失败")

        with pytest.raises(ValueError):
            probe(alpha=1)
        assert seen_kwargs[0] == {"alpha": 1}


# ============================ 日志器 ============================


class TestFailureLogger:

    def test_corrupt_file_backed_up_and_reset(self, tmp_path) -> None:
        """日志文件损坏 -> 备份重置，新记录正常写入，不抛异常。"""
        log_path = tmp_path / "failure_log.json"
        log_path.write_text("{{{这不是JSON", encoding="utf-8")
        logger_ = FailureLogger(str(log_path))

        logger_.record_many([{"task_id": "t", "failed_step": 1, "error_reason": "e",
                              "retry_count": 0, "final_success": False}])

        records = logger_.read_all()
        assert len(records) == 1
        assert (tmp_path / "failure_log.json.corrupt").exists()

    def test_atomic_write_no_tmp_leftover(self, tmp_logger) -> None:
        """原子写入后不残留 .tmp 文件。"""
        tmp_logger.record_many([{"task_id": "t", "failed_step": 1, "error_reason": "e",
                                 "retry_count": 0, "final_success": False}])
        assert not tmp_logger.path.with_suffix(".json.tmp").exists()
        assert tmp_logger.path.exists()

    def test_read_all_valid_json_array(self, tmp_logger) -> None:
        """落盘内容是合法 JSON 数组（用户可直接审计）。"""
        tmp_logger.record_many([
            {"task_id": "t1", "failed_step": 1, "error_reason": "e1",
             "retry_count": 0, "final_success": True},
            {"task_id": "t1", "failed_step": 2, "error_reason": "e2",
             "retry_count": 1, "final_success": True},
        ])
        on_disk = json.loads(tmp_logger.path.read_text(encoding="utf-8"))
        assert isinstance(on_disk, list) and len(on_disk) == 2

    def test_record_many_empty_is_noop(self, tmp_logger) -> None:
        tmp_logger.record_many([])
        assert tmp_logger.read_all() == []
        assert not tmp_logger.path.exists()
