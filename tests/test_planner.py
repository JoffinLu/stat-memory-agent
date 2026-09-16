"""agent/planner.py 单元测试：状态机、DAG 规划、ReAct 执行与 Critic 验证。

全部使用 FakeLLM 测试替身，全程离线，零 LLM API 依赖。
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from transitions.core import MachineError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.planner import (  # noqa: E402
    DAGCycleError,
    EXECUTION_STATES,
    ExecutionStateMachine,
    PlannerError,
    SubTask,
    SubTaskResult,
    TaskPlanner,
)
from app.agent.planner import AgentExecutor  # noqa: E402


# ============================ 测试替身 ============================


class RoleDispatchFakeLLM:
    """按 Prompt 特征分发的 FakeLLM：规划 / ReAct 执行 / Critic 三种角色。

    - 规划请求（含 "任务规划器"）→ 返回预设子任务 JSON；
    - Critic 请求（含 "Critic"）→ 按裁决脚本逐次返回 passed/reason；
    - 其余视为 ReAct 执行请求 → 按执行脚本逐次返回文本。
    """

    def __init__(
        self,
        subtasks: Any = None,
        exec_outputs: List[str] = None,
        critic_verdicts: List[Any] = None,
    ) -> None:
        self.subtasks = subtasks
        self.exec_outputs = exec_outputs or ['Thought: 直接作答\nFinal Answer: 已完成']
        self.critic_verdicts = critic_verdicts or [{"passed": True, "reason": "达标"}]
        self.exec_calls = 0
        self.critic_calls = 0

    def invoke(self, messages: List[Any]) -> Any:
        combined = " ".join(
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in messages
        )
        if "任务规划器" in combined:
            if isinstance(self.subtasks, Exception):
                raise self.subtasks
            if isinstance(self.subtasks, str):
                return SimpleNamespace(content=self.subtasks)
            return SimpleNamespace(content=json.dumps({"subtasks": self.subtasks}, ensure_ascii=False))
        if "Critic" in combined:
            verdict = self.critic_verdicts[min(self.critic_calls, len(self.critic_verdicts) - 1)]
            self.critic_calls += 1
            if isinstance(verdict, Exception):
                raise verdict
            if isinstance(verdict, str):
                return SimpleNamespace(content=verdict)
            return SimpleNamespace(content=json.dumps(verdict, ensure_ascii=False))
        # ReAct 执行请求
        output = self.exec_outputs[min(self.exec_calls, len(self.exec_outputs) - 1)]
        self.exec_calls += 1
        return SimpleNamespace(content=output)


def make_executor(**kwargs) -> AgentExecutor:
    """构造带默认 FakeLLM 的执行器。"""
    subtasks = kwargs.pop("subtasks", [
        {"id": "s1", "description": "子任务一", "depends_on": []},
        {"id": "s2", "description": "子任务二", "depends_on": ["s1"]},
    ])
    llm = RoleDispatchFakeLLM(
        subtasks=subtasks,
        exec_outputs=kwargs.pop("exec_outputs", None),
        critic_verdicts=kwargs.pop("critic_verdicts", None),
    )
    return AgentExecutor(executor_llm=llm, critic_llm=llm, **kwargs), llm


# ============================ 状态机 ============================


class TestExecutionStateMachine:

    def test_happy_path_transitions(self) -> None:
        """完整成功路径：IDLE→PLANNING→EXECUTING→VERIFYING→EXECUTING→...→SUCCESS。"""
        sm = ExecutionStateMachine()
        assert sm.state == "IDLE"
        sm.plan()
        assert sm.state == "PLANNING"
        sm.start()
        sm.verify()
        sm.proceed()
        sm.verify()
        sm.finish()
        assert sm.state == "SUCCESS"
        assert sm.history == [
            ("IDLE", "PLANNING"),
            ("PLANNING", "EXECUTING"),
            ("EXECUTING", "VERIFYING"),
            ("VERIFYING", "EXECUTING"),
            ("EXECUTING", "VERIFYING"),
            ("VERIFYING", "SUCCESS"),
        ]

    def test_retry_loop_transitions(self) -> None:
        """Critic 拒绝走 redo 回 EXECUTING，最终仍可 finish。"""
        sm = ExecutionStateMachine()
        sm.plan()
        sm.start()
        sm.verify()
        sm.redo()
        assert sm.state == "EXECUTING"
        sm.verify()
        sm.finish()
        assert sm.state == "SUCCESS"

    def test_fail_from_any_active_state(self) -> None:
        """fail 触发器从 PLANNING / EXECUTING / VERIFYING 均可达 FAILED。"""
        for trigger_chain in (
            ["plan", "fail"],
            ["plan", "start", "fail"],
            ["plan", "start", "verify", "fail"],
        ):
            sm = ExecutionStateMachine()
            for trigger in trigger_chain:
                getattr(sm, trigger)()
            assert sm.state == "FAILED"

    def test_failed_is_terminal(self) -> None:
        """FAILED 是终态：任何触发器都应被拒绝。"""
        sm = ExecutionStateMachine()
        sm.plan()
        sm.fail()
        with pytest.raises(MachineError):
            sm.start()
        with pytest.raises(MachineError):
            sm.fail()

    def test_illegal_transition_rejected(self) -> None:
        """非法转移（如 IDLE 直接 EXECUTING）显式抛 MachineError。"""
        sm = ExecutionStateMachine()
        with pytest.raises(MachineError):
            sm.start()
        sm.plan()
        with pytest.raises(MachineError):
            sm.verify()

    def test_states_exactly_match_contract(self) -> None:
        """状态集合与用户契约完全一致。"""
        assert set(EXECUTION_STATES) == {
            "IDLE", "PLANNING", "EXECUTING", "VERIFYING", "FAILED", "SUCCESS"
        }


# ============================ DAG 规划 ============================


class TestTaskPlanner:

    def setup_method(self) -> None:
        self.planner = TaskPlanner()

    def test_respects_dependency_order(self) -> None:
        """拓扑排序：被依赖者先执行，即使 LLM 列表顺序颠倒。"""
        llm = RoleDispatchFakeLLM(subtasks=[
            {"id": "s2", "description": "第二步", "depends_on": ["s1"]},
            {"id": "s1", "description": "第一步", "depends_on": []},
            {"id": "s3", "description": "第三步", "depends_on": ["s2"]},
        ])
        result = self.planner.plan("写一份报告", llm)
        assert [st.id for st in result] == ["s1", "s2", "s3"]

    def test_diamond_dag(self) -> None:
        """菱形依赖 s1 -> (s2, s3) -> s4：s2/s3 同层保持稳定顺序。"""
        llm = RoleDispatchFakeLLM(subtasks=[
            {"id": "s4", "description": "汇总", "depends_on": ["s2", "s3"]},
            {"id": "s3", "description": "分支B", "depends_on": ["s1"]},
            {"id": "s1", "description": "起点", "depends_on": []},
            {"id": "s2", "description": "分支A", "depends_on": ["s1"]},
        ])
        result = self.planner.plan("菱形任务", llm)
        ids = [st.id for st in result]
        assert ids[0] == "s1" and ids[-1] == "s4"
        assert set(ids[1:3]) == {"s2", "s3"}

    def test_cycle_detected(self) -> None:
        """环依赖显式报错。"""
        llm = RoleDispatchFakeLLM(subtasks=[
            {"id": "s1", "description": "A", "depends_on": ["s2"]},
            {"id": "s2", "description": "B", "depends_on": ["s1"]},
        ])
        with pytest.raises(DAGCycleError):
            self.planner.plan("环形任务", llm)

    def test_unknown_dependency_rejected(self) -> None:
        """引用不存在的 id 显式报错。"""
        llm = RoleDispatchFakeLLM(subtasks=[
            {"id": "s1", "description": "A", "depends_on": ["ghost"]},
        ])
        with pytest.raises(PlannerError, match="ghost"):
            self.planner.plan("悬空依赖", llm)

    def test_duplicate_ids_rejected(self) -> None:
        llm = RoleDispatchFakeLLM(subtasks=[
            {"id": "s1", "description": "A", "depends_on": []},
            {"id": "s1", "description": "B", "depends_on": []},
        ])
        with pytest.raises(PlannerError, match="重复"):
            self.planner.plan("重复 id", llm)

    def test_empty_subtasks_rejected(self) -> None:
        llm = RoleDispatchFakeLLM(subtasks=[])
        with pytest.raises(PlannerError):
            self.planner.plan("空规划", llm)

    def test_tolerant_parsing_of_fenced_output(self) -> None:
        """LLM 输出带 Markdown 围栏与前后缀文字也能解析。"""
        payload = json.dumps({"subtasks": [{"id": "s1", "description": "A", "depends_on": []}]},
                             ensure_ascii=False)
        llm = RoleDispatchFakeLLM(subtasks=f"规划结果如下：\n```json\n{payload}\n```\n以上。")
        result = self.planner.plan("围栏任务", llm)
        assert len(result) == 1

    def test_blank_task_rejected(self) -> None:
        with pytest.raises(ValueError):
            self.planner.plan("   ", RoleDispatchFakeLLM())


# ============================ ReAct 执行 + Critic 验证 ============================


class TestAgentExecutor:

    def test_happy_path(self) -> None:
        """两个子任务顺序通过 -> SUCCESS，转移历史完整可审计。"""
        executor, llm = make_executor()
        report = executor.run("完成一项多步任务")

        assert report.status == "success"
        assert report.error is None
        assert [r.subtask_id for r in report.results] == ["s1", "s2"]
        assert all(r.passed for r in report.results)
        assert all(r.answer == "已完成" for r in report.results)
        assert report.transitions[-1] == ("VERIFYING", "SUCCESS")
        # 每个子任务都经历 verify
        assert report.transitions.count(("EXECUTING", "VERIFYING")) == 2

    def test_critic_reject_then_pass(self) -> None:
        """第一次裁决拒绝 -> redo 重试 -> 第二次通过，重试数记入报告。"""
        executor, llm = make_executor(
            critic_verdicts=[
                {"passed": False, "reason": "结果不完整"},
                {"passed": True, "reason": "修正后达标"},
            ]
        )
        report = executor.run("需要一次重试的任务")

        assert report.status == "success"
        assert report.results[0].retries == 1
        assert ("VERIFYING", "EXECUTING") in report.transitions  # redo 路径

    def test_critic_always_rejects_fails_task(self) -> None:
        """重试超限 -> 终态 FAILED，报告带失败原因。"""
        executor, _ = make_executor(
            critic_verdicts=[{"passed": False, "reason": "始终不达标"}],
            max_retries=2,
        )
        report = executor.run("注定失败的任务")

        assert report.status == "failed"
        assert "始终不达标" in report.error
        assert report.results[0].retries == 2
        assert report.transitions[-1] == ("VERIFYING", "FAILED")

    def test_react_tool_dispatch(self) -> None:
        """ReAct Action 派发到工具注册表，Observation 回填后给出 Final Answer。"""
        calls = []

        def fake_search(query: str) -> str:
            calls.append(query)
            return f"关于{query}的检索结果"

        executor, _ = make_executor(
            exec_outputs=[
                "Thought: 需要查资料\nAction: search\nAction Input: 统计方法",
                "Thought: 拿到结果\nFinal Answer: 检索完成",
            ]
        )
        executor.tools = {"search": fake_search}
        report = executor.run("查资料并总结")

        assert report.status == "success"
        assert calls == ["统计方法"]
        assert report.results[0].answer == "检索完成"

    def test_unknown_tool_yields_error_observation(self) -> None:
        """不存在的工具 -> Observation 报错 -> LLM 下一轮给出 Final Answer。"""
        executor, _ = make_executor(
            exec_outputs=[
                "Thought: 调用不存在的工具\nAction: nope\nAction Input: x",
                "Thought: 换直接作答\nFinal Answer: 直接完成",
            ]
        )
        report = executor.run("工具不存在场景")
        assert report.status == "success"

    def test_react_steps_exhausted_then_critic_fails(self) -> None:
        """ReAct 步数耗尽 -> 执行异常进入 VERIFYING -> Critic 判未通过 -> FAILED。"""
        executor, _ = make_executor(
            exec_outputs=["Thought: 我一直不给出最终答案"],
            critic_verdicts=[{"passed": False, "reason": "无有效结果"}],
            max_react_steps=3,
            max_retries=0,
        )
        report = executor.run("失控的执行")

        assert report.status == "failed"
        assert report.results[0].answer is None
        assert "无有效结果" in report.error

    def test_planning_failure_fails_fast(self) -> None:
        """规划失败 -> 不进入执行，直接 FAILED，无子任务结果。"""
        executor, _ = make_executor(subtasks="这不是JSON")
        report = executor.run("规划就失败的任务")

        assert report.status == "failed"
        assert "规划失败" in report.error
        assert report.results == []
        assert report.transitions == [("IDLE", "PLANNING"), ("PLANNING", "FAILED")]

    def test_critic_garbage_output_fails_closed(self) -> None:
        """Critic 输出不可解析 -> fail-closed（视为未通过），不会默认放行。"""
        executor, _ = make_executor(
            critic_verdicts=["抱歉我无法输出 JSON"],
            max_retries=0,
        )
        report = executor.run("Critic 失效场景")

        assert report.status == "failed"
        assert "无法解析" in report.error

    def test_critic_non_bool_passed_fails_closed(self) -> None:
        """passed 非严格布尔 true（如字符串 "true"）-> 视为未通过。"""
        executor, _ = make_executor(
            critic_verdicts=[{"passed": "true", "reason": "类型噪声"}],
            max_retries=0,
        )
        report = executor.run("类型噪声场景")
        assert report.status == "failed"

    def test_blank_task_rejected(self) -> None:
        executor, _ = make_executor()
        with pytest.raises(ValueError):
            executor.run("")

    def test_report_records_retry_count_per_subtask(self) -> None:
        """重试计数按子任务独立。"""
        executor, _ = make_executor(
            critic_verdicts=[
                {"passed": False, "reason": "s1 不过"},
                {"passed": True, "reason": "s1 通过"},
                {"passed": True, "reason": "s2 一次通过"},
            ]
        )
        report = executor.run("重试计数任务")
        assert report.results[0].retries == 1
        assert report.results[1].retries == 0
