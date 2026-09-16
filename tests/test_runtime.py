"""AgentRuntime（记忆 ⇄ 执行闭环）单元测试，全离线。"""

import json
from types import SimpleNamespace

import pytest

from app.agent.planner import AgentExecutor
from app.agent.runtime import AgentRuntime
from app.memory.manager import MemoryManager
from app.memory.models import MemoryFact
from app.retrieval.hybrid_search import InMemoryVectorStore


class ScriptedLLM:
    """按 system 内容分发的脚本化 LLM：规划 / ReAct / Critic / 融合。"""

    def __init__(self, subtasks=None, final_answer="结果X", critic_pass=True) -> None:
        self.subtasks = subtasks or [{"id": "s1", "description": "查询用户偏好", "depends_on": []}]
        self.final_answer = final_answer
        # critic_pass 支持 bool（每次相同）或列表（按次消费，耗尽后重复末值）
        self._critic_plan = critic_pass if isinstance(critic_pass, list) else None
        self._static_critic = None if isinstance(critic_pass, list) else critic_pass
        self.planning_prompts: list = []
        self.react_prompts: list = []
        self.critic_prompts: list = []

    def _next_critic(self) -> bool:
        if self._critic_plan is not None:
            if len(self._critic_plan) > 1:
                return self._critic_plan.pop(0)
            return self._critic_plan[0]
        return self._static_critic

    def invoke(self, messages):
        system = ""
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            role = m.get("role", "") if isinstance(m, dict) else getattr(m, "type", "")
            if role == "system":
                system = content
        if "任务规划器" in system:
            self.planning_prompts.append(self._join(messages))
            return SimpleNamespace(content=json.dumps({"subtasks": self.subtasks}, ensure_ascii=False))
        if "验证器" in system:
            self.critic_prompts.append(self._join(messages))
            return SimpleNamespace(content=json.dumps(
                {"passed": self._next_critic(), "reason": "验证理由"}, ensure_ascii=False
            ))
        self.react_prompts.append(self._join(messages))
        return SimpleNamespace(content=f"Thought: 完成\nFinal Answer: {self.final_answer}")

    @staticmethod
    def _join(messages) -> str:
        return " ".join(
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            for m in messages
        )


def make_runtime(critic_pass=True, llm=None, spc_window=5):
    # 记忆侧 llm_client=None：融合路径由 runtime 的 fail-open 降级兜底
    memory = MemoryManager(llm_client=None, vector_store=InMemoryVectorStore())
    executor = AgentExecutor(executor_llm=llm, critic_llm=llm)
    runtime = AgentRuntime(memory, executor, spc_window_size=spc_window)
    return runtime, memory, llm


class TestMemoryInjection:
    def test_recalled_memory_injected_into_prompts(self) -> None:
        llm = ScriptedLLM()
        runtime, memory, _ = make_runtime(llm=llm)
        memory.add_fact(MemoryFact(content="用户偏好使用 pytest 框架编写单元测试", confidence=0.9))

        report = runtime.run("为用户编写单元测试")
        assert report.execution.status == "success"
        # 召回上下文里出现了记忆内容
        assert any("pytest" in c for c in report.context_used)
        # 三层 Prompt 都注入了记忆
        assert all(any("pytest" in p for p in prompts)
                   for prompts in (llm.planning_prompts, llm.react_prompts, llm.critic_prompts))

    def test_empty_memory_still_runs(self) -> None:
        llm = ScriptedLLM()
        runtime, _, _ = make_runtime(llm=llm)
        report = runtime.run("独立任务")
        assert report.execution.status == "success"
        assert report.context_used == []


class TestExperienceWriteback:
    def test_experience_recorded_after_run(self) -> None:
        llm = ScriptedLLM()
        runtime, memory, _ = make_runtime(llm=llm)
        before = memory.size
        report = runtime.run("执行某任务")
        assert report.experiences_recorded == 1
        assert memory.size == before + 1
        contents = [r.fact.content for r in memory.recall("执行某任务", top_k=5)]
        assert any("执行经验" in c and "成功" in c for c in contents)

    def test_failure_experience_has_lower_confidence(self) -> None:
        llm = ScriptedLLM(critic_pass=False)
        runtime, memory, _ = make_runtime(llm=llm)
        report = runtime.run("会失败的任务")
        assert report.execution.status == "failed"
        facts = memory._searcher.facts
        assert facts and facts[0].confidence == pytest.approx(0.6)


class TestSpcSelfCheck:
    def test_success_streak_then_failure_triggers_self_check(self) -> None:
        llm = ScriptedLLM(critic_pass=[True, True, True, False])
        runtime, memory, _ = make_runtime(llm=llm, spc_window=3)
        for _ in range(3):
            runtime.run("正常任务")  # 窗口填满 3 个全 1（零方差）
        # 第 4 次失败：0 落在 [1,1] 控制限之外 -> 失控 -> 自检触发
        report = runtime.run("突然失败的任务")
        assert report.execution.status == "failed"
        assert report.self_check_triggered is True

    def test_steady_success_no_self_check(self) -> None:
        llm = ScriptedLLM()
        runtime, _, _ = make_runtime(llm=llm, spc_window=3)
        for _ in range(5):
            report = runtime.run("正常任务")
            assert report.self_check_triggered is False

    def test_window_not_full_no_check(self) -> None:
        llm = ScriptedLLM(critic_pass=False)
        runtime, _, _ = make_runtime(llm=llm, spc_window=5)
        report = runtime.run("失败任务")
        assert report.self_check_triggered is False  # 窗口未满不监控
