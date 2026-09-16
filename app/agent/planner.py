"""任务规划与执行模块：transitions 状态机 + ReAct 执行器 + Critic 验证。

架构（对应 JD「任务规划 / 状态管理 / 自纠错」）：

    IDLE --plan--> PLANNING --start--> EXECUTING --verify--> VERIFYING
                       |                  ^  ^                  |  |
                       |                  |  +------redo--------+  |
                       |                  +----------proceed-----+  |
                       v                  |                        |
                     FAILED <--fail-------+--------fail------------+--finish--> SUCCESS

组件分工：
1. ExecutionStateMachine —— 基于 transitions 库的执行循环状态机，
   非法转移显式抛 MachineError（每一步可审计）；
2. TaskPlanner —— LLM 将任务分解为子任务 DAG（JSON 契约 + 容错解析 +
   依赖校验 + Kahn 拓扑排序），执行顺序由依赖图决定而非 LLM 列表顺序；
3. AgentExecutor —— ReAct 范式执行子任务（Thought/Action/Observation 循环，
   Final Answer 终止），每个子任务执行后进入 VERIFYING 由 Critic LLM 验证，
   拒绝则重试，超限则终态 FAILED。

与 state_machine.py 的关系：TaskStatus 是长程任务生命周期状态机（含
监控/修正态，接自纠错引擎）；本模块是单次 run 执行循环的细粒度状态机。
两者粒度不同、并存，阶段四集成时统一。

依赖注入模式与全项目一致：LLM 客户端可注入（FakeLLM 全离线测试），
langchain 未安装时消息降级为 dict，transitions 为唯一新增运行时依赖。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from transitions import Machine

logger = logging.getLogger(__name__)


# ============================ 异常体系 ============================


class PlannerError(RuntimeError):
    """规划失败：LLM 输出无法解析 / 子任务契约非法。"""


class DAGCycleError(PlannerError):
    """子任务依赖图存在环。"""


class SubTaskExecutionError(RuntimeError):
    """ReAct 循环内未产出最终答案（步数耗尽等）。"""


# ============================ Prompt 模板 ============================

# 规划层：强制 JSON 对象契约
PLAN_SYSTEM_PROMPT = (
    "你是任务规划器。将总体任务分解为有序子任务 DAG，只输出一个 JSON 对象："
    '{"subtasks": [{"id": "s1", "description": "子任务描述", "depends_on": []}]}。'
    "id 为子任务唯一标识，depends_on 为前置子任务 id 的数组（无前置则为空数组）。"
    "不得包含任何解释性文字或 Markdown 代码块标记。"
)

# ReAct 执行层：标准 Reason + Act 范式
REACT_SYSTEM_PROMPT = (
    "你是使用 ReAct（Reason + Act）范式的子任务执行器。严格按以下格式逐步输出：\n"
    "Thought: 当前的推理\n"
    "Action: 工具名\n"
    "Action Input: 工具入参\n"
    "系统会以 Observation: <结果> 的形式返回工具输出，然后你继续推理。\n"
    "当你能给出最终答案时，输出：\n"
    "Thought: 推理完成\n"
    "Final Answer: 最终答案\n"
    "可用工具：{tool_names}。"
)

# Critic 验证层：JSON 裁决契约
CRITIC_SYSTEM_PROMPT = (
    "你是执行结果验证器（Critic）。根据子任务描述与执行结果判断是否达标，"
    '只输出一个 JSON 对象：{"passed": true 或 false, "reason": "简要理由"}，'
    "不得包含任何其他文字。"
)


# ============================ 数据模型 ============================


class SubTask(BaseModel):
    """子任务节点（DAG 顶点）。

    Attributes:
        id: 唯一标识（如 "s1"）。
        description: 任务描述。
        depends_on: 前置子任务 id 列表。
    """

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    depends_on: List[str] = Field(default_factory=list)

    @field_validator("id", "description")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不可为空白字符串")
        return cleaned


@dataclass
class SubTaskResult:
    """单个子任务的执行记录。"""

    subtask_id: str
    description: str
    answer: Optional[str]          # ReAct 最终答案；执行失败时为 None
    passed: bool                   # Critic 裁决
    retries: int                   # Critic 拒绝后的重试次数
    critic_reason: str = ""


@dataclass
class ExecutionReport:
    """run(task) 的完整执行报告。

    Attributes:
        status: 终态，"success" 或 "failed"。
        results: 逐子任务执行记录。
        transitions: 状态机转移历史 (source, dest) 列表，供审计。
        error: 失败原因；成功时为 None。
    """

    status: str
    results: List[SubTaskResult] = field(default_factory=list)
    transitions: List[Tuple[str, str]] = field(default_factory=list)
    error: Optional[str] = None


# ============================ 容错解析工具 ============================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_response_text(response: Any) -> str:
    """兼容 LangChain 消息对象（.content）与裸字符串返回。"""
    raw = getattr(response, "content", response)
    if not isinstance(raw, str):
        raise PlannerError(f"LLM 返回了非文本内容: {type(raw).__name__}")
    return raw


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """从 LLM 原始输出中容错地提取 JSON 对象。

    与 extractor._extract_json_array 同款三层降级：直接 loads ->
    剥代码围栏 -> 截取最外层 { ... } 子串。全失败抛 PlannerError。
    """
    text = raw_text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise PlannerError(f"无法从 LLM 输出中解析 JSON 对象，前 200 字符: {raw_text[:200]!r}")


def _build_messages(system_text: str, human_text: str) -> List[Any]:
    """构造 [system, human] 消息序列。

    优先 LangChain 消息类型（生产路径）；langchain 未安装时降级为
    OpenAI 风格 dict（离线测试路径），与 extractor 同款双路径。
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        return [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    except ImportError:
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": human_text},
        ]


# ============================ 状态机 ============================

# 状态常量（用户契约的六个状态）
IDLE = "IDLE"
PLANNING = "PLANNING"
EXECUTING = "EXECUTING"
VERIFYING = "VERIFYING"
FAILED = "FAILED"
SUCCESS = "SUCCESS"

EXECUTION_STATES = [IDLE, PLANNING, EXECUTING, VERIFYING, FAILED, SUCCESS]


class ExecutionStateMachine:
    """执行循环状态机（transitions 库实现）。

    转移触发器：
        plan    : IDLE -> PLANNING                 开始规划
        start   : PLANNING -> EXECUTING            开始执行子任务
        verify  : EXECUTING -> VERIFYING           子任务执行完，交 Critic 验证
        redo    : VERIFYING -> EXECUTING           Critic 拒绝，重试当前子任务
        proceed : VERIFYING -> EXECUTING           Critic 通过，进入下一子任务
        finish  : VERIFYING -> SUCCESS             全部子任务通过
        fail    : PLANNING/EXECUTING/VERIFYING -> FAILED

    设计原则：非法转移直接抛 MachineError（ignore_invalid_triggers=False），
    与 state_machine.StateMachine 的「显式拒绝」原则一致 ——
    状态机的每一步都要可审计，这是长程任务 debug 的生命线。
    """

    def __init__(self) -> None:
        self._history: List[Tuple[str, str]] = []
        self.machine = Machine(
            model=self,
            states=EXECUTION_STATES,
            initial=IDLE,
            ignore_invalid_triggers=False,
            send_event=True,
            after_state_change=self._record,
        )
        self.machine.add_transition("plan", IDLE, PLANNING)
        self.machine.add_transition("start", PLANNING, EXECUTING)
        self.machine.add_transition("verify", EXECUTING, VERIFYING)
        self.machine.add_transition("redo", VERIFYING, EXECUTING)
        self.machine.add_transition("proceed", VERIFYING, EXECUTING)
        self.machine.add_transition("finish", VERIFYING, SUCCESS)
        self.machine.add_transition("fail", [PLANNING, EXECUTING, VERIFYING], FAILED)

    def _record(self, event: Any) -> None:
        """after_state_change 回调：记录 (source, dest) 转移历史。"""
        self._history.append((event.transition.source, event.transition.dest))

    # 当前状态直接读 self.state —— transitions.Machine 在 model 上动态
    # 绑定该属性，不要另定义同名 property（会与 Machine 的 setattr 冲突）。

    @property
    def history(self) -> List[Tuple[str, str]]:
        """转移历史（副本）。"""
        return list(self._history)


# ============================ 规划器 ============================


class TaskPlanner:
    """LLM 驱动的子任务 DAG 规划器。"""

    def plan(
        self,
        task: str,
        llm_client: Optional[Any] = None,
        memory_context: str = "",
    ) -> List[SubTask]:
        """将总体任务分解为拓扑有序的子任务列表。

        Args:
            task: 总体任务描述。
            llm_client: 可注入 LLM 客户端；None 时使用全局单例。
            memory_context: 相关记忆上下文文本（AgentRuntime 注入），
                空串表示无记忆可参考。

        Returns:
            拓扑有序（Kahn）的 SubTask 列表，执行顺序即列表顺序。

        Raises:
            ValueError: task 非法。
            PlannerError: LLM 不可用 / 输出无法解析 / 子任务契约非法。
            DAGCycleError: 依赖图存在环。
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task 必须是非空字符串")

        client = llm_client if llm_client is not None else self._default_client()
        if client is None:
            raise PlannerError("LLM 客户端未配置，无法进行任务规划")

        human_text = f"总体任务：{task.strip()}"
        if memory_context.strip():
            human_text += f"\n\n相关记忆上下文（规划时参考）：\n{memory_context.strip()}"

        response = client.invoke(_build_messages(PLAN_SYSTEM_PROMPT, human_text))
        payload = _extract_json_object(_strip_response_text(response))

        raw_items = payload.get("subtasks")
        if not isinstance(raw_items, list) or not raw_items:
            raise PlannerError("规划结果缺少非空 subtasks 数组")

        try:
            subtasks = [SubTask(**item) for item in raw_items if isinstance(item, dict)]
        except Exception as exc:  # pydantic ValidationError 统一转 PlannerError
            raise PlannerError(f"子任务字段非法: {exc}") from exc

        if len(subtasks) != len(raw_items):
            raise PlannerError("subtasks 中存在非对象条目")

        self._validate_ids(subtasks)
        return self._topological_sort(subtasks)

    @staticmethod
    def _default_client() -> Optional[Any]:
        """延迟导入全局客户端，保持本模块离线可导入。"""
        from app.core.llm_client import get_llm_client  # noqa: PLC0415

        return get_llm_client()

    @staticmethod
    def _validate_ids(subtasks: List[SubTask]) -> None:
        """校验 id 唯一性（重复 id 会让 DAG 顶点歧义）。"""
        ids = [st.id for st in subtasks]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise PlannerError(f"子任务 id 重复: {dupes}")

    @staticmethod
    def _topological_sort(subtasks: List[SubTask]) -> List[SubTask]:
        """Kahn 拓扑排序；同层按 LLM 原始顺序稳定输出。

        环依赖与悬空依赖（引用不存在的 id）都在此显式报错 ——
        LLM 生成的依赖图必须经过结构校验才可执行。
        """
        by_id = {st.id: st for st in subtasks}
        order_index = {st.id: i for i, st in enumerate(subtasks)}
        indegree = {st.id: 0 for st in subtasks}
        dependents: Dict[str, List[str]] = {st.id: [] for st in subtasks}

        for st in subtasks:
            for dep in st.depends_on:
                if dep not in by_id:
                    raise PlannerError(f"子任务 {st.id} 依赖了不存在的 id: {dep}")
                if dep == st.id:
                    raise DAGCycleError(f"子任务 {st.id} 依赖自身")
                indegree[st.id] += 1
                dependents[dep].append(st.id)

        # 就绪队列按原始顺序排序，保证输出确定
        ready = sorted((i for i, d in indegree.items() if d == 0), key=order_index.get)
        topo_ids: List[str] = []

        while ready:
            node = ready.pop(0)
            topo_ids.append(node)
            for child in dependents[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort(key=order_index.get)

        if len(topo_ids) != len(subtasks):
            unresolved = sorted(set(by_id) - set(topo_ids))
            raise DAGCycleError(f"子任务依赖存在环，无法排序的节点: {unresolved}")

        return [by_id[i] for i in topo_ids]


# ============================ 执行器 ============================

_ACTION_RE = re.compile(r"Action:\s*(.+?)\s*\nAction Input:\s*(.*)", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


class AgentExecutor:
    """ReAct 执行器 + Critic 验证循环。

    流程（run(task)）：
        IDLE --plan--> PLANNING：LLM 分解任务为子任务 DAG；
        PLANNING --start--> EXECUTING：按拓扑序逐个执行子任务（ReAct 循环）；
        每个子任务执行完 --verify--> VERIFYING：Critic LLM 裁决；
            拒绝 --redo--> EXECUTING 重试，超过 max_retries --fail--> FAILED；
            通过 --proceed--> 下一子任务 / --finish--> SUCCESS。

    Attributes:
        tools: 工具注册表 {工具名: 输入字符串 -> 输出字符串}，
            ReAct 的 Action 派发到此执行；为空时 LLM 被告知只能直接推理作答。
    """

    def __init__(
        self,
        executor_llm: Optional[Any] = None,
        critic_llm: Optional[Any] = None,
        tools: Optional[Dict[str, Callable[[str], str]]] = None,
        max_retries: int = 2,
        max_react_steps: int = 5,
    ) -> None:
        """初始化执行器。

        Args:
            executor_llm: 执行 LLM（规划 + ReAct 执行共用）；None 用全局单例。
            critic_llm: Critic LLM；None 时复用 executor_llm。
            tools: 可注入工具注册表。
            max_retries: 单个子任务允许的最大重试次数。
            max_react_steps: 单次 ReAct 循环的最大步数（防失控）。
        """
        self._executor_llm = executor_llm
        self._critic_llm = critic_llm
        self.tools = dict(tools) if tools else {}
        self.max_retries = max_retries
        self.max_react_steps = max_react_steps
        self._planner = TaskPlanner()
        self._sm = ExecutionStateMachine()

    # ---------- 对外主入口 ----------

    def run(self, task: str, memory_context: str = "") -> ExecutionReport:
        """执行总体任务：规划 -> 逐子任务 ReAct 执行 -> Critic 验证。

        Args:
            task: 总体任务描述（非空字符串）。
            memory_context: 相关记忆上下文（AgentRuntime 注入），
                会进入规划、执行、验证三层的 Prompt。

        Returns:
            ExecutionReport，status 为 "success" 或 "failed"。
            本方法不抛规划/执行异常 —— 一切失败都体现在报告的终态里。

        Raises:
            ValueError: task 非法。
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task 必须是非空字符串")

        # 状态机按次重建：executor 可被 runtime 反复调用（长程循环），
        # 上一轮的终态（SUCCESS/FAILED）不能污染本轮的转移历史
        self._sm = ExecutionStateMachine()
        self._sm.plan()
        try:
            subtasks = self._planner.plan(task, self._executor_llm, memory_context)
        except PlannerError as exc:
            logger.error("任务规划失败: %s", exc)
            self._sm.fail()
            return ExecutionReport(
                status="failed", transitions=self._sm.history, error=f"规划失败: {exc}"
            )

        self._sm.start()
        results: List[SubTaskResult] = []
        total = len(subtasks)

        for idx, subtask in enumerate(subtasks):
            result = self._run_single_subtask(task, subtask, memory_context)
            if not result.passed:
                self._sm.fail()
                return ExecutionReport(
                    status="failed",
                    results=results + [result],
                    transitions=self._sm.history,
                    error=f"子任务 {subtask.id} 验证未通过: {result.critic_reason}",
                )
            results.append(result)

            if idx < total - 1:
                self._sm.proceed()

        self._sm.finish()
        return ExecutionReport(
            status="success", results=results, transitions=self._sm.history
        )

    # ---------- 子任务执行与验证 ----------

    def _run_single_subtask(
        self, task: str, subtask: SubTask, memory_context: str = ""
    ) -> SubTaskResult:
        """执行单个子任务并经 Critic 验证，含重试循环。

        VERIFYING 拒绝 -> redo 回 EXECUTING 重试；重试超限返回未通过结果，
        由 run() 统一 fail()。执行异常（步数耗尽）作为失败结果交给 Critic
        裁决 —— 保留 Critic 对「半成品」的判断权。
        """
        retries = 0
        while True:
            answer: Optional[str] = None
            exec_error: Optional[str] = None
            try:
                answer = self._react_execute(task, subtask, memory_context)
            except SubTaskExecutionError as exc:
                exec_error = str(exc)
                logger.warning("子任务 %s 执行异常: %s", subtask.id, exc)

            self._sm.verify()
            passed, reason = self._critique(task, subtask, answer, exec_error, memory_context)

            if passed:
                return SubTaskResult(
                    subtask_id=subtask.id,
                    description=subtask.description,
                    answer=answer,
                    passed=True,
                    retries=retries,
                    critic_reason=reason,
                )

            if retries >= self.max_retries:
                return SubTaskResult(
                    subtask_id=subtask.id,
                    description=subtask.description,
                    answer=answer,
                    passed=False,
                    retries=retries,
                    critic_reason=reason,
                )

            retries += 1
            self._sm.redo()
            logger.info("子任务 %s 第 %d 次重试（Critic: %s）", subtask.id, retries, reason)

    def _react_execute(self, task: str, subtask: SubTask, memory_context: str = "") -> str:
        """以 ReAct 循环执行子任务，返回 Final Answer。

        Args:
            task: 总体任务（作为上下文）。
            subtask: 当前子任务。
            memory_context: 相关记忆上下文（注入执行 Prompt）。

        Raises:
            SubTaskExecutionError: LLM 不可用 / 步数耗尽仍未给出 Final Answer。
        """
        client = self._executor_llm if self._executor_llm is not None else TaskPlanner._default_client()
        if client is None:
            raise SubTaskExecutionError("LLM 客户端未配置，无法执行子任务")

        tool_names = ", ".join(self.tools) if self.tools else "无（直接推理作答）"
        system_text = REACT_SYSTEM_PROMPT.replace("{tool_names}", tool_names)
        human_text = f"总体任务：{task}\n当前子任务（{subtask.id}）：{subtask.description}"
        if memory_context.strip():
            human_text += f"\n\n相关记忆上下文（执行时参考）：\n{memory_context.strip()}"
        messages = _build_messages(system_text, human_text)

        for _step in range(self.max_react_steps):
            response = client.invoke(messages)
            text = _strip_response_text(response)

            final_match = _FINAL_RE.search(text)
            if final_match:
                return final_match.group(1).strip()

            action_match = _ACTION_RE.search(text)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                tool_fn = self.tools.get(tool_name)
                if tool_fn is None:
                    observation = f"错误：工具 {tool_name} 不存在"
                else:
                    observation = tool_fn(tool_input)
            else:
                # 既无 Final Answer 也无 Action：视为格式违约，提示后继续
                observation = "格式错误：输出中既没有 Final Answer 也没有 Action/Action Input"

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        raise SubTaskExecutionError(
            f"ReAct 步数耗尽（{self.max_react_steps} 步）仍未产出 Final Answer"
        )

    def _critique(
        self,
        task: str,
        subtask: SubTask,
        answer: Optional[str],
        exec_error: Optional[str],
        memory_context: str = "",
    ) -> Tuple[bool, str]:
        """调用 Critic LLM 裁决子任务结果。

        Critic 不可用或输出非法时按「未通过」处理（fail-closed）——
        验证器的失效不能被默认为通过，否则整个 VERIFYING 环节形同虚设。
        """
        shown_result = answer if answer is not None else f"（执行失败：{exec_error}）"
        human_text = (
            f"总体任务：{task}\n子任务（{subtask.id}）：{subtask.description}\n"
            f"执行结果：{shown_result}"
        )
        if memory_context.strip():
            human_text += f"\n相关记忆上下文（验证时参考）：\n{memory_context.strip()}"

        client = self._critic_llm if self._critic_llm is not None else (
            self._executor_llm if self._executor_llm is not None else TaskPlanner._default_client()
        )
        if client is None:
            return False, "Critic LLM 不可用，按未通过处理"

        try:
            response = client.invoke(_build_messages(CRITIC_SYSTEM_PROMPT, human_text))
            payload = _extract_json_object(_strip_response_text(response))
        except PlannerError as exc:
            return False, f"Critic 输出无法解析: {exc}"

        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        # passed 必须严格为布尔 True，其他值（"true"/1 等）一律视为未通过
        return payload.get("passed") is True, reason


class LLMUnavailableError(PlannerError):
    """兼容命名：LLM 客户端不可用（离线模式）。"""
