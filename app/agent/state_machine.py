"""长程任务状态机（阶段四实现）。

规划：显式状态机管理长程任务生命周期，避免"全量历史塞上下文"。
统计要点：轨迹摘要用 EWMA（指数加权移动平均）压缩历史状态，
近期步骤权重高、远期步骤按指数衰减，兼顾细节与长期趋势。
"""

from enum import Enum
from typing import Any, List


class TaskStatus(str, Enum):
    """长程任务的状态枚举。"""

    PENDING = "pending"          # 已创建未开始
    PLANNING = "planning"        # 规划中
    EXECUTING = "executing"      # 执行中
    MONITORING = "monitoring"    # 自纠错引擎监控中（触发偏差检测）
    CORRECTING = "correcting"    # 修正中（反思 + 策略调整）
    COMPLETED = "completed"      # 成功完成
    FAILED = "failed"            # 终态失败


# 合法的状态转移表：{当前状态: 允许的下一状态集合}
VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PLANNING, TaskStatus.FAILED},
    TaskStatus.PLANNING: {TaskStatus.EXECUTING, TaskStatus.FAILED},
    TaskStatus.EXECUTING: {TaskStatus.MONITORING, TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.MONITORING: {TaskStatus.EXECUTING, TaskStatus.CORRECTING, TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.CORRECTING: {TaskStatus.EXECUTING, TaskStatus.FAILED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
}


class StateMachine:
    """任务状态机（阶段四实现完整逻辑）。

    状态转移表已实现并可独立单测，转移逻辑与 LLM 无关。
    """

    def __init__(self, initial: TaskStatus = TaskStatus.PENDING) -> None:
        """初始化状态机。

        Args:
            initial: 初始状态，默认 PENDING。
        """
        self._status = initial
        self._history: List[tuple[TaskStatus, TaskStatus]] = []

    @property
    def status(self) -> TaskStatus:
        """当前状态（只读）。"""
        return self._status

    @property
    def history(self) -> List[tuple[TaskStatus, TaskStatus]]:
        """状态转移历史：(from_status, to_status) 列表。"""
        return list(self._history)

    def can_transition(self, target: TaskStatus) -> bool:
        """判断是否允许向目标状态转移。

        Args:
            target: 目标状态。

        Returns:
            允许转移返回 True，否则 False。
        """
        return target in VALID_TRANSITIONS[self._status]

    def transition(self, target: TaskStatus) -> None:
        """执行状态转移。

        Args:
            target: 目标状态。

        Raises:
            ValueError: 目标状态不在当前状态的合法转移集合内。

        Note:
            显式拒绝非法转移而非静默通过 —— 状态机的每一步都要可审计，
            这是长程任务 debug 的生命线。
        """
        if not self.can_transition(target):
            raise ValueError(
                f"非法状态转移: {self._status.value} -> {target.value}，"
                f"允许的目标状态: {sorted(s.value for s in VALID_TRANSITIONS[self._status])}"
            )
        self._history.append((self._status, target))
        self._status = target
