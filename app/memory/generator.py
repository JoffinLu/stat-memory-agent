"""记忆生成模块（阶段二实现）。

规划：从任务执行轨迹（trajectory）中抽取结构化事实并生成 MemoryFact。
统计要点：对同一事件的多来源观察，按证据强度加权合成单一事实，
置信度初值由 LLM 自评 + 启发式规则（如工具返回码）联合决定。
"""

from typing import Any, List

from app.memory.models import MemoryFact


class MemoryGenerator:
    """执行轨迹 → 结构化记忆事实的生成器。

    TODO(阶段二):
        - extract_facts(): 从轨迹中识别值得长期保留的信息
        - assign_confidence(): 置信度初值估计
    """

    def extract_facts(self, trajectory: List[dict[str, Any]]) -> List[MemoryFact]:
        """从执行轨迹中抽取事实。

        Args:
            trajectory: 执行轨迹（步骤字典列表）。

        Returns:
            抽取出的记忆事实列表。

        Raises:
            NotImplementedError: 阶段二实现。
        """
        raise NotImplementedError("阶段二实现")

    def assign_confidence(self, fact: MemoryFact, evidence: dict[str, Any]) -> float:
        """为事实估计初始置信度。

        Raises:
            NotImplementedError: 阶段二实现。
        """
        raise NotImplementedError("阶段二实现")
