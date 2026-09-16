"""重排模块（阶段二实现）。

规划：对混合检索的粗排候选做精排。
统计要点：以「语义相似度 + 置信度 + 时近性 + 历史调用收益」为特征的
线性/学习排序模型（learning-to-rank），权重可离线用标注数据拟合。
"""

from typing import List

from app.memory.models import MemoryFact


class Reranker:
    """候选记忆重排器。

    TODO(阶段二):
        - rerank(): 对候选列表重排，返回新顺序及精排得分
    """

    def rerank(self, query: str, candidates: List[MemoryFact]) -> List[tuple[MemoryFact, float]]:
        """重排候选记忆。

        Returns:
            (记忆, 精排得分) 元组列表，按得分降序。

        Raises:
            NotImplementedError: 阶段二实现。
        """
        raise NotImplementedError("阶段二实现")
