"""pytest 共享夹具与测试替身。

FakeVectorStore / toy_embedding 供检索与编排层测试共用：
生产路径应注入 ChromaVectorStore，测试注入本假库实现全离线。
"""

import sys
import hashlib
import math
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.self_correction import FailureLogger  # noqa: E402
from app.memory.models import MemoryFact  # noqa: E402


@pytest.fixture
def tmp_logger(tmp_path):
    """临时 FailureLogger（重试/止损类测试共用）。"""
    return FailureLogger(str(tmp_path / "failure_log.json"))


def toy_embedding(text: str, dim: int = 256) -> List[float]:
    """确定性玩具嵌入：字符二元组哈希计数向量（L2 归一化）。

    共享二元组越多 -> 余弦相似度越高，足够近似语义检索行为，
    且完全确定、零依赖。dim 取 256 以压低哈希碰撞噪声
    （真实嵌入模型对无关文本的余弦接近 0，玩具版也应对齐此性质）。
    """
    vec = [0.0] * dim
    for gram in _bigrams(text):
        digest = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[digest % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _bigrams(text: str) -> List[str]:
    """字符二元组（与 hybrid_search._tokenize 同规则）。"""
    cleaned = "".join(text.split())
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


class FakeVectorStore:
    """内存向量库测试替身：toy 嵌入 + 余弦相似度。

    NOISE_FLOOR：噪声地板。真实嵌入模型对无关文本的余弦接近 0，
    玩具嵌入的哈希碰撞会留下少量正相似度噪声，须滤除才能
    对齐真实向量库"无关即不命中"的行为。
    """

    NOISE_FLOOR = 0.2

    def __init__(self) -> None:
        self._items = {}  # id -> (fact, embedding)

    def add(self, fact: MemoryFact, embedding: Optional[List[float]] = None) -> None:
        self._items[fact.id] = (fact, embedding or toy_embedding(fact.content))

    def delete(self, fact_id: str) -> None:
        """删除文档（与 ChromaVectorStore.delete 同接口）。"""
        self._items.pop(fact_id, None)

    def query(self, query_text: str, top_k: int) -> List[Tuple[MemoryFact, float]]:
        q = toy_embedding(query_text)
        scored = []
        for fact, emb in self._items.values():
            sim = sum(a * b for a, b in zip(q, emb))
            if sim > self.NOISE_FLOOR:  # 低于噪声地板视为无关
                scored.append((fact, sim))
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:top_k]
