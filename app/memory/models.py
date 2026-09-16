"""核心数据模型定义。

设计说明（相对原始需求的三处显式优化，均已确认）：
1. confidence 限定在 [0, 1] 区间 —— 它是贝叶斯信念更新的概率语义，
   越界值会在冲突消解层引发静默错误，因此在模型层拦截。
2. embedding 为可选字段 —— 向量属于检索层关注点，API 传输时不应强制携带。
3. timestamp 统一为 UTC 时区感知时间 —— 时区混乱是时间衰减模型的头号坑。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryFact(BaseModel):
    """单条结构化记忆事实。

    这是整个记忆系统的最小数据单元：
    - 生成层产出 MemoryFact；
    - 检索层按 relevance score 排序 MemoryFact；
    - 冲突消解层对同一主题的多条 MemoryFact 做 Bayesian 更新。

    Attributes:
        id: 全局唯一标识，默认自动生成 UUID4。
        content: 记忆的自然语言内容，非空。
        timestamp: 记录时间（UTC），用于检索时的时间衰减加权。
        confidence: 事实置信度，取值 [0, 1]，初始默认 0.5（最大熵先验）。
        embedding: 内容的向量表示（可选），由检索层按需生成与填充。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "content": "用户偏好使用 pytest 而非 unittest 编写测试",
                "timestamp": "2026-09-15T09:00:00Z",
                "confidence": 0.85,
                "embedding": [0.012, -0.034, 0.101],
            }
        }
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="全局唯一标识（UUID4）",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="记忆的自然语言内容，不允许为空",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="记录时间（UTC 时区感知）",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="事实置信度，[0, 1] 区间，默认 0.5（无信息先验）",
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="内容的向量表示，由检索层按需填充",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        """确保 timestamp 带时区信息：naive datetime 一律按 UTC 处理。

        统计依据：时间衰减模型（指数加权）要求所有时间戳在同一时间基准上
        做差值计算，混入 naive/aware 会导致 TypeError 或错误的衰减权重。
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @field_validator("embedding")
    @classmethod
    def validate_embedding_values(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        """校验向量元素为有限浮点数（拒绝 NaN / Inf）。

        NaN 会污染向量相似度计算（cosine similarity 对 NaN 不报错但结果无意义），
        必须在入口处拦截。
        """
        if value is None:
            return value
        import math

        for dim in value:
            if not math.isfinite(dim):
                raise ValueError("embedding 中包含非有限值（NaN 或 Inf）")
        return value
