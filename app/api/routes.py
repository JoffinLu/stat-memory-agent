"""API 路由定义。

当前（阶段一）仅提供：
- POST /api/v1/memory/facts —— MemoryFact 模型的校验回显端点，
  用于验证数据模型在 HTTP 层的序列化 / 反序列化行为。

阶段二起，此路由将接入真正的记忆写入服务。
"""

from fastapi import APIRouter

from app.memory.models import MemoryFact

router = APIRouter(prefix="/api/v1", tags=["memory"])


@router.post("/memory/facts", response_model=MemoryFact, status_code=201)
async def create_memory_fact(fact: MemoryFact) -> MemoryFact:
    """创建一条记忆事实（阶段一：校验并回显）。

    请求体经 Pydantic 完整校验（confidence 边界、timestamp 时区、
    embedding 有限性），校验通过后原样返回（含服务端补全的默认值：
    id、timestamp、confidence）。

    Args:
        fact: 记忆事实请求体。

    Returns:
        服务端补全默认值后的完整 MemoryFact。
    """
    return fact
