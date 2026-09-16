"""LLM 客户端初始化（懒加载单例，默认指向本地 vLLM 服务）。

设计要点：
1. 懒加载 —— langchain 依赖较重，在函数体内延迟 import，
   保证 `import app` 零重量依赖。
2. 未配置 API key 时返回 None 并给出明确警告，而不是抛异常 ——
   统计模块（记忆/检索/自纠错）必须能在离线模式下单测。
3. 默认端点为本地 vLLM（deployment/Dockerfile 部署的微调模型）：
   vLLM 暴露 OpenAI 兼容 API（/v1/chat/completions），LangChain
   ChatOpenAI 直接可用；vLLM 不校验 key，用 "EMPTY" 占位。
4. 初始化后探测一次 /health，服务不可达时告警但不阻断 ——
   调用失败在业务层（fail-open 降级）自然处理，初始化失败
   不应让进程起不来。
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlsplit

from app.core.config import settings

logger = logging.getLogger(__name__)

_llm_client: Optional[Any] = None


def ping_vllm(base_url: str, timeout: Optional[float] = None) -> bool:
    """探测 vLLM 服务健康（GET {base}/health）。

    Args:
        base_url: OpenAI 兼容端点根，如 http://localhost:8000/v1。
        timeout: 超时秒数；None 时用 settings.llm_healthcheck_timeout，
            <=0 直接返回 False（跳过探测）。

    Returns:
        服务可达返回 True；超时/连接拒绝/HTTP 错误返回 False。
    """
    if timeout is None:
        timeout = settings.llm_healthcheck_timeout
    if timeout <= 0:
        return False

    # /health 挂在端点根的上一级（.../v1 -> .../health）
    parts = urlsplit(base_url)
    root = f"{parts.scheme}://{parts.netloc}"
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("vLLM 健康探测失败（%s）: %s", root, exc)
        return False


def get_llm_client() -> Optional[Any]:
    """获取 LLM 客户端单例（LangChain ChatOpenAI -> 本地 vLLM）。

    Returns:
        配置好的客户端实例；若 API key 未配置则返回 None（离线模式）。

    Note:
        - 端点默认 http://localhost:8000/v1（vLLM），云 API 用
          SMA_LLM_* 环境变量覆盖；
        - Token 用量统计（SPC 成本图）在阶段四接入 invoke 回调。
    """
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    if not settings.llm_api_key:
        logger.warning("SMA_LLM_API_KEY 未配置，运行于离线模式（统计模块不受影响）")
        return None

    try:
        # 延迟导入：仅在实际需要 LLM 时才加载 langchain
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
    except ImportError:
        logger.warning("langchain-openai 未安装，运行于离线模式")
        return None

    _llm_client = ChatOpenAI(
        api_key=settings.llm_api_key or "EMPTY",  # vLLM 不校验，占位即可
        base_url=settings.llm_base_url,
        model=settings.llm_model_name,
        temperature=settings.llm_temperature,
    )
    logger.info(
        "LLM 客户端初始化完成: model=%s base_url=%s",
        settings.llm_model_name,
        settings.llm_base_url,
    )

    # 启动探测：不可达只告警，不阻断（业务层自有降级路径）
    if not ping_vllm(settings.llm_base_url):
        logger.error(
            "vLLM 服务不可达（%s）。请先部署: "
            "docker compose -f deployment/docker-compose.yml up -d --build",
            settings.llm_base_url,
        )
    return _llm_client


def chat_completion_via_vllm(
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = 512,
) -> str:
    """直接走 vLLM HTTP API 的轻量补全（不经 langchain，零重依赖）。

    用途：langchain 未安装但 vLLM 在跑的环境（如纯部署机上的
    评估/生成脚本），与 get_llm_client() 的消息格式兼容
    （[{"role": ..., "content": ...}] 或 LangChain 消息对象）。

    Args:
        messages: 消息列表。
        temperature: 覆盖默认温度；None 用 settings.llm_temperature。
        max_tokens: 生成长度上限。

    Returns:
        首个 choice 的文本内容。

    Raises:
        RuntimeError: HTTP 非 200 或响应结构异常（调用方按各自
            fail-open/fail-closed 语义处理）。
    """
    # LangChain 消息 .type 为 human/ai，OpenAI 协议要求 user/assistant
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    normalized = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        if isinstance(m, dict)
        else {
            "role": role_map.get(getattr(m, "type", "user"), "user"),
            "content": getattr(m, "content", ""),
        }
        for m in messages
    ]
    payload = {
        "model": settings.llm_model_name,
        "messages": normalized,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            req, timeout=settings.llm_healthcheck_timeout * 30 or 60
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vLLM 调用失败: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"vLLM 响应结构异常: {data}") from exc
