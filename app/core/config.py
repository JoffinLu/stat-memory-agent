"""全局配置管理。

所有统计模块的超参数集中在此，避免魔法数字散落各处：
- 混合检索权重（bm25_weight / vector_weight）
- 时间衰减系数（time_decay_lambda）
- SPC 控制图参数（spc_window_size / spc_sigma）

通过环境变量覆盖，前缀 SMA_，例如 SMA_LLM_API_KEY、SMA_SPC_SIGMA。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置。

    读取优先级：环境变量 > .env 文件 > 此处默认值。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SMA_",
        extra="ignore",
    )

    # ---------- 应用基础 ----------
    app_name: str = "Stat-Memory Agent"
    app_version: str = "0.1.0"
    debug: bool = False

    # ---------- LLM 客户端 ----------
    # 默认指向本地 vLLM 服务（deployment/Dockerfile 部署的微调模型）。
    # 切换云 API 只需环境变量覆盖，例如：
    #   SMA_LLM_BASE_URL=https://api.openai.com/v1
    #   SMA_LLM_MODEL_NAME=gpt-4o-mini
    #   SMA_LLM_API_KEY=sk-xxx
    llm_api_key: str = ""                 # 空 -> 离线模式；vLLM 用 "EMPTY" 占位
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model_name: str = "Qwen2.5-7B-Instruct"  # 与 vLLM --served-model-name 对齐
    llm_temperature: float = 0.2
    # vLLM 健康探测超时（秒）；0 -> 跳过探测
    llm_healthcheck_timeout: float = 2.0

    # ---------- 记忆系统 ----------
    chroma_persist_dir: str = "./data/chroma"
    embedding_dim: int = 1536

    # ---------- 混合检索 ----------
    retrieval_top_k: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    # 时间衰减 lambda：score *= exp(-lambda * age_days)
    time_decay_lambda: float = 0.01

    # ---------- 自纠错（统计过程控制） ----------
    # SPC 控制图滑动窗口大小（样本数）
    spc_window_size: int = 30
    # 控制限宽度（标准差倍数，3-sigma 为经典 Shewhart 控制图）
    spc_sigma: float = 3.0
    # SPRT 序贯检验的 I 类错误率上限
    sprt_alpha: float = 0.05
    # SPRT 序贯检验的 II 类错误率上限
    sprt_beta: float = 0.10
    # ---- SPRT 重试止损（with_retry 接入）----
    # 是否启用 SPRT 提前止损（False = 固定 max_retries 的朴素重试）
    sprt_retry_enabled: bool = True
    # H0：单次重试成功率基线（历史正常水平）
    sprt_retry_p0: float = 0.4
    # H1：退化水平（p1 < p0），连续失败证据累积到 Wald 上界即止损
    sprt_retry_p1: float = 0.15
    # 自纠错失败日志路径（JSON 数组追加写入）
    failure_log_path: str = "./logs/failure_log.json"


# 模块级单例：全局共享同一份配置
settings = Settings()
