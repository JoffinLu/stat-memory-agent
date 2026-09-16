# =============================================================================
# vLLM 部署：微调后的 Qwen2.5-7B（QLoRA 适配器已构建期合并）
#
# 用法（在项目根目录）：
#   docker build -t sma-vllm -f deployment/Dockerfile .
#   docker run --gpus all -p 8000:8000 sma-vllm
#
# 构建前提：finetune_output/final_adapter/ 存在（DPO 训练产物）。
# 基座模型在构建期从 HF 下载（国内走 hf-mirror），合并后仅保留完整权重。
# =============================================================================

# 官方 vLLM 镜像：自带 OpenAI 兼容服务器（锁版本，Qwen2.5 支持完整）
FROM vllm/vllm-openai:v0.6.6

# 国内构建提速：HF 下载走镜像站（海外构建可删）
ENV HF_ENDPOINT=https://hf-mirror.com \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 合并脚本 + LoRA 适配器（构建上下文根 = 项目根目录）
COPY deployment/merge_adapter.py /build/merge_adapter.py
COPY finetune_output/final_adapter /adapter

# 构建期合并：base + LoRA adapter -> 完整权重
# （运行时无需 --enable-lora，无 adapter 装载开销，支持 tensor-parallel）
RUN python /build/merge_adapter.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --adapter /adapter \
        --output /models/qwen2.5-7b-sma \
    && rm -rf /build /adapter /root/.cache/huggingface/.lock

# 模型已内置于镜像，禁止运行时联网拉取
ENV HF_HUB_OFFLINE=1

# 运行时可调参数（见 docker-compose.yml）
ENV SERVED_MODEL_NAME=Qwen2.5-7B-Instruct \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90

EXPOSE 8000

# 官方镜像的 ENTRYPOINT 即 OpenAI 兼容服务器，这里只补参数
CMD ["--model", "/models/qwen2.5-7b-sma", \
     "--served-model-name", "Qwen2.5-7B-Instruct", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.90", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

# 健康检查：python 标准库探测（镜像内不一定有 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
