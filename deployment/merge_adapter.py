# -*- coding: utf-8 -*-
"""构建期合并 LoRA 适配器到基座模型（QLoRA -> 部署权重的最后一公里）。

在 vLLM 官方镜像内执行（也可本地使用）：
    python merge_adapter.py --base Qwen/Qwen2.5-7B-Instruct \
        --adapter /adapter --output /models/qwen2.5-7b-sma

为什么构建期合并而非运行时 --enable-lora：
1. 推理路径无 adapter 装载开销，吞吐更高；
2. 支持 tensor-parallel（vLLM 的 LoRA 分支对 TP 支持受限）；
3. 线上少一个运行时故障点（adapter 文件缺失/版本不匹配）。
"""

import argparse
import gc
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("merge_adapter")


def merge(base: str, adapter: str, output: str, device: str = "cpu") -> Path:
    """加载基座 + 适配器，merge_and_unload 后保存完整权重。

    Args:
        base: 基座模型（HF id 或本地路径）。
        adapter: PEFT LoRA 适配器目录（finetune 产物）。
        output: 合并后模型输出目录。
        device: 合并设备；构建期用 cpu（内存换显存，镜像内无 GPU 要求）。

    Returns:
        输出目录路径。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("加载基座模型: %s", base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map=device,          # cpu：构建期不依赖 GPU
        low_cpu_mem_usage=True,
    )

    logger.info("加载 LoRA 适配器: %s", adapter)
    model = PeftModel.from_pretrained(model, adapter)

    logger.info("执行 merge_and_unload ...")
    model = model.merge_and_unload()

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    # tokenizer 随基座保存：部署目录自成一体，服务器无需再读 HF
    logger.info("保存 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(out_dir)

    # 释放构建期内存（镜像层内不残留无用缓存）
    del model, tokenizer
    gc.collect()

    logger.info("合并完成: %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器到基座模型")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", required=True, help="LoRA 适配器目录")
    parser.add_argument("--output", required=True, help="合并模型输出目录")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not Path(args.adapter).exists():
        raise SystemExit(f"适配器目录不存在: {args.adapter}")
    merge(args.base, args.adapter, args.output, args.device)


if __name__ == "__main__":
    main()
