"""微调数据管道 + QLoRA DPO 训练脚本（自进化闭环的最后一环）。

职责划分：
1. 数据管道（纯函数，零重依赖，离线可测）：
   读取 logs/failure_log.json -> 按任务分组 -> 提取成功/失败轨迹 ->
   构建偏好对 (Prompt, Chosen_Action, Rejected_Action) ->
   导出 ShareGPT 格式 JSON。

   偏好对语义（统计立场，勿破坏）：
   - 只有「最终成功」的重试链才产出偏好对 —— Rejected 取首次失败尝试
     （错误最原生态），Chosen 取 resolved_action（已被执行验证有效）；
   - 纯失败链（重试耗尽）没有经过验证的 Chosen 侧，绝不伪造
     （如硬凑"去掉报错参数"），只计入报告跳过数。脏偏好对会让
     DPO 学到反向信号，质量永远优先于数量。

2. 训练脚本（torch / transformers / peft / trl 懒加载）：
   Qwen2.5-7B-Instruct + 4bit NF4 (QLoRA) + DPOTrainer。
   未安装训练依赖时仅 train() 不可用，数据管道与单元测试不受影响。

CLI:
    # 仅导出偏好对数据集
    python -m app.agent.finetune --log logs/failure_log.json \
        --output data/finetune/preferences_sharegpt.json

    # 导出 + 启动训练（需 GPU 与训练依赖）
    python -m app.agent.finetune --log logs/failure_log.json \
        --output data/finetune/preferences_sharegpt.json --train
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================ 数据管道 ============================


@dataclass
class PreferencePair:
    """一条 DPO 偏好对。"""

    task_id: str
    prompt: str
    chosen: str
    rejected: str


@dataclass
class TrajectoryReport:
    """数据管道处理报告（透明可审计）。"""

    n_records: int = 0        # 日志总记录数
    n_tasks: int = 0          # 按任务分组后的任务数
    n_success: int = 0        # 最终成功的任务链数
    n_failure: int = 0        # 重试耗尽的失败任务链数
    n_pairs: int = 0          # 产出的偏好对数量
    n_skipped: int = 0        # 跳过的任务数（纯失败链 / 缺 resolved_action）
    output_path: str = ""


def load_records(log_path: str) -> List[Dict[str, Any]]:
    """容错读取失败日志（JSON 数组）。

    文件缺失 / 损坏 / 顶层非数组时返回空列表并告警 —— 数据管道
    的健壮性原则与 FailureLogger 一致：上游故障不炸下游。
    """
    path = Path(log_path)
    if not path.exists():
        logger.warning("失败日志不存在: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("失败日志读取失败（%s）: %s", exc, path)
        return []
    if not isinstance(data, list):
        logger.warning("失败日志顶层结构异常（非数组）: %s", path)
        return []
    return [rec for rec in data if isinstance(rec, dict) and rec.get("task_id")]


def group_by_task(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 task_id 分组，组内按 failed_step 升序（重试时间序）。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["task_id"])].append(rec)
    for chain in grouped.values():
        chain.sort(key=lambda r: r.get("failed_step", 0))
    return dict(grouped)


def _build_prompt(task_id: str, chain: List[Dict[str, Any]]) -> str:
    """从失败链构造偏好对的 Prompt（与 Replan Prompt 同源的上下文）。

    微调目标：让基座内化 Replan 的纠错能力 —— 因此 Prompt 结构
    必须与线上 Replan 请求一致，否则存在分布漂移。
    """
    first = chain[0]
    history_lines = "\n".join(
        f"第{rec.get('failed_step', '?')}次尝试失败: "
        f"{rec.get('error_reason', '未知错误')}"
        for rec in chain
    )
    return (
        "你是具备自纠错能力的智能体。以下工具调用刚刚失败，"
        "请根据错误信息与历史执行轨迹给出修正后的调用参数。\n\n"
        f"函数/任务：{task_id}\n"
        f"失败调用：{first.get('action', '（未记录）')}\n"
        f"最新错误信息：{first.get('error_reason', '未知错误')}\n"
        f"历史执行轨迹：\n{history_lines}\n\n"
        "请输出修正后的正确调用。"
    )


def build_preference_pairs(
    records: List[Dict[str, Any]],
) -> Tuple[List[PreferencePair], TrajectoryReport]:
    """从失败日志构建 DPO 偏好对。

    规则：
    - 按 task_id 分组；任一记录 final_success=True 即为成功链；
    - 成功链 -> Rejected = 首次失败尝试的 action，
      Chosen = resolved_action（链成功时由 with_retry 回填）；
    - 纯失败链 / 缺 resolved_action 的链 -> 跳过并计数（不伪造 Chosen）。
    """
    grouped = group_by_task(records)
    pairs: List[PreferencePair] = []
    n_success = 0
    n_failure = 0
    n_skipped = 0

    for task_id, chain in grouped.items():
        succeeded = any(rec.get("final_success") is True for rec in chain)
        if succeeded:
            n_success += 1
        else:
            n_failure += 1

        # Chosen 侧必须存在且来自被验证的成功调用
        resolved = next(
            (rec.get("resolved_action") for rec in chain if rec.get("resolved_action")),
            None,
        )
        first_action = chain[0].get("action")
        if not succeeded or not resolved or not first_action:
            n_skipped += 1
            continue

        pairs.append(
            PreferencePair(
                task_id=task_id,
                prompt=_build_prompt(task_id, chain),
                chosen=resolved,
                rejected=first_action,
            )
        )

    report = TrajectoryReport(
        n_records=len(records),
        n_tasks=len(grouped),
        n_success=n_success,
        n_failure=n_failure,
        n_pairs=len(pairs),
        n_skipped=n_skipped,
    )
    return pairs, report


def to_sharegpt(pairs: List[PreferencePair]) -> List[Dict[str, Any]]:
    """转换为 ShareGPT 格式（LLaMA-Factory DPO 变体）。

    每条：conversations 只含 human 轮（Prompt），
    chosen / rejected 为平级的 gpt 回复，由 DPOTrainer 消费。
    """
    return [
        {
            "conversations": [{"from": "human", "value": pair.prompt}],
            "chosen": {"from": "gpt", "value": pair.chosen},
            "rejected": {"from": "gpt", "value": pair.rejected},
        }
        for pair in pairs
    ]


def export_sharegpt(pairs: List[PreferencePair], output_path: str) -> Path:
    """导出 ShareGPT JSON（UTF-8，数组顶层）。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(to_sharegpt(pairs), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def run_data_pipeline(log_path: str, output_path: str) -> Tuple[List[PreferencePair], TrajectoryReport]:
    """一键数据管道：读取 -> 偏好对 -> 导出 -> 报告。"""
    records = load_records(log_path)
    pairs, report = build_preference_pairs(records)
    if pairs:
        out = export_sharegpt(pairs, output_path)
        report.output_path = str(out)
        logger.info("偏好对 %d 条已导出: %s", len(pairs), out)
    else:
        logger.warning("日志中无可用的成功链，未产出偏好对")
    return pairs, report


# ============================ 训练配置与脚本 ============================


@dataclass
class QLoRADPOConfig:
    """QLoRA + DPO 完整训练超参数（Qwen2.5-7B-Instruct 基准配置）。

    选型依据：
    - 4bit NF4 + double_quant：QLoRA 标准量化，7B 单卡 24G 可训；
    - lora_r=32 / alpha=64：偏好数据量小（百级），alpha=2r 补偿
      低秩容量不足；目标模块覆盖全部线性层（经验上优于仅 q/v）；
    - lr=5e-6：7B QLoRA DPO 的经验区间（1e-5 量级会把 KL 约束冲垮，
      表现为 chosen/rejected 分数同时下降）；beta=0.1 为 DPO 默认
      KL 惩罚强度；
    - per_device_batch=1 + grad_accum=16：有效 batch 16，小显存友好。
    """

    # ---------- 模型与量化 ----------
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True            # QLoRA 核心：4bit 权重加载
    bnb_quant_type: str = "nf4"          # NF4 分位数量化（QLoRA 论文最优）
    bnb_double_quant: bool = True        # 嵌套量化，再省 ~0.4 bit/权重
    bnb_compute_dtype: str = "bfloat16"  # 计算精度（反量化后的计算类型）

    # ---------- LoRA ----------
    lora_r: int = 32                     # 秩
    lora_alpha: int = 64                 # 缩放（经验：2r 起步）
    lora_dropout: float = 0.05
    # 覆盖注意力 + MLP 全部线性层
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ---------- DPO ----------
    dpo_beta: float = 0.1                # KL 惩罚系数（偏离参考模型的代价）
    max_length: int = 2048               # 序列总长上限
    max_prompt_length: int = 1024        # Prompt 截断上限

    # ---------- 训练超参数 ----------
    num_train_epochs: int = 3            # 偏好对少，3 轮足够（过多易过拟合）
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16   # 有效 batch = 16
    learning_rate: float = 5e-6          # DPO 对 lr 极敏感，宁小勿大
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True                    # Ampere+ 全开
    gradient_checkpointing: bool = True  # 用计算换显存
    optim: str = "paged_adamw_8bit"      # 分页优化器，防显存尖峰
    logging_steps: int = 1
    save_strategy: str = "epoch"
    output_dir: str = "finetune_output"
    seed: int = 42

    # ---------- 数据 ----------
    dataset_path: str = "data/finetune/preferences_sharegpt.json"


def train(config: QLoRADPOConfig) -> str:
    """执行 QLoRA + DPO 训练（需要 GPU 与训练依赖）。

    依赖：torch / transformers / peft / trl / datasets / bitsandbytes。
    训练依赖懒加载：未安装时本函数抛出带安装指引的 RuntimeError，
    数据管道与单元测试不受影响。

    Returns:
        适配器输出目录路径。
    """
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "训练依赖缺失，请先安装：pip install torch transformers peft "
            "trl datasets bitsandbytes accelerate（GPU 环境另需匹配的 CUDA 版本）"
        ) from exc

    # ---------- 数据 ----------
    data = json.loads(Path(config.dataset_path).read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"数据集为空: {config.dataset_path}")
    dataset = Dataset.from_list(data).train_test_split(test_size=0.1, seed=config.seed)

    # ---------- 量化与模型 ----------
    compute_dtype = getattr(torch, config.bnb_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_quant_type,
        bnb_4bit_double_quant=config.bnb_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False  # gradient checkpointing 下必须关闭
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # ---------- PEFT / DPO ----------
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    training_args = DPOConfig(
        output_dir=config.output_dir,
        beta=config.dpo_beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        seed=config.seed,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = str(Path(config.output_dir) / "final_adapter")
    trainer.save_model(adapter_dir)
    return adapter_dir


def main() -> None:
    """CLI 入口：数据管道必跑，训练按 --train 开关。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="失败日志 -> DPO 偏好对 -> QLoRA 微调")
    parser.add_argument("--log", default="logs/failure_log.json", help="失败日志路径")
    parser.add_argument("--output", default=QLoRADPOConfig.dataset_path, help="ShareGPT 输出路径")
    parser.add_argument("--train", action="store_true", help="导出后启动训练")
    args = parser.parse_args()

    pairs, report = run_data_pipeline(args.log, args.output)
    print(
        f"日志 {report.n_records} 条 / 任务 {report.n_tasks} 个 "
        f"（成功链 {report.n_success}，失败链 {report.n_failure}，"
        f"跳过 {report.n_skipped}）-> 偏好对 {report.n_pairs} 条"
    )
    if report.output_path:
        print(f"已导出: {report.output_path}")

    if args.train:
        if not pairs:
            raise SystemExit("无偏好对可训练，请先积累自纠错数据")
        adapter_dir = train(QLoRADPOConfig(dataset_path=args.output))
        print(f"训练完成，LoRA 适配器: {adapter_dir}")


if __name__ == "__main__":
    main()
