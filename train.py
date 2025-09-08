#!/usr/bin/env python3
"""
Unsloth curriculum trainer for AML/KYC (easy → medium → hard).
Supports LoRA, early stopping, and optional LLM-as-Judge evaluation.
"""
from __future__ import annotations
import os, json, math, random, subprocess, shutil
from dataclasses import dataclass
from typing import Dict, Any, List
from dotenv import load_dotenv

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict, Dataset
from peft import AutoPeftModelForCausalLM
from unsloth import FastLanguageModel
from tqdm import tqdm

load_dotenv()

DEFAULT_MODEL = "meta-llama/Llama-3.3-8B-Instruct"
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

SYSTEM_PROMPT = (
    "You are an AML/KYC analyst. Provide careful reasoning citing specific fields (DOB, nationality, IDs, "
    "aliases, ownership links). Conclude with a one-line decision: MATCH, NO_MATCH, or EDGE with rationale."
)

def format_example(ex: Dict[str, Any], use_cot: bool = True) -> str:
    user_block = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
    )
    if use_cot:
        assistant = f"### Assistant:\n{ex['cot'].strip()}\n\nFinal decision: {ex['output'].strip()}"
    else:
        assistant = f"### Assistant:\n{ex['output'].strip()}"
    return user_block + assistant

def build_packed_dataset(data_path: str, split_filter: str, use_cot: bool, seed: int = 42) -> DatasetDict:
    ds = load_dataset("json", data_files=data_path)["train"]
    ds_phase = ds.filter(lambda ex: ex.get("difficulty","").lower() == split_filter.lower())
    ds_phase = ds_phase.train_test_split(test_size=min(0.1, max(0.1, 0.1)), seed=seed)
    def map_fmt(ex):
        ex["text"] = format_example(ex, use_cot=use_cot)
        return ex
    ds_phase = ds_phase.map(map_fmt, remove_columns=[c for c in ds_phase["train"].column_names if c != "text"])
    return DatasetDict({"train": ds_phase["train"], "validation": ds_phase["test"]})

class PlateauEarlyStop:
    def __init__(self, patience_epochs: int = 3, min_rel_improve: float = 0.05):
        self.patience = patience_epochs
        self.min_rel = min_rel_improve
        self.best = float("inf")
        self.worse_epochs = 0

    def on_epoch_end(self, eval_loss: float) -> bool:
        if math.isfinite(eval_loss) and eval_loss < self.best * (1 - self.min_rel):
            self.best = eval_loss
            self.worse_epochs = 0
        else:
            self.worse_epochs += 1
        return self.worse_epochs >= self.patience

def train_phase(
    model, tokenizer, ds_phase: DatasetDict, args, phase_name: str, output_dir: str
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    steps_per_epoch = max(1, len(ds_phase["train"]) // max(1, args.batch_size))
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=getattr(args, f"epochs_{phase_name}"),
        learning_rate=args.lr if phase_name == "easy" else (args.lr * 0.7 if phase_name == "medium" else args.lr * 0.5),
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=max(1, steps_per_epoch // 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_phase["train"],
        eval_dataset=ds_phase["validation"],
        data_collator=collator,
        args=training_args,
    )

    stopper = PlateauEarlyStop(patience_epochs=3, min_rel_improve=0.05)

    history = []
    for epoch in range(int(training_args.num_train_epochs)):
        trainer.train(resume_from_checkpoint=None)
        eval_metrics = trainer.evaluate()
        eval_loss = float(eval_metrics.get("eval_loss", float("inf")))
        history.append({"epoch": epoch + 1, "eval_loss": eval_loss})
        with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
            json.dump({"history": history, "last_eval": eval_metrics}, f, indent=2)
        if stopper.on_epoch_end(eval_loss):
            print(f"[{phase_name}] Early stop triggered.")
            break

    trainer.save_model(output_dir)
    return {"history": history, "best_eval_loss": stopper.best}

def merge_and_save(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    FastLanguageModel.save_pretrained_merged(model, tokenizer, out_dir, max_shard_size="5GB")
    print(f"Merged weights saved to: {out_dir}")

def push_to_hub_if_needed(out_dir: str, repo_id: str):
    if not repo_id:
        return
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    try:
        create_repo(repo_id, exist_ok=True, private=True)
    except Exception:
        pass
    api.upload_folder(folder_path=out_dir, repo_id=repo_id)
    print(f"Pushed to HF Hub: {repo_id}")

def run_judge(
    adapter_or_merged_path: str,
    val_path: str,
    provider: str,
    out_report: str,
    max_new_tokens: int = 256,
):
    cmd = [
        "python", "eval_judge.py",
        "--model_path", adapter_or_merged_path,
        "--dataset", val_path,
        "--provider", provider,
        "--out", out_report,
        "--max_new_tokens", str(max_new_tokens),
    ]
    print(f"Running judge: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--val_file", type=str, default="val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/llm-aml-curriculum")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs_easy", type=int, default=2)
    parser.add_argument("--epochs_medium", type=int, default=2)
    parser.add_argument("--epochs_hard", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_cot", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--merge_lora", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--push_to_hub", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--hub_repo", type=str, default="")
    parser.add_argument("--judge_provider", type=str, choices=["anthropic", "perplexity"], default="anthropic")
    parser.add_argument("--run_judge", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rope_scale", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Loading base model: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=TARGET_MODULES,
    )

    try:
        if hasattr(model.config, "rope_scaling") and args.rope_scale and args.rope_scale != 1.0:
            model.config.rope_scaling = {"type": "linear", "factor": args.rope_scale}
            print(f"Applied RoPE scaling: {model.config.rope_scaling}")
    except Exception:
        pass

    phases = ["easy", "medium", "hard"]
    full_train_path = os.path.join(args.data_dir, args.train_file)
    full_val_path = os.path.join(args.data_dir, args.val_file)

    os.makedirs(args.output_dir, exist_ok=True)
    config_dump = vars(args).copy()
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

    results = {}

    for phase in phases:
        print(f"\n=== Phase: {phase} ===")
        ds_phase = build_packed_dataset(full_train_path, split_filter=phase, use_cot=args.use_cot, seed=args.seed)
        phase_out = os.path.join(args.output_dir, f"phase_{phase}")
        metrics = train_phase(model, tokenizer, ds_phase, args, phase_name=phase, output_dir=phase_out)
        results[phase] = metrics

        if args.run_judge:
            judge_out = os.path.join(args.output_dir, "judge_reports", f"phase_{phase}.json")
                run_judge(phase_out, full_val_path, provider=args.judge_provider, out_report=judge_out)

    with open(os.path.join(args.output_dir, "curriculum_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    final_adapter = os.path.join(args.output_dir, "adapter")
    if not os.path.exists(final_adapter):
        shutil.copytree(os.path.join(args.output_dir, "phase_hard"), final_adapter, dirs_exist_ok=True)
    print(f"Saved final LoRA adapter at: {final_adapter}")

    if args.merge_lora:
        merged_dir = os.path.join(args.output_dir, "merged")
        merge_and_save(model, tokenizer, merged_dir)

    if args.push_to_hub:
        push_to_hub_if_needed(
            out_dir=os.path.join(args.output_dir, "merged" if args.merge_lora else "adapter"),
            repo_id=args.hub_repo,
        )

    snippet = f"""
# Transformers inference (merged or adapter)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "{os.path.abspath(os.path.join(args.output_dir, 'merged' if args.merge_lora else 'adapter'))}"
tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "### Instruction:\\nDecide MATCH/NO_MATCH/EDGE...\\n\\n### Input:\\n..."
inputs = tok(prompt, return_tensors="pt").to(mdl.device)
out = mdl.generate(**inputs, max_new_tokens=256, temperature=0.0)
print(tok.decode(out[0], skip_special_tokens=True))
"""
    with open(os.path.join(args.output_dir, "INTEGRATION_SNIPPET.py"), "w") as f:
        f.write(snippet)

    print("Done.")

if __name__ == "__main__":
    main()
