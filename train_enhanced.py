#!/usr/bin/env python3
"""
Enhanced AML/KYC curriculum trainer with logging and eval integration.
"""
from __future__ import annotations
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["TORCH_DISABLE_FLASH_ATTENTION"] = "1"

import unsloth
# Disable flash attention at import level
try:
    unsloth.disable_fast_attention = True
except:
    pass
import json, math, random, subprocess, shutil, logging, datetime, time
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

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger = logging.getLogger('aml_kyc_trainer')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

DEFAULT_MODEL = "unsloth/llama-3-8b-bnb-4bit"
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

SYSTEM_PROMPT = (
    "You are an AML/KYC analyst. Provide careful reasoning citing specific fields (DOB, nationality, IDs, "
    "aliases, ownership links). Conclude with a one-line decision: MATCH, NO_MATCH, or EDGE with rationale."
)

@dataclass
class PlateauEarlyStop:
    patience_epochs: int = 3
    min_rel_improve: float = 0.05
    
    def __post_init__(self):
        self.best = float('inf')
        self.wait = 0
        self.stopped = False

    def on_epoch_end(self, val_loss: float) -> bool:
        if val_loss < self.best * (1 - self.min_rel_improve):
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience_epochs:
            self.stopped = True
        
        return self.stopped

def format_example(ex: Dict[str, Any], use_cot: bool) -> str:
    user_block = f"### Human:\n{ex['instruction']}\n{ex['input']}\n\n"
    if use_cot and ex.get("cot"):
        assistant = f"### Assistant:\n{ex['cot'].strip()}\n\n{ex['output'].strip()}"
    else:
        assistant = f"### Assistant:\n{ex['output'].strip()}"
    return user_block + assistant

def build_packed_dataset(data_path: str, split_filter: str, use_cot: bool, seed: int = 42) -> DatasetDict:
    ds = load_dataset("json", data_files=data_path)["train"]
    ds_phase = ds.filter(lambda ex: ex.get("difficulty","").lower() == split_filter.lower())
    ds_phase = ds_phase.train_test_split(test_size=min(0.2, max(0.1, 0.2)), seed=seed)
    def map_fmt(ex):
        ex["text"] = format_example(ex, use_cot=use_cot)
        return ex
    ds_phase = ds_phase.map(map_fmt, remove_columns=[c for c in ds_phase["train"].column_names if c != "text"])
    return DatasetDict({"train": ds_phase["train"], "validation": ds_phase["test"]})

def train_phase(
    model, tokenizer, ds_phase: DatasetDict, args, phase_name: str, output_dir: str, logger
) -> Dict[str, Any]:
    logger.info(f"=== Starting {phase_name} phase training ===")
    logger.info(f"Training samples: {len(ds_phase['train'])}, Validation samples: {len(ds_phase['validation'])}")
    
    os.makedirs(output_dir, exist_ok=True)

    steps_per_epoch = max(1, len(ds_phase["train"]) // max(1, args.batch_size))
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=getattr(args, f"epochs_{phase_name}"),
        learning_rate=args.lr if phase_name == "easy" else (args.lr * 0.7 if phase_name == "medium" else args.lr * 0.5),
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
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
        tokenizer=tokenizer,
        train_dataset=ds_phase["train"],
        eval_dataset=ds_phase["validation"],
        args=training_args,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
    )

    stopper = PlateauEarlyStop(patience_epochs=3, min_rel_improve=0.05)

    history = []
    for epoch in range(int(training_args.num_train_epochs)):
        logger.info(f"=== Epoch {epoch + 1}/{int(training_args.num_train_epochs)} ({phase_name}) ===")
        start_time = time.time()
        
        trainer.train(resume_from_checkpoint=None)
        eval_metrics = trainer.evaluate()
        epoch_time = time.time() - start_time
        
        eval_loss = float(eval_metrics.get("eval_loss", float("inf")))
        train_loss = float(eval_metrics.get("train_loss", float("inf")))
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        logger.info(f"  Training loss: {train_loss:.4f}")
        logger.info(f"  Validation loss: {eval_loss:.4f}")
        
        history.append({
            "epoch": epoch + 1, 
            "eval_loss": eval_loss,
            "train_loss": train_loss,
            "epoch_time": epoch_time,
            "eval_metrics": eval_metrics
        })
        
        state_path = os.path.join(output_dir, "trainer_state.json")
        with open(state_path, "w") as f:
            json.dump({"history": history, "last_eval": eval_metrics}, f, indent=2)
        if stopper.on_epoch_end(eval_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    
    phase_metrics = {
        "history": history, 
        "best_eval_loss": stopper.best,
        "epochs_completed": len(history),
        "early_stopped": stopper.stopped,
        "total_training_time": sum(h.get("epoch_time", 0) for h in history)
    }
    
    logger.info(f"=== {phase_name} phase completed ===")
    logger.info(f"  Best validation loss: {stopper.best:.4f}")
    logger.info(f"  Epochs completed: {len(history)}")
    logger.info(f"  Early stopped: {stopper.stopped}")
    logger.info(f"  Total training time: {phase_metrics['total_training_time']:.2f}s")
    
    return phase_metrics

def run_eval_judge(model_path: str, dataset_path: str, provider: str, output_path: str, logger) -> Dict[str, Any]:
    try:
        logger.info(f"Running eval_judge on {model_path} with {provider}")
        cmd = [
            "python", "eval_judge.py",
            "--model_path", model_path,
            "--dataset", dataset_path,
            "--provider", provider,
            "--tau", "0.5",
            "--out", output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            logger.info(f"Eval judge completed successfully, results saved to {output_path}")
            logger.info(f"Eval judge stdout: {result.stdout[-500:]}")
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    return json.load(f)
        else:
            logger.error(f"Eval judge failed with return code {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Eval judge timed out after 30 minutes")
    except Exception as e:
        logger.error(f"Error running eval judge: {e}")
    
    return {}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs_easy", type=int, default=2)
    parser.add_argument("--epochs_medium", type=int, default=2)  
    parser.add_argument("--epochs_hard", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--use_cot", type=bool, default=True)
    parser.add_argument("--judge_provider", choices=["anthropic", "perplexity"], default="anthropic")
    parser.add_argument("--run_judge", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logger = setup_logging(args.output_dir)
    logger.info("=== AML/KYC Enhanced Training Started ===")
    logger.info(f"Training started at: {datetime.datetime.now()}")
    logger.info("=== Training Configuration ===")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Loading base model: {args.model_id}")
    model_start_time = time.time()
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_len,
            load_in_4bit=False,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        
        model_load_time = time.time() - model_start_time
        logger.info(f"Model loaded successfully in {model_load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load model {args.model_id}: {e}")
        logger.info("Falling back to CPU training...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_len,
            load_in_4bit=False,
            dtype=None,
            attn_implementation="sdpa",
        )
        model_load_time = time.time() - model_start_time
        logger.info(f"Model loaded in CPU mode in {model_load_time:.2f}s")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    
    # Ensure tokenizer is properly configured for SFTTrainer
    if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length > args.max_seq_len:
        tokenizer.model_max_length = args.max_seq_len

    logger.info("Applying LoRA configuration")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    logger.info("LoRA applied successfully")

    phases = ["easy", "medium", "hard"]
    all_metrics = {}
    total_training_start = time.time()

    for phase in phases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting phase: {phase.upper()}")
        logger.info(f"{'='*60}")
        
        phase_out = os.path.join(args.output_dir, f"phase_{phase}")
        full_train_path = os.path.join(args.data_dir, "train.jsonl")
        
        if not os.path.exists(full_train_path):
            logger.error(f"Training data not found at {full_train_path}")
            continue
        logger.info(f"Building dataset for {phase} phase from {full_train_path}")
        try:
            ds_phase = build_packed_dataset(full_train_path, split_filter=phase, use_cot=args.use_cot, seed=args.seed)
        except Exception as e:
            logger.error(f"Failed to build dataset for {phase}: {e}")
            continue
        
        if len(ds_phase["train"]) == 0:
            logger.warning(f"No training samples found for {phase} phase, skipping...")
            continue
        
        try:
            metrics = train_phase(model, tokenizer, ds_phase, args, phase_name=phase, output_dir=phase_out, logger=logger)
            all_metrics[phase] = metrics
        except Exception as e:
            logger.error(f"Training failed for {phase} phase: {e}")
            continue

        if args.run_judge:
            logger.info(f"Running LLM-as-Judge evaluation for {phase} phase")
            val_path = os.path.join(args.data_dir, "val.jsonl")
            model_path = phase_out  # Use the saved model directory
            eval_output = os.path.join(phase_out, f"judge_eval_{phase}.json")
            
            if os.path.exists(model_path) and os.path.exists(val_path):
                judge_metrics = run_eval_judge(model_path, val_path, args.judge_provider, eval_output, logger)
                if judge_metrics:
                    all_metrics[phase]['judge_eval'] = judge_metrics
                    logger.info(f"Judge evaluation completed for {phase}")
                else:
                    logger.warning(f"Judge evaluation failed for {phase}")
            else:
                logger.warning(f"Skipping judge eval - missing model ({model_path}) or validation data ({val_path})")

    total_training_time = time.time() - total_training_start
    
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    training_summary = {
        "training_config": vars(args),
        "model_load_time": model_load_time,
        "total_training_time": total_training_time,
        "phases_completed": list(all_metrics.keys()),
        "phase_metrics": all_metrics,
        "final_model_path": os.path.join(args.output_dir, "phase_hard"),
        "training_completed_at": datetime.datetime.now().isoformat(),
        "training_successful": len(all_metrics) > 0
    }
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")
    integration_path = os.path.join(args.output_dir, "integration_snippet.py")
    snippet = f'''# Load fine-tuned AML/KYC model
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Load the final trained model
model = AutoPeftModelForCausalLM.from_pretrained("{args.output_dir}/phase_hard")
tokenizer = AutoTokenizer.from_pretrained("{args.model_id}")

# Set up for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Example inference function
def analyze_aml_case(instruction: str, case_input: str) -> str:
    prompt = f"### Human:\\n{{instruction}}\\n{{case_input}}\\n\\n### Assistant:\\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Assistant:\\n")[-1].strip()

'''
    with open(integration_path, "w") as f:
        f.write(snippet)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
    logger.info(f"Phases completed: {list(all_metrics.keys())}")
    logger.info(f"Training summary: {summary_path}")
    logger.info(f"Integration example: {integration_path}")
    for phase, metrics in all_metrics.items():
        best_loss = metrics.get('best_eval_loss', float('inf'))
        epochs = metrics.get('epochs_completed', 0)
        logger.info(f"Phase {phase}: Best Loss {best_loss:.4f}, Epochs {epochs}")
        if 'judge_eval' in metrics:
            judge_acc = metrics['judge_eval'].get('accuracy', 'N/A')
            logger.info(f"  Judge Accuracy: {judge_acc}")
    logger.info("Training log saved to training.log")
    logger.info("=== AML/KYC Enhanced Training Finished ===")

if __name__ == "__main__":
    main()