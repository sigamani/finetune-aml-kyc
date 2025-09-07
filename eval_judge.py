#!/usr/bin/env python3
"""
LLM-as-Judge evaluation (Anthropic Sonnet 3.5 or Perplexity).
- Samples 100 items (stratified by label & difficulty) from a JSONL dataset.
- Runs the fine-tuned model to produce answers.
- Asks the Judge LLM to grade decision correctness, sufficiency, and hallucinations.
- Emits accuracy, precision/recall/F1 (macro & per-class), and rubric mean.
Caching + retries included.

Usage:
  python eval_judge.py \
    --model_path ./outputs/llama33-8b-aml-curriculum/adapter \
    --dataset ./data/val.jsonl \
    --provider anthropic \
    --tau 0.5 \
    --out ./outputs/judge_reports/phase_easy.json

Env:
  ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default: claude-3-5-sonnet-20240620)
  PERPLEXITY_API_KEY, PPLX_MODEL (default: llama-3.1-70b-instruct)
"""
from __future__ import annotations
import os, json, math, hashlib, random, re, time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datasets import load_dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from pydantic import BaseModel, Field
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

# ----------------- Schema -----------------
class Item(BaseModel):
    instruction: str
    input: str
    cot: str
    output: str
    difficulty: str
    label: str

# ----------------- Judge adapters -----------------
class JudgeBase:
    def __init__(self, cache_dir: str = "./.cache/judge"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _cache_get(self, prompt: str) -> str | None:
        fp = os.path.join(self.cache_dir, self._cache_key(prompt) + ".json")
        if os.path.exists(fp):
            try:
                with open(fp, "r") as f:
                    return f.read()
            except Exception:
                return None
        return None

    def _cache_set(self, prompt: str, response: str):
        fp = os.path.join(self.cache_dir, self._cache_key(prompt) + ".json")
        with open(fp, "w") as f:
            f.write(response)

    def build_prompt(self, case: Item, model_answer: str) -> str:
        return (
            "You are scoring an AML/KYC model's answer.\n"
            "Return STRICT JSON with keys: decision (match|no_match|edge), "
            "sufficiency (0|0.5|1), hallucination (true|false), "
            "confidence (0..1).\n"
            "Rules:\n"
            "- decision = what the model committed to.\n"
            "- sufficiency: did the model cite >=2 concrete fields? (0, 0.5, or 1)\n"
            "- hallucination: true if model invented fields not present in the input.\n"
            "- confidence: your confidence in the decision (0..1).\n\n"
            f"Task: {case.instruction}\n"
            f"Input:\n{case.input}\n\n"
            f"Model Answer:\n{model_answer}\n\n"
            "Respond as JSON only."
        )

    def parse_json(self, text: str) -> Dict[str, Any]:
        # Try to extract JSON
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Last resort
        return {"decision": "edge", "sufficiency": 0, "hallucination": False, "confidence": 0.5}

    def judge(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

class AnthropicJudge(JudgeBase):
    def __init__(self, cache_dir: str = "./.cache/judge"):
        super().__init__(cache_dir)
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError("Install anthropic to use the Anthropic judge.") from e
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20), reraise=True)
    def judge(self, prompt: str) -> Dict[str, Any]:
        cached = self._cache_get(prompt)
        if cached:
            return self.parse_json(cached)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"])
        self._cache_set(prompt, text)
        return self.parse_json(text)

class PerplexityJudge(JudgeBase):
    def __init__(self, cache_dir: str = "./.cache/judge"):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set.")
        self.model = os.environ.get("PPLX_MODEL", "llama-3.1-70b-instruct")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.client = httpx.Client(timeout=60)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20), reraise=True)
    def judge(self, prompt: str) -> Dict[str, Any]:
        cached = self._cache_get(prompt)
        if cached:
            return self.parse_json(cached)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = self.client.post(self.base_url, headers=headers, json=payload)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        self._cache_set(prompt, text)
        return self.parse_json(text)

# ----------------- Model inference -----------------
class InferenceModel:
    def __init__(self, model_path: str, max_new_tokens: int = 256, device: str | None = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.max_new_tokens = max_new_tokens
        self.device = device

    @torch.no_grad()
    def generate_answer(self, case: Item, with_cot: bool = True) -> str:
        prompt = (
            "### Instruction:\n"
            f"{case.instruction}\n\n"
            "### Input:\n"
            f"{case.input}\n\n"
            "### Assistant:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the assistant part after the last marker
        ans = text.split("### Assistant:")[-1].strip()
        return ans

# ----------------- Sampling -----------------
def stratified_sample(items: List[Item], n: int = 100) -> List[Item]:
    by_bucket = defaultdict(list)
    for it in items:
        key = (it.label, it.difficulty)
        by_bucket[key].append(it)

    # allocate proportionally
    total = len(items)
    plan = {}
    remaining = n
    keys = list(by_bucket.keys())
    for i, k in enumerate(keys):
        proportion = len(by_bucket[k]) / total
        take = int(round(proportion * n))
        if i == len(keys) - 1:
            take = remaining
        take = min(take, len(by_bucket[k]))
        plan[k] = take
        remaining -= take

    sampled = []
    for k, take in plan.items():
        pool = by_bucket[k]
        random.shuffle(pool)
        sampled.extend(pool[:take])
    random.shuffle(sampled)
    return sampled[:n]

# ----------------- Metrics -----------------
def compute_metrics(golds: List[str], preds: List[str], rubric_scores: List[float]) -> Dict[str, Any]:
    # Normalize labels
    norm = lambda x: x.lower().strip()
    y_true = [norm(x) for x in golds]
    y_pred = [norm(x) for x in preds]
    labels = ["match", "no_match", "edge"]
    prf_macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prf_match = precision_recall_fscore_support(y_true, y_pred, labels=["match"], average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))
    return {
        "accuracy": acc,
        "precision_macro": prf_macro[0],
        "recall_macro": prf_macro[1],
        "f1_macro": prf_macro[2],
        "precision_match": prf_match[0],
        "recall_match": prf_match[1],
        "f1_match": prf_match[2],
        "rubric_mean": float(sum(rubric_scores) / max(1, len(rubric_scores))),
        "report": report,
    }

# ----------------- Main -----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--provider", type=str, choices=["anthropic", "perplexity"], default="anthropic")
    parser.add_argument("--tau", type=float, default=0.5, help="Confidence threshold (used if provider returns it)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Load items
    ds = load_dataset("json", data_files={"val": args.dataset})["val"]
    items = [Item(**{k: ds[i][k] for k in ["instruction", "input", "cot", "output", "difficulty", "label"]}) for i in range(len(ds))]
    pool = stratified_sample(items, n=args.samples)

    infer = InferenceModel(args.model_path, max_new_tokens=args.max_new_tokens)

    judge = AnthropicJudge() if args.provider == "anthropic" else PerplexityJudge()

    golds, preds, rubric_scores, raw = [], [], [], []
    for case in tqdm(pool, desc="Judging"):
        model_answer = infer.generate_answer(case)
        jprompt = judge.build_prompt(case, model_answer)
        jres = judge.judge(jprompt)
        # Decide final predicted label
        pred = str(jres.get("decision", "edge")).lower().strip()
        conf = float(jres.get("confidence", 0.5))
        # Optional thresholding; if confidence too low, bucket to 'edge'
        if conf < args.tau:
            pred = "edge"

        # Rubric: score 0 if hallucination; else mean of (decision_correct {0,1}, sufficiency {0,0.5,1})
        decision_correct = 1.0 if pred == case.label.lower() else 0.0
        if bool(jres.get("hallucination", False)):
            rubric = 0.0
        else:
            suff = float(jres.get("sufficiency", 0))
            rubric = (decision_correct + (suff / 1.0)) / 2.0

        golds.append(case.label)
        preds.append(pred)
        rubric_scores.append(rubric)
        raw.append({
            "input": case.input, "gold": case.label, "model_answer": model_answer, "judge": jres,
            "pred": pred, "rubric": rubric
        })

    metrics = compute_metrics(golds, preds, rubric_scores)
    out = {"provider": args.provider, "tau": args.tau, "samples": len(pool), "metrics": metrics, "details": raw}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
