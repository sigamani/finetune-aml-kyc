#!/usr/bin/env python3
"""
Programmatically generate synthetic AML/KYC curriculum data (easy/medium/hard),
validate schema, dedupe, inject light noise, and write train/val JSONL.

Providers: Anthropic (default Sonnet 3.5) or Perplexity.
Caching + retries included.

Usage:
  python generate_synthetics.py \
    --provider anthropic \
    --out_dir ./data \
    --per_tier 80 \
    --seed 42

Env:
  ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default: claude-3-5-sonnet-20240620)
  PERPLEXITY_API_KEY, PPLX_MODEL (default: llama-3.1-70b-instruct)
"""
from __future__ import annotations
import os, json, random, hashlib, re
from typing import Dict, Any, List
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
from tqdm import tqdm

load_dotenv()

# ---------------- Schema ----------------
class Record(BaseModel):
    instruction: str = Field(..., min_length=5)
    input: str = Field(..., min_length=10, max_length=1500)
    cot: str = Field(..., min_length=5, max_length=2000)
    output: str = Field(..., min_length=3, max_length=1000)
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    label: str = Field(..., pattern="^(match|no_match|edge)$")

# ---------------- Providers ----------------
class ProviderBase:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _get(self, prompt: str) -> str | None:
        fp = os.path.join(self.cache_dir, self._key(prompt) + ".txt")
        return open(fp, "r").read() if os.path.exists(fp) else None

    def _put(self, prompt: str, text: str):
        fp = os.path.join(self.cache_dir, self._key(prompt) + ".txt")
        with open(fp, "w") as f:
            f.write(text)

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

class AnthropicProvider(ProviderBase):
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError("Install anthropic to use Anthropic provider") from e
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20), reraise=True)
    def complete(self, prompt: str) -> str:
        cached = self._get(prompt)
        if cached:
            return cached
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"])
        self._put(prompt, text)
        return text

class PerplexityProvider(ProviderBase):
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set.")
        self.model = os.environ.get("PPLX_MODEL", "llama-3.1-70b-instruct")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.client = httpx.Client(timeout=90)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20), reraise=True)
    def complete(self, prompt: str) -> str:
        cached = self._get(prompt)
        if cached:
            return cached
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = self.client.post(self.base_url, headers=headers, json=payload)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        self._put(prompt, text)
        return text

# ---------------- Noise & Utils ----------------
def light_noise(s: str) -> str:
    """Injects minor OCR-ish noise in a safe way (rarely)."""
    if random.random() > 0.15:
        return s
    s = s.replace("0", "O") if random.random() < 0.5 else s
    s = s.replace("l", "1") if random.random() < 0.5 else s
    s = re.sub(r"(?i)\b(sa|ltd|llc)\b", lambda m: m.group(0).upper(), s)
    return s

def iter_json_lines(text: str) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            out.append(obj)
        except Exception:
            # Attempt to extract one JSON object per block
            m = re.search(r"\{.*\}", ln)
            if m:
                try:
                    out.append(json.loads(m.group(0)))
                except Exception:
                    continue
    return out

# ---------------- Meta Prompt (VERBATIM) ----------------
META_PROMPT = r"""
You are creating supervised fine-tuning data for an AML/KYC screening model that must decide whether a candidate entity MATCHES a watchlist/PEP/sanctions entry and explain the reasoning.

Output STRICTLY one JSON object per line with fields:
- instruction: concise task description for the model.
- input: realistic case context (names, DOB bands, nationality, IDs, addresses, company relationships, prior alerts). Include enough detail to decide.
- cot: step-by-step reasoning leading to a conclusion (chain of thought). The reasoning must cite specific fields (DOB, country, alias evidence, ownership links) and weigh conflicting evidence.
- output: a one-line final decision and rationale summary (e.g., "MATCH — alias + DOB + LEI link confirm identity").
- label: one of {match, no_match, edge}. "edge" = borderline cases requiring careful judgment.
- difficulty: one of {easy, medium, hard}. Easy = clear matches/mismatches; Medium = minor noise; Hard = transliteration, mixed scripts, partial IDs, corporate hierarchies, near-collisions.

Constraints:
- Vary locales and scripts (Latin, Cyrillic, Arabic, Chinese transliterations).
- Include both person and organization entities.
- Ensure single unambiguous gold label per sample.
- Avoid real PII; use fictionalized yet plausible data inspired by public patterns (OpenSanctions, GLEIF, etc.).
- Keep cot < 250 tokens; input < 350 tokens.

Generate N samples for DIFFICULTY="{tier}".
"""

# ---------------- Main generation ----------------
def main():
    import argparse, pathlib, itertools
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "perplexity"], default="anthropic")
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--per_tier", type=int, default=80, help="Samples per tier (50-100 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    provider = AnthropicProvider("./.cache/generate") if args.provider == "anthropic" else PerplexityProvider("./.cache/generate")

    all_records: List[Record] = []
    seen = set()

    for tier in ["easy", "medium", "hard"]:
        print(f"\n=== Generating {tier} tier ===")
        prompt = META_PROMPT.replace('{tier}', tier)
        # Encourage diversity by multi-seeding and locale hints
        header = (
            f"Generate {args.per_tier} diverse samples.\n"
            "Vary locales/scripts (Latin, Cyrillic, Arabic, Chinese transliterations), "
            "cover persons and organizations, include corporate hierarchies and UBOs in some.\n"
            "Return one JSON object per line. No prose.\n\n"
        )
        print(f"Requesting {args.per_tier} samples from {args.provider}...")
        text = provider.complete(header + prompt)
        objs = iter_json_lines(text)
        
        # Validate + de-dup + noise
        tier_records = []
        validation_errors = 0
        duplicates = 0
        
        for obj in objs:
            obj["difficulty"] = tier  # enforce tier label
            try:
                r = Record(**obj)
            except ValidationError:
                validation_errors += 1
                continue
            key = (r.instruction.strip().lower(), r.input.strip().lower())
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            # light noise injection on input occasionally
            noisy = r.model_copy()
            noisy.input = light_noise(noisy.input)
            tier_records.append(noisy)
            if len(tier_records) >= args.per_tier:
                break

        print(f"[{tier}] Generated: {len(objs)} raw, Kept: {len(tier_records)}, "
              f"Validation errors: {validation_errors}, Duplicates: {duplicates}")
        
        out_fp = os.path.join(args.out_dir, f"{tier}.jsonl")
        with open(out_fp, "w") as f:
            for r in tier_records:
                f.write(r.model_dump_json() + "\n")
        all_records.extend(tier_records)

    # Combined train
    train_fp = os.path.join(args.out_dir, "train.jsonl")
    with open(train_fp, "w") as f:
        for r in all_records:
            f.write(r.model_dump_json() + "\n")

    # Balanced validation set across difficulty & label
    by_bucket: Dict[tuple, List[Record]] = {}
    for r in all_records:
        by_bucket.setdefault((r.difficulty, r.label), []).append(r)

    target_val = min(100, len(all_records) // 5)  # 20% for validation, max 100
    buckets = list(by_bucket.items())
    per_bucket = max(1, target_val // max(1, len(buckets)))
    val = []
    for (k, arr) in buckets:
        random.shuffle(arr)
        val.extend(arr[:per_bucket])
    val = val[:target_val]

    val_fp = os.path.join(args.out_dir, "val.jsonl")
    with open(val_fp, "w") as f:
        for r in val:
            f.write(r.model_dump_json() + "\n")

    # Print summary statistics
    print(f"\n=== Summary ===")
    print(f"Total records: {len(all_records)}")
    print(f"Training samples: {len(all_records)}")
    print(f"Validation samples: {len(val)}")
    
    # Distribution by difficulty
    difficulty_dist = {}
    label_dist = {}
    for r in all_records:
        difficulty_dist[r.difficulty] = difficulty_dist.get(r.difficulty, 0) + 1
        label_dist[r.label] = label_dist.get(r.label, 0) + 1
    
    print("Difficulty distribution:", difficulty_dist)
    print("Label distribution:", label_dist)
    
    print(f"\nFiles written:")
    print(f"  {train_fp}")
    print(f"  {val_fp}")
    for tier in ["easy", "medium", "hard"]:
        print(f"  {os.path.join(args.out_dir, f'{tier}.jsonl')}")

if __name__ == "__main__":
    main()