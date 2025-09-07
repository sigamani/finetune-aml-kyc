# GOAL: Fine-tuning an 8B parameter LLM used an Agentic AI framework for automated AML/KYC compliance checks
The idea would be to use this model to do offline nightly (or prefreably weekly) jobs on alerts from Banks/Regulators to maintain sensitivity (recall is maximised threshold > 99%) and we optimise for better specificity (i.e. lower false positive rates as much as possible goal < 50%). 
The proof of concept would be a key indicator that a cost effective low infra architecture could be built from this. We want to avoid real time model hosting, we want to avoid models > 8B, we want to avoid quantisations above FP8, we want to keep total costs to less than 100 dollars, we want to keep total compute time to 8 hours, we want to keep batch completion rate to > 95%.

The setup could be run feasibly with 7 A100's (roughly 1 dollar an hour for 8 hours). Some quickmafs:


 
## 1.	train_unsloth_curriculum.py — end-to-end trainer that:
  •	Loads one of:
	•	meta-llama/Llama-3.3-8B-Instruct (default), or
	•	mistralai/Mistral-7B-Instruct-v0.3, or a user-provided HF model id (reasoning-tuned allowed).
	•	Uses Unsloth + PEFT/LoRA on top of Transformers/TRL.
	•	Implements curriculum learning with 3 phases: easy, medium, hard, each its own SFT cycle continuing from prior weights.
	•	Consumes a JSONL dataset with fields:

```{"instruction": "...", "input": "...", "output": "...", "cot": "...", "difficulty": "easy|medium|hard", "label": "match|no_match|edge"}```

  •	Trains with CoT (the target text is cot + final answer), and also emits a non-CoT distillation variant (instructions in code) if user toggles.
	•	Tracks loss per step/epoch; early-stops when validation loss plateaus (<5% improvement across 3 consecutive epochs).
	•	After each curriculum phase (easy, then medium, then hard), runs LLM-as-Judge evaluation on 100 sampled items (stratified) using either Anthropic Sonnet 3.5 or Perplexity via simple wrappers (JUDGE_PROVIDER=anthropic|perplexity), computes accuracy, precision, recall at τ, plus a normalized rubric score.
	•	Saves LoRA adapters and (optionally) merged weights, tokenizer, and HF Hub push (gated by env flags).
	•	Exposes sane defaults; all hyperparams configurable via CLI (argparse) and .env.

## 2.	generate_synthetics.py — utility that programmatically generates 50–100 synthetic samples per tier (easy/medium/hard) by prompting a source LLM (configurable; default Sonnet 3.5). Includes:

	•	A domain prompt (below) and a schema validator to ensure JSONL fields and difficulty balance.
	•	Deduplication, light noise injection, and validation that each record has a single unambiguous gold label.

	3.	eval_judge.py — minimal LLM-as-Judge wrapper with a clear rubric and provider adapters (Anthropic/Perplexity). Includes caching and retry/backoff.
	4.	requirements.txt and README.md with quickstart.

# Synthetic data design

## Goal: Supervised fine-tuning for entity matching & risk reasoning in AML/KYC screening with long-context hints. You must explain what to simulate and cite realistic, augmentable open sources.
	•	What to simulate:
	•	Sanctions / PEP screening cases: person/org name variants, transliterations (Arabic/Chinese/Cyrillic→Latin), aliases, maiden names, diacritics, ordering changes, partial DOBs, nationality/country noise, ID number formats, company relationships (subsidiary/parent/UBO), and near-collisions (same name, different DOB/country).
	•	KYC document reasoning: mismatched document fields, expired IDs, jurisdiction-specific formats, red flags (PO boxes, high-risk occupations, shell company indicators).
	•	Geographic and product risk context: high-risk jurisdictions, correspondent banking, MSBs, crypto exposure, high-value transfers, structuring/smurfing patterns.
	•	Edge cases: homographs, OCR noise, mixed scripts, dual citizens, mergers/acquisitions changing legal names, sanctions evasion via intermediary entities.
	•	Open datasets to mine/augment (reference, not to copy PII):
	•	OpenSanctions (sanctions + PEP entities & aliases), UN Consolidated List, EU Consolidated List, UK HMT, US OFAC SDN (metadata/aliases).
	•	GLEIF (LEI registry) for org canonicalization; Companies House (UK) for corporate name variations.
	•	AMLSim and PAYSim (synthetic transactions) to craft realistic transactional narratives.
	•	Elliptic crypto dataset (for crypto-risk stories; do not copy labels verbatim).
	•	Augmentation ideas: phonetic encodings (Soundex/Metaphone), transliteration tables, edit-distance noise, token swaps, DOB fuzzing within regulatory bands, organization suffix variants (Ltd/LLC/S.A./GmbH).

## LLM prompt for generating synthetic items (include verbatim in generate_synthetics.py)

Goal is to produce JSONL compliant samples. The script should iterate difficulty tiers and seed diversity (locale/script/risk type).

Prompt to local GPT-OSS-20B model: ```You are creating supervised fine-tuning data for an AML/KYC screening model that must decide whether a candidate entity MATCHES a watchlist/PEP/sanctions entry and explain the reasoning.```

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

Deliverable:
	•	Call a provider LLM (Anthropic/Perplexity/OpenAI-compatible) with temperature control and seeds.
	•	Validate JSON, enforce schema, and write to {tier}.jsonl and a combined train.jsonl.
	•	Produce a 100-item validation set balanced by label and difficulty.

Training requirements
	•	Libraries: unsloth, transformers, trl, peft, accelerate, datasets, bitsandbytes, python-dotenv, tqdm, pydantic, tenacity (for retries), plus provider SDKs for judge/generation.
	•	LoRA config (defaults): r=16, alpha=32, dropout=0.05; target modules appropriate for Llama/Mistral (e.g., q_proj,k_proj,v_proj,o_proj,gate,up,down).
	•	Sequence length: respect the base model’s max context; set max_seq_len via CLI; enable RoPE scaling if supported by the chosen checkpoint.
	•	Optimizer: AdamW or Lion; cosine schedule with warmup.
	•	Batching: dynamic packing to maximize tokens/GPU; gradient checkpointing; bf16 where available.
	•	Curriculum: train easy → eval → if gate passed, continue on medium (lower lr), then hard.
	•	Early stopping rule: compute moving best val loss; if relative improvement < 5% over 3 consecutive epochs, stop current phase.
	•	Checkpoints: save per epoch and best-val; keep last 2 to limit disk.

LLM-as-Judge evaluation

Implement eval_judge.py and call it from the trainer after each phase on 100 sampled items:
	•	Rubric (score 0–1 per item):
	1.	Decision correctness (primary): match/no_match/edge agrees with gold (1 or 0).
	2.	Reasoning sufficiency: cites at least two concrete fields (0/0.5/1).
	3.	Harmful hallucination check: no fabricated fields (deduct to 0).
	•	Metrics: accuracy, precision/recall/F1 on match, macro-F1, and rubric mean. Print and write JSON report.
	•	Providers: ANTHROPIC_API_KEY and/or PERPLEXITY_API_KEY (env). Include clean adapters (rate limit, retry, cache).

CLI and outputs

train_unsloth_curriculum.py examples:

python train_unsloth_curriculum.py \
  --model_id meta-llama/Llama-3.3-8B-Instruct \
  --data_dir ./data \
  --output_dir ./outputs/llama33-8b-aml-curriculum \
  --max_seq_len 4096 \
  --epochs_easy 2 --epochs_medium 2 --epochs_hard 1 \
  --lr 1e-5 --batch_size 2 --grad_accum 16 \
  --judge_provider anthropic --run_judge true \
  --push_to_hub false --merge_lora false

Outputs:
	•	outputs/*/adapter (LoRA), optional merged (full weights if merged), tokenizer, trainer_state.json, eval reports per phase, and a short integration snippet showing how to load with transformers.AutoModelForCausalLM and serve via vLLM.

Implementation notes (must-dos)
	•	Use Unsloth’s FastLanguageModel (or equivalent) loader to enable 8-bit/4-bit loading and LoRA; include a merge_and_save() utility.
	•	Ensure tokenizer and special tokens are set consistently for instruction tuning (system/user/assistant tags or simple instruction-style—you choose one and document).
	•	Implement schema checks for the synthetic generator; fail fast on malformed items.
	•	Add determinism knobs: seed everywhere; log full config to JSON.
	•	Provide unit-test-ish smoke checks in __main__: load a tiny synthetic batch and run one training step.

Quality bar
	•	Code must run without edits (once API keys are set).
	•	Clear comments; no dead code; no placeholders left unresolved.
	•	Sensible defaults for a single A100-80GB; document 24GB GPU compromises (smaller seq len, lower batch, KV8).

# Deployment ideas (minimal model size, isolated actors, must work within vast.ai or access similar competitive market GPU market place for ease of deployment and cost effectiveness).

1. Workload Specifications

Processing 20M tokens (100k jobs × 200 tokens each) with shared 30k prefix
Compute requirement: 37-74 A100-hours depending on throughput
Target completion time: 12 hours

2. GPU Requirements & Costs

Need 4-7 A100-80GB GPUs for 12-hour target (75-150 tok/s/GPU)
Vast.ai pricing: $1.00-$1.20/hr per A100-80GB
Total cost estimate: $50-$100 for complete batch run

3. Architecture Plan

Ray cluster with vLLM workers (one per GPU)
Prefix caching: Build 30k KV cache once per actor, reuse for all jobs
Sharding: Distribute 100k jobs across N shards matching GPU count
Storage: S3-compatible for I/O, only push job-specific suffixes

4. vLLM Configuration

max_model_len=32768, tensor_parallel_size=1 for 8B models
max_num_seqs 32-64, high max_num_batched_tokens
FP16 KV precision (try FP8/INT8 for 1.3-2× throughput)
gpu_memory_utilization≈0.9 with headroom

5. RTX 3090 Alternative Analysis

VRAM constraints: 24GB too tight for 30k context (needs KV compression or tensor parallelism)
Bandwidth limitation: ~0.45× A100 performance
Would require 9-16 single GPUs or 18-32 with TP=2
Cost-effective only if 3090 hourly rate ≤ 50% of A100 rate
