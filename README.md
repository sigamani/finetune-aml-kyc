# AML/KYC Fine-Tuning System

Fine-tunes 8B parameter LLMs for automated AML/KYC compliance checks. Targets >99% recall with <50% false positives for offline batch processing of regulatory alerts.

## Quick Start

### Setup
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Environment variables are in .env file
```

### Generate Test Data & Train
```bash
# Create small test dataset
uv run create_dev_data.py

# Train with test data
uv run train_enhanced.py --data_dir ./data --output_dir ./outputs/test-model

# Generate synthetic data with OpenAI
uv run generate_synthetics.py --provider openai --out_dir ./data --per_tier 100

# Train with synthetic data  
uv run train_enhanced.py --data_dir ./data --output_dir ./outputs/aml-model
```

## Architecture

**Goal**: Achieve >99% recall, <50% false positives for batch processing regulatory alerts on 7 A100s (~$100, 8 hours)

**Components**:
- `create_dev_data.py` - Quick test data generator
- `generate_synthetics.py` - OpenAI-powered synthetic data generator  
- `train_enhanced.py` - Curriculum learning trainer (easy→medium→hard)
- `eval_judge.py` - LLM-as-Judge evaluation system
- `train.py` - Original trainer

**Training Process**:
1. Curriculum learning: 3 difficulty phases with early stopping
2. LoRA fine-tuning on base models (Llama-3.3-8B, Mistral-7B, TinyLlama)
3. LLM-as-Judge evaluation after each phase

## Data Format

JSONL with fields:
- `instruction`: Task description
- `input`: Entity matching case context  
- `output`: Final decision and rationale
- `cot`: Step-by-step reasoning
- `difficulty`: easy|medium|hard
- `label`: match|no_match|edge

## Deployment

Designed for offline batch processing:
- 20M tokens (100k jobs × 200 tokens each)
- Ray cluster with vLLM workers
- Prefix caching for 30k shared context
- S3-compatible storage for I/O

Cost: $50-$100 for complete batch run on cloud GPUs