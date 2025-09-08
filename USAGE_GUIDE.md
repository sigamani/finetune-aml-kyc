# AML/KYC Fine-Tuning System - Usage Guide

## Quick Start

### 1. Setup Environment
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# API key is already configured in .env file
```

### 2. Generate Test Data & Quick Training
```bash
# Create small test dataset (20 samples)
uv run create_dev_data.py

# Quick training run with test data
uv run train_enhanced.py \
  --model_id unsloth/tinyllama-chat \
  --data_dir ./data \
  --output_dir ./outputs/test-model \
  --max_seq_len 1024 \
  --epochs_easy 1 --epochs_medium 1 --epochs_hard 1 \
  --lr 2e-5 --batch_size 1 --grad_accum 2
```

### 3. Generate Synthetic Data & Full Training
```bash
# Generate comprehensive synthetic dataset with OpenAI
uv run generate_synthetics.py \
  --provider openai \
  --out_dir ./data \
  --per_tier 100 \
  --seed 42

# Full training with synthetic data
uv run train_enhanced.py \
  --model_id unsloth/tinyllama-chat \
  --data_dir ./data \
  --output_dir ./outputs/aml-model \
  --max_seq_len 1024 \
  --epochs_easy 2 --epochs_medium 2 --epochs_hard 1 \
  --lr 2e-5 --batch_size 2 --grad_accum 4 \
  --run_judge true --judge_provider openai
```

### 4. Evaluation
```bash
# Evaluate trained model
uv run eval_judge.py \
  --model_path ./outputs/aml-model/phase_hard \
  --dataset ./data/val.jsonl \
  --provider openai \
  --tau 0.5 \
  --out ./outputs/evaluation_results.json
```

## Data Format

Training data uses JSONL with these fields:

```json
{
  "instruction": "Determine if this entity matches the watchlist entry.",
  "input": "Candidate: John Smith, DOB: 1980-05-15, Nationality: UK...",
  "output": "MATCH — DOB alignment and nationality confirm identity.",
  "cot": "Step 1: Compare names... Step 2: Verify DOB... Step 3: Check nationality...",
  "difficulty": "easy",
  "label": "match"
}
```

## Configuration

### Model Support
- **Recommended**: `unsloth/tinyllama-chat` (tested, stable)
- **Alternative**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Default**: `meta-llama/Llama-3.3-8B-Instruct`

### Training Parameters

| Parameter | Test Run | Full Run | Description |
|-----------|----------|----------|-------------|
| `--max_seq_len` | 1024 | 2048 | Maximum sequence length |
| `--epochs_easy` | 1 | 2 | Easy phase epochs |
| `--epochs_medium` | 1 | 2 | Medium phase epochs |
| `--epochs_hard` | 1 | 1 | Hard phase epochs |
| `--lr` | 2e-5 | 1e-5 | Learning rate |
| `--batch_size` | 1 | 2 | Per-device batch size |
| `--grad_accum` | 2 | 4-16 | Gradient accumulation steps |

### Hardware Requirements

**Minimum**: RTX 4090, A100 (24GB+ VRAM)
**Avoid**: RTX 5090 (Flash Attention CUDA issues)

## Output Structure

```
outputs/
├── phase_easy/
│   ├── adapter/              # LoRA weights
│   └── judge_eval_easy.json  # Evaluation results
├── phase_medium/
│   ├── adapter/
│   └── judge_eval_medium.json
├── phase_hard/
│   ├── adapter/              # Final trained model
│   └── judge_eval_hard.json
├── training_summary.json     # Complete training metrics
└── training.log             # Detailed logs
```

## Loading Trained Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("./outputs/aml-model/phase_hard")
tokenizer = AutoTokenizer.from_pretrained("unsloth/tinyllama-chat")

def analyze_case(instruction: str, case_input: str) -> str:
    prompt = f"### Human:\\n{instruction}\\n{case_input}\\n\\n### Assistant:\\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Troubleshooting

**Flash Attention CUDA Error**: Use RTX 4090/A100 instead of RTX 5090

**Out of Memory**: Reduce `--max_seq_len`, `--batch_size`, or increase `--grad_accum`

**Model Access Denied**: Use `unsloth/tinyllama-chat` or authenticate with HuggingFace

## Expected Results

- **Training Time**: 2-8 hours depending on hardware
- **Accuracy**: 85-95% on validation set  
- **Recall**: >90% for match detection
- **False Positives**: <30% with proper tuning