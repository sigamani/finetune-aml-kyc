# AML/KYC Fine-Tuning System - Usage Guide

## Overview

This repository contains a complete AML/KYC (Anti-Money Laundering/Know Your Customer) compliance system that fine-tunes large language models for automated regulatory screening. The system implements curriculum learning (easy → medium → hard) and includes synthetic data generation, training, and evaluation components.

## System Architecture

### 🎯 **Core Goal**
Fine-tune an 8B parameter LLM to achieve >99% recall with <50% false positives for offline batch processing of regulatory alerts.

### 📁 **Key Files**

1. **`generate_synthetics.py`** - Synthetic AML/KYC data generator
2. **`train_enhanced.py`** - Enhanced curriculum learning trainer with logging
3. **`train.py`** - Original Unsloth + TRL trainer 
4. **`eval_judge.py`** - LLM-as-Judge evaluation system
5. **`create_dev_data.py`** - Quick test data generator

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (for synthetic data generation and evaluation)
export ANTHROPIC_API_KEY=your_key_here
export PERPLEXITY_API_KEY=your_key_here  # Optional, if using Perplexity
```

### 2. Generate Synthetic Training Data

```bash
# Generate comprehensive synthetic dataset
python generate_synthetics.py \
  --provider anthropic \
  --out_dir ./data \
  --per_tier 100 \
  --seed 42

# OR generate quick test data  
python create_dev_data.py
```

### 3. Training Options

#### Option A: Enhanced Training (Recommended)
```bash
python train_enhanced.py \
  --model_id unsloth/tinyllama-chat \
  --data_dir ./data \
  --output_dir ./outputs/aml-model \
  --max_seq_len 1024 \
  --epochs_easy 2 --epochs_medium 2 --epochs_hard 1 \
  --lr 2e-5 --batch_size 2 --grad_accum 4 \
  --run_judge true --judge_provider anthropic
```

#### Option B: Original Training
```bash
python train.py \
  --model_id mistralai/Mistral-7B-Instruct-v0.3 \
  --data_dir ./data \
  --output_dir ./outputs/aml-model \
  --max_seq_len 2048 \
  --epochs_easy 2 --epochs_medium 2 --epochs_hard 1 \
  --lr 1e-5 --batch_size 2 --grad_accum 16 \
  --run_judge true
```

### 4. Evaluation

```bash
python eval_judge.py \
  --model_path ./outputs/aml-model/phase_hard \
  --dataset ./data/val.jsonl \
  --provider anthropic \
  --tau 0.5 \
  --out ./outputs/evaluation_results.json
```

## 📊 Data Format

All training data uses this JSONL schema:

```json
{
  "instruction": "Determine if this entity matches the watchlist entry.",
  "input": "Candidate: John Smith, DOB: 1980-05-15, Nationality: UK, ID: UK123456789...",
  "output": "MATCH — DOB alignment and nationality confirm identity despite minor name variation.",
  "cot": "Step 1: Compare names... Step 2: Verify DOB... Step 3: Check nationality...",
  "difficulty": "easy|medium|hard",
  "label": "match|no_match|edge"
}
```

## 🔧 Configuration Options

### Model Support
- **Recommended**: `unsloth/tinyllama-chat` (compatible, stable)
- **Alternative**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Default**: `meta-llama/Llama-3.3-8B-Instruct` (requires HF access)

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_seq_len` | 1024/4096 | Maximum sequence length |
| `--epochs_easy` | 2 | Epochs for easy phase |
| `--epochs_medium` | 2 | Epochs for medium phase |
| `--epochs_hard` | 1 | Epochs for hard phase |
| `--lr` | 2e-5/1e-5 | Learning rate |
| `--batch_size` | 1-2 | Per-device batch size |
| `--grad_accum` | 2-16 | Gradient accumulation steps |

### Hardware Recommendations

#### ✅ **Optimal Setup**
- **GPU**: RTX 4090, A100, H100 (avoid RTX 5090 due to Flash Attention issues)
- **VRAM**: 24GB+ for full training
- **Time**: ~8 hours for complete curriculum
- **Cost**: ~$100 on cloud GPUs

#### ⚠️ **Known Issues**
- **RTX 5090**: Flash Attention CUDA errors - use different GPU or wait for software updates
- **CPU Training**: Not supported with Unsloth (NVIDIA/Intel GPU required)

## 📋 Output Structure

After training, you'll find:

```
outputs/
├── phase_easy/
│   ├── adapter/              # LoRA weights for easy phase
│   └── judge_eval_easy.json  # Evaluation results
├── phase_medium/
│   ├── adapter/
│   └── judge_eval_medium.json
├── phase_hard/
│   ├── adapter/              # Final model weights
│   └── judge_eval_hard.json
├── training_summary.json     # Complete training metrics
├── integration_snippet.py    # Loading example
└── training.log             # Detailed training logs
```

## 🎯 Production Usage

### Loading Trained Model

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# Load the trained model
model = AutoPeftModelForCausalLM.from_pretrained("./outputs/aml-model/phase_hard")
tokenizer = AutoTokenizer.from_pretrained("unsloth/tinyllama-chat")

# Run inference
def analyze_case(instruction: str, case_input: str) -> str:
    prompt = f"### Human:\\n{instruction}\\n{case_input}\\n\\n### Assistant:\\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 🔍 Evaluation Metrics

The system tracks:
- **Accuracy**: Overall decision correctness
- **Precision/Recall**: Per-class performance (match/no_match/edge)
- **F1 Score**: Balanced performance measure
- **Reasoning Quality**: LLM-as-Judge assessment of explanations
- **Hallucination Detection**: Factual accuracy checks

## 🐛 Troubleshooting

### Common Issues

1. **Flash Attention CUDA Error**
   - **Solution**: Use RTX 4090/A100 instead of RTX 5090
   - **Alternative**: Wait for software updates

2. **Model Access Denied**
   - **Solution**: Use `unsloth/tinyllama-chat` or authenticate with HuggingFace

3. **Out of Memory**
   - **Solution**: Reduce `--max_seq_len`, `--batch_size`, or use `--grad_accum`

4. **API Key Issues**
   - **Solution**: Set environment variables: `ANTHROPIC_API_KEY`, `PERPLEXITY_API_KEY`

### Performance Tips

- Start with `unsloth/tinyllama-chat` for testing
- Use gradient accumulation to simulate larger batches
- Monitor `training.log` for detailed progress
- Enable judge evaluation for comprehensive metrics

## 📈 Expected Results

With proper setup, expect:
- **Training Time**: 2-8 hours depending on hardware
- **Accuracy**: 85-95% on validation set
- **Recall**: >90% for match detection
- **False Positives**: <30% with proper tuning

## 🤝 Next Steps

1. **Scale Up**: Generate more synthetic data (`--per_tier 500+`)
2. **Fine-tune**: Adjust learning rates and epochs based on validation curves
3. **Deploy**: Use the integration snippet for production inference
4. **Monitor**: Track performance on real-world data

## 📚 Additional Resources

- **Unsloth Documentation**: https://github.com/unslothai/unsloth
- **Model Hub**: https://huggingface.co/models
- **OpenSanctions**: Public sanctions data for reference patterns