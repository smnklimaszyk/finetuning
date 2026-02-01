# Quick Start Guide

## TL;DR

```bash
# 1. Setup (one time)
export PYTORCH_ALLOC_CONF=expandable_segments:True
pip install -e ".[dev]"

# 2. Run experiment
python main.py --experiment full

# 3. View results
open outputs/reports/evaluation_report.html
```

---

## What Changed?

### New Research Question
**Can small finetuned models (3B) beat large untrained models (7-8B)?**

### New Structure

**LLMs (Large, Untrained):**
- Llama 8B (zero-shot)
- Mistral 7B (zero-shot)

**SLMs (Small, Finetuned):**
- Llama 3B (LoRA finetuned)
- Qwen 3B (LoRA finetuned)

### Memory Fix
Set this environment variable to avoid CUDA OOM:
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

---

## Commands

### Full Pipeline
```bash
# Evaluates LLMs + Trains & Evaluates SLMs
python main.py --experiment full
```

### Quick Test
```bash
# Test with 100 samples only
python main.py --experiment baseline --max-eval-samples 100
```

### Only LLM Evaluation
```bash
# Skip training, just test large models
python main.py --experiment baseline
```

### Only SLM Training
```bash
# Skip LLM eval, just train small models
python main.py --experiment training
```

### Custom Models
```bash
# Use specific models
python main.py \
  --model-llm "mistralai/Mistral-7B-Instruct-v0.3" \
  --model-slm "Qwen/Qwen2.5-3B-Instruct"
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# 1. Set environment variable (REQUIRED!)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2. If still OOM, reduce batch size in config
# Edit src/config/base_config.py:
# training.per_device_train_batch_size = 16  # default: 32
```

### Slow Performance
```bash
# Check GPU usage
nvidia-smi

# Check config optimization
# Make sure in src/config/base_config.py:
# - bf16 = True
# - torch_compile = True (if PyTorch 2.0+)
# - attn_implementation = "flash_attention_2"
```

### Model Download Issues
```bash
# For meta-llama models, accept terms first:
# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

# Then login
huggingface-cli login
```

---

## Expected Results

### Execution Time
- LLM Evaluation: ~10-15 min
- SLM Training (each): ~45-60 min
- SLM Evaluation (each): ~5 min
- **Total**: ~2-3 hours

### Memory Usage (RTX 5090)
- Peak: ~10GB VRAM
- Available: 31GB
- Safety Margin: Plenty!

### Output Files
```
outputs/
├── reports/
│   ├── results.json          # Raw results
│   └── evaluation_report.html  # Visual report
├── plots/
│   └── model_comparison.png
└── logs/
    └── app.log
```

---

## Configuration

### Default Models (in `src/config/base_config.py`)

```python
# LLMs (large, untrained)
llm_models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 8B
    "mistralai/Mistral-7B-Instruct-v0.3",     # 7B
]

# SLMs (small, finetuned)
slm_models = [
    "meta-llama/Llama-3.2-3B-Instruct",  # 3B
    "Qwen/Qwen2.5-3B-Instruct",          # 3B
]
```

### Adjust for Smaller GPU

For GPUs with <16GB VRAM:
```python
# Use only one LLM and one SLM
llm_models = ["mistralai/Mistral-7B-Instruct-v0.3"]
slm_models = ["Qwen/Qwen2.5-3B-Instruct"]

# Reduce batch sizes
training.per_device_train_batch_size = 8
evaluation.eval_batch_size = 16
```

---

## Results Interpretation

### Example Output
```json
{
  "LLM_Meta-Llama-3_1-8B-Instruct_untrained": {
    "exact_match_accuracy": 0.123,
    "f1": 0.234,
    "model_type": "LLM",
    "model_size": "8B"
  },
  "SLM_Llama-3_2-3B-Instruct_finetuned": {
    "exact_match_accuracy": 0.456,
    "f1": 0.567,
    "model_type": "SLM",
    "model_size": "3B"
  }
}
```

### What to Look For

**If SLM > LLM:**
- Specialization beats size
- Use small finetuned models in production
- Better cost/performance ratio

**If LLM > SLM:**
- Size/knowledge beats specialization
- May need larger models even with finetuning
- Consider 7B finetuned models

**If Mixed:**
- Analyze per-category performance
- Different models for different cases
- Hybrid system possible

---

## More Information

- **Full Documentation**: `GUIDE.md`
- **Architecture Details**: `ARCHITECTURE_REDESIGN.md`
- **Implementation Details**: `REFACTORING_SUMMARY.md`

---

## Testing Custom Interviews

After training, test your models on custom cases:

```bash
# Test on your own doctor-patient conversations
python test_custom_interviews.py

# Create custom test cases in JSON format
# See custom_interviews/ for examples
```

See [CUSTOM_INTERVIEW_TESTING.md](CUSTOM_INTERVIEW_TESTING.md) for details.

---

**Ready to start? Run:**
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
python main.py --experiment full
```
