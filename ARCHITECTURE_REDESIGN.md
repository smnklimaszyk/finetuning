# Architecture Redesign Plan

## New Experimental Design

### Research Hypothesis
**Can smaller finetuned models (3B) outperform larger untrained models (7-8B) on specialized tasks?**

This tests the fundamental trade-off between:
- **Size** (more parameters, more knowledge)
- **Specialization** (task-specific training)

---

## Model Structure

### LLMs (Large Language Models - Untrained Reference)
**Definition:** Large models (7-8B parameters) used zero-shot for comparison

| Model | Size | Purpose | Training |
|-------|------|---------|----------|
| Meta-Llama-3.1-8B-Instruct | 8B | Large reference | Zero-shot |
| Mistral-7B-Instruct-v0.3 | 7B | Large reference | Zero-shot |

**Evaluation:** Prompt engineering only, no finetuning

### SLMs (Small Language Models - Finetuned Specialized)
**Definition:** Small models (3B parameters) finetuned on medical ICD-10 data

| Model | Size | Purpose | Training |
|-------|------|---------|----------|
| Llama-3.2-3B-Instruct | 3B | Small specialized | LoRA finetuned |
| Qwen2.5-3B-Instruct | 3B | Small specialized | LoRA finetuned |

**Evaluation:** After domain-specific finetuning

---

## Configuration Changes

### Old Config Structure
```python
class ModelConfig:
    baseline_llms: List[LLMModelConfig]  # Mixed sizes, confusing
    slm_name: str                         # Single model only
    slm_load_in_4bit: bool
```

### New Config Structure
```python
class ModelConfig:
    # Large untrained models (7-8B) - for zero-shot comparison
    llm_models: List[LLMConfig] = [
        LLMConfig(
            name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            size="8B",
            description="Large reference model",
            load_in_4bit=True,
        ),
        LLMConfig(
            name="mistralai/Mistral-7B-Instruct-v0.3",
            size="7B",
            description="Medium reference model",
            load_in_4bit=True,
        ),
    ]

    # Small models (3B) - will be finetuned
    slm_models: List[SLMConfig] = [
        SLMConfig(
            name="meta-llama/Llama-3.2-3B-Instruct",
            size="3B",
            description="Compact Llama for finetuning",
            load_in_4bit=False,  # Full precision for finetuning
        ),
        SLMConfig(
            name="Qwen/Qwen2.5-3B-Instruct",
            size="3B",
            description="Compact Qwen for finetuning",
            load_in_4bit=False,
        ),
    ]
```

---

## Pipeline Flow

### Old Pipeline
```
1. Load data
2. Evaluate baseline LLMs (Qwen 3B, Llama 3B, Mistral 7B)
3. Evaluate baseline SLM (Llama 8B untrained)
4. Finetune SLM (Llama 8B)
5. Evaluate finetuned SLM (Llama 8B)
6. Compare results
```

### New Pipeline
```
1. Load data
2. Evaluate LLMs (untrained) - Llama 8B, Mistral 7B
3. FOR EACH SLM model:
     a. Finetune (Llama 3B or Qwen 3B)
     b. Evaluate finetuned model
4. Compare: LLMs (size advantage) vs SLMs (specialization advantage)
```

---

## Memory Management Strategy

### Challenge
- Llama 8B (4-bit): ~5GB VRAM
- Mistral 7B (4-bit): ~4GB VRAM
- Llama 3B (BF16 for training): ~6GB VRAM
- Qwen 3B (BF16 for training): ~6GB VRAM
- Total RTX 5090 VRAM: 31GB ✅ Plenty!

### Problem
Memory fragmentation when loading/unloading models sequentially

### Solution
**Evaluation Order (Smart Scheduling):**
```
1. Llama 8B (untrained)    → unload → cleanup
2. Mistral 7B (untrained)  → unload → cleanup
3. Llama 3B (finetune)     → save checkpoint → unload → cleanup
4. Llama 3B (evaluate)     → unload → cleanup
5. Qwen 3B (finetune)      → save checkpoint → unload → cleanup
6. Qwen 3B (evaluate)      → unload → cleanup
```

**Enhanced Cleanup Between Models:**
```python
def aggressive_cleanup():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()  # Extra cleanup
```

---

## File Structure Changes

### Configuration Files
```
src/config/
├── __init__.py
├── base_config.py          # MODIFIED: New model structure
└── model_configs.py        # NEW: Model-specific configs
```

### Model Classes
```
src/models/
├── __init__.py
├── base_model.py           # MODIFIED: Enhanced cleanup
├── llm_model.py           # MODIFIED: Untrained LLM wrapper
└── slm_model.py           # MODIFIED: Finetuned SLM wrapper
```

### Training
```
src/training/
├── __init__.py
├── trainer.py             # MODIFIED: Support multiple models
└── multi_model_trainer.py # NEW: Manages training multiple SLMs
```

### Main Pipeline
```
main.py                    # MAJOR REFACTOR: New evaluation loop
```

---

## Results Structure

### Old Results
```json
{
  "Baseline_LLM_Qwen2.5-3B-Instruct": {...},
  "Baseline_LLM_Llama-3.2-3B-Instruct": {...},
  "Baseline_LLM_Mistral-7B-Instruct-v0.3": {...},
  "Baseline_SLM": {...},
  "Finetuned_SLM": {...}
}
```

### New Results
```json
{
  "LLM_Llama-8B_untrained": {
    "model_type": "LLM",
    "size": "8B",
    "training": "zero-shot",
    "metrics": {...}
  },
  "LLM_Mistral-7B_untrained": {
    "model_type": "LLM",
    "size": "7B",
    "training": "zero-shot",
    "metrics": {...}
  },
  "SLM_Llama-3B_finetuned": {
    "model_type": "SLM",
    "size": "3B",
    "training": "lora_finetuned",
    "metrics": {...}
  },
  "SLM_Qwen-3B_finetuned": {
    "model_type": "SLM",
    "size": "3B",
    "training": "lora_finetuned",
    "metrics": {...}
  }
}
```

---

## Visualization Updates

### New Comparison Plots

1. **Size vs. Specialization:**
   ```
   Accuracy
     │
   1.0├─────────────────────
     │        ● SLM Llama 3B (finetuned)
   0.8├────────● SLM Qwen 3B (finetuned)
     │    ● LLM Llama 8B (untrained)
   0.6├────● LLM Mistral 7B (untrained)
     │
   0.0└─────────────────────
        Small        Large
       (Finetuned) (Untrained)
   ```

2. **Performance vs. Cost:**
   - Y-axis: F1 Score
   - X-axis: Inference latency (ms)
   - Shows: Smaller models = faster + potentially better

3. **Resource Efficiency:**
   - Compare VRAM usage
   - Compare tokens/second
   - Show: Deployment advantages of SLMs

---

## Implementation Phases

### Phase 1: Configuration Refactor ✅
- Update `base_config.py` with new structure
- Create model config classes
- Ensure backward compatibility for data loading

### Phase 2: Memory Management ✅
- Enhanced cleanup in `base_model.py`
- Add memory monitoring utilities
- Test sequential loading/unloading

### Phase 3: Training Pipeline ✅
- Support multiple finetuning targets
- Separate checkpoints for each model
- Proper naming conventions

### Phase 4: Evaluation Logic ✅
- New evaluation loop in `main.py`
- Updated result structure
- Memory cleanup between models

### Phase 5: Documentation ✅
- Updated GUIDE.md
- New architecture diagram
- Usage examples

---

## Compatibility Considerations

### What MUST be preserved:
✅ Dataset loading (MedSynth)
✅ Tokenization pipeline
✅ LoRA training configuration
✅ Metrics computation
✅ Caching system
✅ Model download from HuggingFace

### What can change:
- Config structure (with migration path)
- Model naming conventions
- Results format
- Visualization

---

## Testing Strategy

### Unit Tests
```python
def test_llm_evaluation():
    """Test untrained LLM evaluation"""

def test_slm_finetuning():
    """Test SLM finetuning pipeline"""

def test_memory_cleanup():
    """Test aggressive memory cleanup"""

def test_multi_model_training():
    """Test sequential training of multiple SLMs"""
```

### Integration Tests
```bash
# Quick test with small dataset
python main.py --experiment full --max-eval-samples 100

# Full test
python main.py --experiment full
```

---

## Expected Results

### Hypothesis Testing

**If SLMs win:** Finetuning > Size for specialized tasks
- Deploy smaller, faster models in production
- Lower costs, better privacy

**If LLMs win:** General knowledge > Specialization
- Need larger models for medical tasks
- Finetuning may not help much

**Most likely:** Nuanced trade-offs
- SLMs better on common diagnoses
- LLMs better on rare/complex cases
- Different models for different use cases

---

## Migration Path

### For existing runs:
```python
# Add migration utility
def migrate_old_config(old_config):
    """Converts old config to new structure"""
    new_config = Config()

    # Map old baseline_llms to new llm_models/slm_models
    for model in old_config.model.baseline_llms:
        if "8B" in model.name or "7B" in model.name:
            new_config.model.llm_models.append(model)
        else:
            new_config.model.slm_models.append(model)

    return new_config
```

### For existing caches:
- Prediction cache remains compatible (keyed by model name + config)
- Old results can be loaded and converted

---

## Summary

This redesign:
1. ✅ Clarifies the experimental design
2. ✅ Makes the research question explicit
3. ✅ Improves maintainability
4. ✅ Preserves all working functionality
5. ✅ Solves the memory issue
6. ✅ Enables meaningful comparisons

Next: Implement the changes!
