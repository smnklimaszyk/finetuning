# 🔄 Prediction Caching System - Design Document

## Current Behavior (Issue)

### What's Happening:
When you run `python main.py` multiple times, the system:
1. ✅ **Saves models** correctly to disk
2. ❌ **Regenerates predictions** every time (even if models haven't changed)
3. ⏱️ **Wastes time** re-running inference on the same test set

### Why This Happens:
Looking at the code flow:

```python
# main.py line 317+
results[result_key] = evaluate_baseline_llm(
    config, test_dataset, processor, llm_config=llm_config
)

# This calls evaluate_model() which ALWAYS generates predictions:
predictions_raw = model.predict_batch(...)  # No caching check!
```

**The Problem**: There's no mechanism to:
- Check if predictions already exist for this model
- Load cached predictions if available
- Only regenerate if model/data changed

---

## Why Caching Matters

### Time Savings:
- **Baseline LLM Evaluation**: ~10-30 minutes per model
- **Test Set Size**: 1000+ samples
- **3 Baseline Models**: ~30-90 minutes wasted on reruns
- **With Caching**: ~5 seconds to load results

### Use Cases:
1. **Iterative Development**: Tweaking visualization code
2. **Adding New Models**: Don't re-evaluate existing ones
3. **Experiments**: Compare new model against cached baselines
4. **Debugging**: Fast iteration without waiting for inference

---

## Solution: Smart Caching System

### Cache Key Design:
```python
cache_key = hash(
    model_name,
    model_version,
    test_dataset_hash,
    config_hash  # generation params: temperature, max_tokens, etc.
)
```

### Cache Structure:
```
outputs/
  cache/
    predictions/
      Qwen2.5-3B-Instruct_<hash>.json
      Llama-3.2-3B-Instruct_<hash>.json
      Meta-Llama-3.1-8B-Instruct_finetuned_<hash>.json
```

### Cache File Format:
```json
{
  "metadata": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "model_hash": "abc123...",
    "test_dataset_size": 1000,
    "test_dataset_hash": "def456...",
    "generation_config": {
      "temperature": 0.1,
      "max_new_tokens": 50,
      "do_sample": false
    },
    "timestamp": "2026-01-31T16:34:34",
    "evaluation_time_seconds": 1234.56
  },
  "predictions": [
    {"input": "Patient has...", "prediction": "J06.9", "reference": "J06.9"},
    ...
  ],
  "metrics": {
    "exact_match_accuracy": 0.85,
    "f1": 0.87,
    ...
  }
}
```

---

## Implementation Strategy

### Phase 1: Basic Caching (Quick Win)
Add cache check to `evaluate_model()`:

```python
def evaluate_model(model, test_dataset, processor, batch_size=8):
    # 1. Generate cache key
    cache_key = generate_cache_key(model, test_dataset, config)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    # 2. Check if cache exists and is valid
    if cache_file.exists() and not config.evaluation.force_recompute:
        logger.info(f"✅ Loading cached predictions from {cache_file}")
        cached_data = load_json(cache_file)
        return cached_data["metrics"]
    
    # 3. Generate predictions (existing code)
    logger.info(f"🔄 Generating predictions (no cache found)...")
    predictions_raw = model.predict_batch(...)
    
    # 4. Save to cache
    cache_data = {
        "metadata": {...},
        "predictions": [...],
        "metrics": metrics
    }
    save_json(cache_data, cache_file)
    logger.info(f"💾 Predictions cached to {cache_file}")
    
    return metrics
```

### Phase 2: Cache Invalidation
Detect when to invalidate cache:

1. **Model Changed**: Compare model checkpoint timestamp
2. **Data Changed**: Hash test dataset
3. **Config Changed**: Hash generation parameters
4. **Manual Override**: `--force-recompute` flag

### Phase 3: Cache Management
```bash
# CLI commands
python main.py --clear-cache  # Delete all cached predictions
python main.py --list-cache   # Show cached models
python main.py --force-recompute  # Ignore cache
```

---

## Configuration Addition

Add to `EvaluationConfig`:

```python
class EvaluationConfig(BaseModel):
    # ... existing fields ...
    
    # Prediction Caching
    use_prediction_cache: bool = True
    cache_predictions_dir: Path = Field(
        default_factory=lambda: Path("outputs/cache/predictions")
    )
    force_recompute: bool = False
    cache_max_age_days: int = 30  # Auto-delete old caches
```

---

## Benefits Summary

| Metric | Without Caching | With Caching | Improvement |
|--------|----------------|--------------|-------------|
| **First Run** | 30-90 min | 30-90 min | No change |
| **Subsequent Runs** | 30-90 min | **~5 sec** | 🚀 360-1080x faster |
| **Add 1 New Model** | 30-90 min (all) | **10-30 min** (new only) | 🎯 3x faster |
| **Tweak Visualization** | 30-90 min | **~5 sec** | ⚡ Instant |
| **Disk Space** | N/A | ~50-200 MB | 💾 Minimal |

---

## Implementation Priority

### High Priority (Implement Now):
✅ Basic prediction caching in `evaluate_model()`
✅ Cache key generation based on model name + dataset
✅ `--force-recompute` CLI flag
✅ Cache directory in config

### Medium Priority (Next Sprint):
⏸️ Advanced cache invalidation (hash-based)
⏸️ Cache management CLI commands
⏸️ Cache expiration (auto-cleanup)

### Low Priority (Future):
⏸️ Distributed cache (Redis/S3)
⏸️ Incremental caching (batch-level)
⏸️ Cache compression

---

## Current Issue Summary

**Q: Is it supposed to regenerate predictions on every run?**

**A: NO - It's a missing feature!**

**Explanation**:
- ✅ Models are correctly saved to disk
- ✅ Model loading works (checkpoints are reused)
- ❌ **Predictions** are NOT cached
- ❌ Every run = full inference on test set
- ❌ Wastes 30-90 minutes on repeated evaluations

**Why It Happens**:
The code was designed to:
1. **Train** models → Save checkpoints ✅
2. **Load** models → From checkpoints ✅
3. **Evaluate** models → Generate predictions ✅

But it's missing:
4. **Cache** predictions → **NOT IMPLEMENTED** ❌
5. **Reuse** predictions → **NOT IMPLEMENTED** ❌

**Solution**: Implement the caching system described above.

---

## Next Steps

1. ✅ **Understand the issue** ← YOU ARE HERE
2. ⚠️ **Implement basic caching** in `evaluate_model()`
3. ⚠️ **Add `--force-recompute` flag** to CLI
4. ⚠️ **Test with your workflow**
5. ⚠️ **Extend to advanced features** (optional)

Would you like me to implement the basic caching system now?
