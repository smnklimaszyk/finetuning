# Custom Interview Testing Guide

Test your trained models on custom medical interviews to validate their real-world performance!

---

## Quick Start

```bash
# Test all interviews in the custom_interviews/ directory
python test_custom_interviews.py

# Test a specific interview
python test_custom_interviews.py --interview custom_interviews/example_001.json

# Use custom directory
python test_custom_interviews.py --interviews-dir my_interviews/
```

---

## JSON Format

Create JSON files in this format:

```json
{
  "name": "Interview Name",
  "expected_icd10": "J06.9",
  "description": "Optional description",
  "messages": [
    {
      "role": "doctor",
      "message": "What brings you here today?"
    },
    {
      "role": "patient",
      "message": "I have a sore throat and runny nose."
    },
    {
      "role": "doctor",
      "message": "How long have you had these symptoms?"
    },
    {
      "role": "patient",
      "message": "About three days now."
    }
  ]
}
```

### Required Fields

- **name**: Interview identifier (displayed in results)
- **expected_icd10**: The correct ICD-10 code for validation
- **messages**: Array of conversation turns

### Message Format

Each message must have:
- **role**: Either "doctor" or "patient"
- **message**: The actual text

---

## How It Works

### 1. Load All Models

The script automatically loads:
- **LLMs (untrained)**: Llama 8B, Mistral 7B
- **SLMs (finetuned)**: Llama 3B, Qwen 3B (if trained)

### 2. Test Each Interview

For each interview:
1. Converts messages to dialog format
2. Applies each model to predict ICD-10 code
3. Compares predictions with expected code
4. Shows results in console

### 3. Save Results

Results are saved to JSON for further analysis.

---

## Example Output

```
================================================================================
TESTING INTERVIEW: Fall 001 - Akute Infektion der oberen Atemwege
Expected ICD-10: J06.9
================================================================================

Dialog Preview:
--------------------------------------------------------------------------------
Doctor: Guten Tag, was führt Sie heute zu mir?
Patient: Ich habe Halsschmerzen und eine verstopfte Nase.
...
--------------------------------------------------------------------------------

================================================================================
PHASE 1: Testing Large Language Models (Untrained)
================================================================================

============================================================
Testing LLM: Meta-Llama-3.1-8B-Instruct (8B)
============================================================
Prediction: J06.9

============================================================
Testing LLM: Mistral-7B-Instruct-v0.3 (7B)
============================================================
Prediction: J00

================================================================================
PHASE 2: Testing Small Language Models (Finetuned)
================================================================================

============================================================
Testing Finetuned SLM: Llama-3.2-3B-Instruct (3B)
============================================================
Prediction: J06.9

============================================================
Testing Finetuned SLM: Qwen2.5-3B-Instruct (3B)
============================================================
Prediction: J06.9

================================================================================
RESULTS SUMMARY: Fall 001 - Akute Infektion der oberen Atemwege
================================================================================

Expected ICD-10: J06.9

Model Predictions:
--------------------------------------------------------------------------------
Meta-Llama-3.1-8B-Instruct (8B, untrained)       → J06.9          ✅ CORRECT
Mistral-7B-Instruct-v0.3 (7B, untrained)         → J00            ❌ INCORRECT
Llama-3.2-3B-Instruct (3B, finetuned)            → J06.9          ✅ CORRECT
Qwen2.5-3B-Instruct (3B, finetuned)              → J06.9          ✅ CORRECT
--------------------------------------------------------------------------------

Correct Predictions: 3/4
================================================================================
```

---

## Command Line Options

### Basic Usage

```bash
# Test all interviews in default directory
python test_custom_interviews.py

# Specify custom directory
python test_custom_interviews.py --interviews-dir path/to/interviews/

# Test single interview
python test_custom_interviews.py --interview path/to/interview.json

# Custom output location
python test_custom_interviews.py --output results/my_test.json
```

### All Options

| Option | Description | Default |
|--------|-------------|---------|
| `--interviews-dir` | Directory with interview JSON files | `custom_interviews/` |
| `--interview` | Path to single interview file | None |
| `--output` | Output path for results JSON | `outputs/custom_interview_results.json` |

---

## Creating Your Own Interviews

### Step 1: Create JSON File

```bash
# Create new interview file
nano custom_interviews/my_case.json
```

### Step 2: Write Interview

```json
{
  "name": "My Test Case",
  "expected_icd10": "I10",
  "messages": [
    {"role": "doctor", "message": "Your question here"},
    {"role": "patient", "message": "Patient's answer"},
    {"role": "doctor", "message": "Follow-up question"},
    {"role": "patient", "message": "Patient's response"}
  ]
}
```

### Step 3: Test It

```bash
python test_custom_interviews.py --interview custom_interviews/my_case.json
```

---

## Tips for Good Test Cases

### 1. Realistic Conversations

Make dialogues sound natural:
```json
✅ Good: "I've had a headache for three days now. It's on the left side and throbs."
❌ Bad: "Headache. Left side. Throbbing. Three days."
```

### 2. Include Key Symptoms

Make sure to include diagnostic information:
```json
✅ Good conversation includes:
- Chief complaint
- Symptom duration
- Symptom characteristics
- Associated symptoms
- Relevant medical history
```

### 3. Clear ICD-10 Codes

Use specific, valid ICD-10 codes:
```json
✅ "J06.9" - Acute upper respiratory infection, unspecified
✅ "I10"   - Essential (primary) hypertension
✅ "G43.0" - Migraine without aura
❌ "J06"   - Too general (missing decimal)
❌ "COLD"  - Not a valid ICD code
```

### 4. Test Edge Cases

Create interviews that test:
- **Similar conditions**: J06.9 vs J00 (common cold)
- **Rare diagnoses**: Less common ICD codes
- **Ambiguous cases**: Multiple possible diagnoses
- **Complex cases**: Patients with multiple symptoms

---

## Example Use Cases

### 1. Validate Finetuning

Test if your finetuned models actually learned:
```bash
# Create 10 test cases from different ICD categories
# Test before and after finetuning
python test_custom_interviews.py
```

### 2. Compare Model Performance

See which models work best for different conditions:
```bash
# Test respiratory infections
python test_custom_interviews.py --interview respiratory_cases/

# Test cardiovascular conditions
python test_custom_interviews.py --interview cardio_cases/
```

### 3. Error Analysis

Find where models fail:
```bash
# Test edge cases
python test_custom_interviews.py --interviews-dir edge_cases/

# Analyze which models failed on which cases
cat outputs/custom_interview_results.json | jq '.[] | select(.predictions[].predicted_icd10 != .expected_icd10)'
```

### 4. Real-World Validation

Test on actual anonymized patient data:
```bash
# Convert real cases to JSON (anonymized!)
# Test model predictions
python test_custom_interviews.py --interviews-dir real_cases/
```

---

## Results File Format

Results are saved as JSON:

```json
[
  {
    "interview_name": "Fall 001 - Upper Respiratory Infection",
    "expected_icd10": "J06.9",
    "dialog": "Doctor: ...\nPatient: ...",
    "predictions": [
      {
        "model_name": "Meta-Llama-3.1-8B-Instruct",
        "model_type": "LLM",
        "model_size": "8B",
        "training_status": "untrained",
        "predicted_icd10": "J06.9",
        "raw_output": "Based on the symptoms..."
      },
      {
        "model_name": "Llama-3.2-3B-Instruct",
        "model_type": "SLM",
        "model_size": "3B",
        "training_status": "finetuned",
        "predicted_icd10": "J06.9",
        "raw_output": "J06.9"
      }
    ]
  }
]
```

You can analyze this with:
- Python (`json` module)
- jq (command-line JSON processor)
- Excel/Google Sheets (import JSON)
- Custom analysis scripts

---

## Analyzing Results

### Python Analysis

```python
import json

# Load results
with open('outputs/custom_interview_results.json', 'r') as f:
    results = json.load(f)

# Calculate accuracy per model
model_accuracy = {}
for interview in results:
    expected = interview['expected_icd10']
    for pred in interview['predictions']:
        model = pred['model_name']
        if model not in model_accuracy:
            model_accuracy[model] = {'correct': 0, 'total': 0}

        model_accuracy[model]['total'] += 1
        if pred['predicted_icd10'].upper() == expected.upper():
            model_accuracy[model]['correct'] += 1

# Print accuracy
for model, stats in model_accuracy.items():
    accuracy = stats['correct'] / stats['total']
    print(f"{model}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
```

### Command Line Analysis

```bash
# Count correct predictions per model
jq '[.[] | .predictions[] | select(.predicted_icd10 == .expected_icd10) | .model_name] | group_by(.) | map({model: .[0], count: length})' outputs/custom_interview_results.json

# Find all incorrect predictions
jq '.[] | select(.predictions[].predicted_icd10 != .expected_icd10) | {name, expected: .expected_icd10, predictions: [.predictions[] | {model: .model_name, predicted: .predicted_icd10}]}' outputs/custom_interview_results.json
```

---

## Troubleshooting

### Issue: "Finetuned model not found"

**Cause:** SLMs haven't been trained yet

**Solution:**
```bash
# Train models first
python main.py --experiment full

# Then test
python test_custom_interviews.py
```

### Issue: "No JSON files found"

**Cause:** Wrong directory or no files

**Solution:**
```bash
# Check directory exists
ls custom_interviews/

# Check file format
file custom_interviews/*.json

# Verify JSON is valid
cat custom_interviews/example_001.json | python -m json.tool
```

### Issue: "Missing required field 'messages'"

**Cause:** Invalid JSON format

**Solution:**
```json
// Make sure your JSON has this structure:
{
  "name": "...",
  "expected_icd10": "...",
  "messages": [...]  // ← This is required!
}
```

### Issue: CUDA OOM during testing

**Cause:** Too many models in memory

**Solution:**
The script already does aggressive cleanup between models, but if you still have issues:

```python
# Edit test_custom_interviews.py
# Reduce batch processing or test fewer models at once
```

---

## Best Practices

1. **Start Small**: Test with 2-3 cases first
2. **Validate JSON**: Use `python -m json.tool` to check format
3. **Clear Expectations**: Use specific ICD-10 codes
4. **Document Cases**: Add descriptions to understand failures
5. **Version Control**: Track your test cases in git
6. **Anonymize Data**: Never include real patient information

---

## Next Steps

1. **Create test suite**: Build 20-30 test cases covering common diagnoses
2. **Automate testing**: Run after each training to catch regressions
3. **Analyze patterns**: Which models work best for which conditions?
4. **Improve training**: Use insights to enhance finetuning data

---

## Example Test Suite Structure

```
custom_interviews/
├── respiratory/
│   ├── upper_resp_001.json (J06.9)
│   ├── pneumonia_001.json (J18.9)
│   └── bronchitis_001.json (J40)
├── cardiovascular/
│   ├── hypertension_001.json (I10)
│   ├── angina_001.json (I20.9)
│   └── heart_failure_001.json (I50.9)
├── neurological/
│   ├── migraine_001.json (G43.0)
│   ├── tension_headache_001.json (G44.2)
│   └── vertigo_001.json (H81.9)
└── edge_cases/
    ├── ambiguous_001.json
    ├── rare_disease_001.json
    └── complex_comorbid_001.json
```

---

Happy Testing! 🎯
