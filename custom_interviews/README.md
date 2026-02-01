# Custom Interviews Directory

Place your test interview JSON files here!

## Quick Reference

### JSON Format
```json
{
  "name": "Interview Name",
  "expected_icd10": "ICD10_CODE",
  "messages": [
    {"role": "doctor", "message": "..."},
    {"role": "patient", "message": "..."}
  ]
}
```

### Test Your Interviews
```bash
# From project root
python test_custom_interviews.py
```

## Example Files Included

1. **example_001_upper_respiratory_infection.json** - J06.9
   - Common cold/upper respiratory infection
   - Good baseline test case

2. **example_002_hypertension.json** - I10
   - Essential hypertension
   - Tests cardiovascular diagnosis

3. **example_003_migraine.json** - G43.0
   - Migraine without aura
   - Tests neurological diagnosis

## Create Your Own

1. Copy an example file
2. Modify the conversation
3. Set the expected ICD-10 code
4. Run the test!

See [CUSTOM_INTERVIEW_TESTING.md](../CUSTOM_INTERVIEW_TESTING.md) for detailed documentation.
