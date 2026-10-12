# Demonstration Guide - Utility Bill Decoder

This directory documents how to run an interactive demonstration of the solution on the competition bills.

---

## 1. Prerequisites
- Python 3.11+ (Python 3.14 verified)
- Dependencies installed:
  ```bash
  pip install -r source/requirements.txt
  ```
- Environment configured with a valid Google Gemini API key:
  ```bash
  export GEMINI_API_KEY="your_api_key_here"
  ```
  *(Or set in PowerShell: `$env:GEMINI_API_KEY="your_api_key_here"`)*

---

## 2. Running the Live Demo

### Option A: Interactive Single-Bill Inspection
To demonstrate extraction and customer question answering on a sample bill:
```bash
python test_bill.py test/bills/KESC_0002.png -q "How much tax am I paying?"
```
**Expected Output**:
1. Formatted Level 1 breakdown (Provider, Tariff, Units, Total Charges, Taxes, Due Date, Amounts).
2. Level 2 plain-English grounded answer showing explicit values and arithmetic.

### Option B: Batch Generation
To run end-to-end generation across the competition test bills:
```bash
python source/extract.py --bills test/bills --template test/csv/level1.csv --out output/level1.csv --questions test/csv/level2.csv --out2 output/level2.csv
```

---

## 3. Verified Results
The pre-generated results are included in the `output/` directory:
- `output/level1.csv`: 10 rows containing the structured JSON extractions.
- `output/level2.csv`: 120 rows containing answers to all 12 questions for each of the 10 bills.
