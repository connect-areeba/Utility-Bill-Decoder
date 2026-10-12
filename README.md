# Utility Bill Decoder - Level 1 & Level 2

A command-line Python utility built for the **Women Tech Quest (WTQ) Build Track: "Utility Bill Decoder"** hackathon. It processes electricity utility bill images (K-Electric, LESCO, IESCO) and generates:
- **Level 1**: Extracted structured billing details as JSON into CSV (`bill_id,json`).
- **Level 2**: Answers to customer billing questions grounded strictly in bill contents into CSV (`bill_id,question_id,question,answer`).

---

## 1. Project Overview

The utility bill decoder processes Pakistani electricity utility bills where sensitive customer identifiers (names, addresses, CNICs, account numbers) are masked. It leverages vision capabilities via the official Google Gemini API to extract non-identifying billing details, readings, line-item charges, taxes, and payable amounts, as well as answer customer questions grounded purely in the bill image.

---

## 2. Setup

### Runtime
- **Python 3.14** (fully compatible with Python 3.11+)

### Installation
Clone or navigate to the project directory and install the required dependencies:
```bash
pip install -r source/requirements.txt
```

> **Note on Windows Execution**:
> If running in PowerShell where `python` refers to an alternate interpreter, specify the installed Python 3.14 launcher or path directly, e.g.:
> `py source/extract.py ...`

### Environment Configuration
The script securely reads the API key from the `GEMINI_API_KEY` environment variable. Never hard-code your API key.

**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

**On Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_actual_gemini_api_key_here
```

**On Linux / macOS:**
```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

A template `.env.example` is provided in `source/` for reference.

---

## 3. Exact Run Commands

### Level 1: Extraction Only
To process all bill images in a folder and write the results to `output/level1.csv`:
```bash
python source/extract.py --bills train --out output/level1.csv
```

With a template CSV (e.g., `level1.csv`) to preserve row order:
```bash
python source/extract.py --bills test/bills --template test/csv/level1.csv --out output/level1.csv
```

### Level 2: Answering Customer Questions Only
When answering questions from a template CSV (e.g., `level2.csv`):
```bash
python source/extract.py --bills test/bills --questions test/csv/level2.csv --out2 output/level2.csv
```

### Combined Execution (Both Level 1 and Level 2)
You can run both levels in a single command:
```bash
python source/extract.py --bills test/bills --template test/csv/level1.csv --out output/level1.csv --questions test/csv/level2.csv --out2 output/level2.csv
```

### CLI Arguments
- `--bills <path>`: (Required) Directory containing bill images (`.png`, `.jpg`).
- `--out <path>`: (Optional) Output path for Level 1 CSV file (`bill_id,json`).
- `--template <path>`: (Optional) Template CSV path for Level 1 (reads `bill_id` values and preserves row order).
- `--questions <path>`: (Optional) Template CSV path for Level 2 (contains columns `bill_id,question_id,question,answer`).
- `--out2 <path>`: (Optional, required if `--questions` is provided) Output path for Level 2 CSV file.
- `--delay <seconds>`: (Optional, default `4.0`) Inter-request delay in seconds to remain comfortably within API rate limits (15 RPM free tier).
- `--limit <N>`: (Optional) Limit processing to the first `N` bills for testing.

---

## 4. Models, Services, and Frameworks Used

- **Approved Vision LLM**: `gemini-3.5-flash-lite` via the official Google Gemini API (`google-genai` SDK).
- **Libraries Used**:
  - `google-genai==2.29.0` (Official Google GenAI SDK for API interaction)
  - Python Standard Library: `argparse`, `csv`, `json`, `mimetypes`, `os`, `re`, `sys`, `time`, `typing`

> **Note on Compliance:**
> - No OCR libraries (no Tesseract, no PaddleOCR) are used. The vision model inspects the images directly.
> - No prebuilt invoice/receipt extractors or external cloud services (no Azure Prebuilt Invoice, no AWS Textract, no Mindee, no Document AI) are used.
> - No secondary or fallback models are called. Only `gemini-3.5-flash-lite` is utilized.

---

## 5. AI Usage

- **Google Antigravity**: AI coding assistant used to design, write, inspect, and debug the Python code, prompt construction, and documentation.
- **Other AI Tools / Chatbots**: None.

---

## 6. Design Notes & Reproducibility

### Reproducibility Settings
To ensure identical reruns and eliminate nondeterminism:
- `temperature = 0.0`: Maximizes deterministic, greedy token selection.
- `top_p = 1.0`: Eliminates stochastic nucleus truncation.
- `seed = 42`: Fixed random seed passed directly into the generation configuration.
- `response_mime_type = "application/json"`: Enforces strict structured output.
- `response_schema`: Provided directly to the API using Google GenAI schema enforcement.
- **Static Prompts**: Both Level 1 and Level 2 prompt instructions remain strictly identical across all runs.

### Level 1 Schema Validation & Integrity
- Enforces all 18 required fields (`null` when absent or unreadable, `[]` for empty lists).
- Cleans currency symbols (`Rs.`, commas) and parses credit indicators (`38209CR` $\rightarrow$ `-38209`).
- No recalculating totals, rounding, correcting amounts, or guessing values.

### Level 2 Design & Rate Limit Efficiency
- **Batching per Bill**: For each bill, all questions assigned to that bill are submitted together in **ONE single API call**. The model returns a structured JSON mapping `{"answers": {"Q1": "...", "Q2": "..."}}`. This stays well within free-tier rate limits (15 RPM) while minimizing token overhead.
- **Strict Grounding**: Answers are instructed to cite printed numbers, units, months, and explicit calculation steps without hallucinating outside tariffs.
- **No Empty Cells**: If a bill cannot be processed or a question cannot be resolved, it falls back to `"This could not be determined from the bill."`, ensuring every cell in the CSV is populated.

### Error Handling & Graceful Degradation
- **Exponential Backoff**: Handles transient network errors and HTTP 429 rate limits with up to 3 automatic retries.
- **Safe Fallback**: Prevents crashes by populating standard fallback structures if an image fails all retries.
- **CSV Quoting**: Output is serialized via Python's standard `csv.writer` and `csv.DictWriter` with proper UTF-8 encoding and quotation handling.
