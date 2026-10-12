#!/usr/bin/env python3
"""
Utility Bill Decoder - Level 1 & Level 2
Reads electricity bill images (K-Electric, LESCO, IESCO) and generates:
- Level 1: Structured JSON billing extraction into CSV (bill_id, json)
- Level 2: Grounded answers to customer questions into CSV (bill_id, question_id, question, answer)

Strictly adheres to WTQ Build Track competition rules:
- Uses ONLY approved Google Gemini model: gemini-3.5-flash-lite via google-genai SDK
- API key read strictly from GEMINI_API_KEY environment variable
- No hardcoded bill values, no external OCR libraries, no prebuilt extractors
- Reproducible extraction with temperature=0, top_p=1, fixed seed=42
- Robust validation, error handling, retries with exponential backoff
"""

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# Single approved model constant
MODEL_ID: str = "gemini-3.5-flash-lite"

# Allowed providers, charge types, and tax types according to competition specification
VALID_PROVIDERS = {"KE", "LESCO", "IESCO"}
VALID_CHARGE_TYPES = {
    "energy",
    "fixed",
    "fpa",
    "quarterly_adjustment",
    "surcharge",
    "meter_rent",
    "subsidy",
    "other",
}
VALID_TAX_TYPES = {
    "gst",
    "electricity_duty",
    "income_tax",
    "municipal_tax",
    "other_tax",
}

# The required Level 1 JSON schema fields
ALL_JSON_KEYS = [
    "provider",
    "tariff",
    "sanctioned_load_kw",
    "bill_month",
    "reading_date",
    "issue_date",
    "due_date",
    "previous_reading",
    "current_reading",
    "units_consumed",
    "charges",
    "total_charges",
    "taxes",
    "total_taxes",
    "current_bill",
    "arrears",
    "payable_within_due_date",
    "payable_after_due_date",
]

EXTRACTION_SYSTEM_PROMPT = """You are an expert utility bill decoder for Pakistani electricity bills (K-Electric / KE / KESC, LESCO, and IESCO).
Your task is to extract billing details from the provided electricity bill image and output strictly ONE valid JSON object conforming exactly to the specification.

EXTRACTION RULES:

DO:
1. Extract values exactly as printed on the bill.
2. Amounts are plain numbers in PKR (remove currency symbols like 'Rs.', 'PKR' and commas). E.g., 'Rs. 19,964' -> 19964. Credits and subsidies are negative: '38209CR' -> -38209. Record a subsidy as a deduction/negative amount.
3. Dates: Output in exact YYYY-MM-DD format (e.g., '22nd Sep 2026' -> '2026-09-22'). bill_month must be in exact YYYY-MM format (e.g., 'MAR 2026' -> '2026-03').
4. provider: Must be exactly one of 'KE', 'LESCO', or 'IESCO'. For K-Electric (or KESC), use 'KE'.
5. tariff: The tariff code exactly as printed (e.g. 'A1-R', 'A-1a(01)').
6. sanctioned_load_kw: Sanctioned load in kW as a number, labelled Sanc Load, S.Load, San Load, etc.
7. due_date: The last date to pay without a late payment surcharge.
8. charges: One entry per printed line in the CHARGES section. Skip blank lines and lines printed as 0. If the same type appears more than once (e.g. two energy lines), keep them as separate entries in order.
9. charge type mapping:
   - 'energy': cost of units / Variable Upto ... Units / Cost of Electricity / Energy Charges
   - 'fixed': fixed charges
   - 'fpa': fuel price adjustment (both FPA and FCA)
   - 'quarterly_adjustment': quarterly tariff adjustment (QTA or Uniform Quarterly Adjustment)
   - 'surcharge': surcharges (Additional Surcharge (PHL), F.C Surcharge, etc.)
   - 'meter_rent': meter rent or service rent
   - 'subsidy': government subsidy or relief lines (record amount as negative)
   - 'other': any other charge line
10. taxes: One entry per printed line in the TAX / GOVERNMENT section. Classify a line by the SECTION it is printed in, not by its name.
    - 'gst': sales tax or GST, including GST on FPA
    - 'electricity_duty': electricity duty, often labelled ED
    - 'income_tax': income tax / advance tax
    - 'municipal_tax': municipal or local charges printed under taxes (such as KE MUCT / KMC)
    - 'other_tax': any other tax line, such as extra tax, further tax, TV fee
11. If only a combined tax amount is printed on the bill: return taxes = [] and put that amount in total_taxes.
12. total_charges / total_taxes: The printed subtotals, or null if no subtotal is printed.
13. current_bill: The printed amount for this month's bill / current bill.
14. arrears: Previous dues carried forward. Keep the printed sign (a printed credit stays negative); null if blank.
15. payable_within_due_date: The amount payable by the due date.
16. payable_after_due_date: The amount payable after the due date. If several late-payment amounts are shown, return the HIGHEST.
17. Use null for any value not printed, blank, covered by a grey box, or unreadable.
18. The bill may be rotated, tilted, low quality or partly in Urdu. Read all sections carefully.

DO NOT:
1. DO NOT recalculate, round, correct, or sanity-fix any number.
2. DO NOT guess. When unsure or if a field is covered/not present, return null. A correct null is scored as correct.
3. DO NOT return names, addresses, CNIC, account numbers, reference numbers, consumer numbers, or meter numbers.
4. DO NOT invent extra fields, and DO NOT add commentary, markdown formatting, or code fences. Output only the pure JSON object.
5. DO NOT put a line in the wrong list because of its name. The printed section decides whether a line belongs in charges or taxes.
"""

QA_SYSTEM_PROMPT = """You are an expert utility bill decoder for Pakistani electricity bills (K-Electric / KE / LESCO / IESCO).
Your task is to answer customer questions about the provided electricity bill.

ANSWER RULES:
DO:
1. Answer in plain English, even if parts of the bill are in Urdu.
2. Use only information printed on this bill. No outside tariff, rate or price knowledge.
3. For any calculation (average, difference, count, percentage, estimate), list the numbers you read from the bill, show the arithmetic step by step, and double-check it.
4. Label every estimate as an estimate and explain how it was calculated.
5. If a question has more than one reasonable meaning, state the meaning you used or answer both.
6. If the bill does not show a needed value, say so clearly instead of guessing.
7. Be specific: quote the amounts, units and months as printed on the bill (PKR).
8. Read the usage history and billing history tables carefully, including the month labels.

DO NOT:
9. DO NOT include names, addresses, CNIC, account, reference, consumer or meter numbers.
10. DO NOT invent values or use example values in the prompt. Use only placeholders.
11. DO NOT add markdown, code fences or extra keys. Output the JSON object only.
"""

FALLBACK_QA_ANSWER = "This could not be determined from the bill."


def get_default_null_json() -> Dict[str, Any]:
    """Return a valid fallback JSON dictionary with all fields null / empty lists."""
    return {
        "provider": None,
        "tariff": None,
        "sanctioned_load_kw": None,
        "bill_month": None,
        "reading_date": None,
        "issue_date": None,
        "due_date": None,
        "previous_reading": None,
        "current_reading": None,
        "units_consumed": None,
        "charges": [],
        "total_charges": None,
        "taxes": [],
        "total_taxes": None,
        "current_bill": None,
        "arrears": None,
        "payable_within_due_date": None,
        "payable_after_due_date": None,
    }


MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_number(val: Any) -> Optional[float]:
    """
    Safely convert numeric string or number to float/int without recalculating or guessing.
    Handles 'Rs.', commas, currency labels, and 'CR' trailing credit markers.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned or cleaned.lower() == "null":
            return None
        # Check for credit marker e.g. '38209CR' or '0.53 CR'
        is_credit = False
        if cleaned.upper().endswith("CR"):
            is_credit = True
            cleaned = cleaned[:-2].strip()
        # Remove currency words/abbreviations and commas (avoid stripping decimal point)
        cleaned = re.sub(r"(?i)\brs\b\.?", "", cleaned)
        cleaned = re.sub(r"(?i)\bpkr\b\.?", "", cleaned)
        cleaned = cleaned.replace(",", "").replace(" ", "").strip()
        if not cleaned or cleaned in ("-", "."):
            return None
        try:
            num = float(cleaned)
            if is_credit and num > 0:
                num = -num
            # Convert to int if whole number
            if num.is_integer():
                return int(num)
            return round(num, 2)
        except ValueError:
            return None
    return None


def normalize_date(val: Any, is_month: bool = False) -> Optional[str]:
    """Normalize date strings to YYYY-MM-DD or YYYY-MM format without guessing."""
    if not val or not isinstance(val, str):
        return None
    cleaned = val.strip()
    if is_month:
        # Match YYYY-MM or YYYY/MM
        m = re.search(r"^(\d{4})[-/.](\d{1,2})$", cleaned)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}"
        # Match Month YYYY (e.g. 'MAR 2026')
        m = re.search(r"([a-zA-Z]{3,9})[\s,.-]+(\d{4})", cleaned)
        if m:
            mon = m.group(1)[:3].lower()
            if mon in MONTH_MAP:
                return f"{m.group(2)}-{MONTH_MAP[mon]:02d}"
        # Match YYYY Month (e.g. '2026 MAR')
        m = re.search(r"(\d{4})[\s,.-]+([a-zA-Z]{3,9})", cleaned)
        if m:
            mon = m.group(2)[:3].lower()
            if mon in MONTH_MAP:
                return f"{m.group(1)}-{MONTH_MAP[mon]:02d}"
        return cleaned if re.match(r"^\d{4}-\d{2}$", cleaned) else None
    else:
        # Match YYYY-MM-DD
        m = re.search(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", cleaned)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        # Match 22nd Sep 2026 / 22 Sep 2026
        m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[\s,.-]+([a-zA-Z]{3,9})[\s,.-]+(\d{4})", cleaned)
        if m:
            mon = m.group(2)[:3].lower()
            if mon in MONTH_MAP:
                return f"{m.group(3)}-{MONTH_MAP[mon]:02d}-{int(m.group(1)):02d}"
        return cleaned if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned) else None


def validate_and_clean_json(raw_data: Any) -> Tuple[bool, Dict[str, Any], str]:
    """
    Validates and normalizes extracted JSON according to strict Level 1 rules.
    Returns (is_valid, cleaned_dict, error_message).
    """
    if not isinstance(raw_data, dict):
        return False, get_default_null_json(), "Output is not a JSON object"

    cleaned: Dict[str, Any] = get_default_null_json()

    # Provider
    raw_provider = raw_data.get("provider")
    if raw_provider in VALID_PROVIDERS:
        cleaned["provider"] = raw_provider
    elif isinstance(raw_provider, str) and raw_provider.upper() in ("KESC", "KE"):
        cleaned["provider"] = "KE"
    else:
        cleaned["provider"] = None

    # Tariff
    cleaned["tariff"] = str(raw_data.get("tariff")).strip() if raw_data.get("tariff") is not None else None

    # Sanctioned load
    cleaned["sanctioned_load_kw"] = normalize_number(raw_data.get("sanctioned_load_kw"))

    # Dates
    cleaned["bill_month"] = normalize_date(raw_data.get("bill_month"), is_month=True)
    cleaned["reading_date"] = normalize_date(raw_data.get("reading_date"), is_month=False)
    cleaned["issue_date"] = normalize_date(raw_data.get("issue_date"), is_month=False)
    cleaned["due_date"] = normalize_date(raw_data.get("due_date"), is_month=False)

    # Meter readings and units
    cleaned["previous_reading"] = normalize_number(raw_data.get("previous_reading"))
    cleaned["current_reading"] = normalize_number(raw_data.get("current_reading"))
    cleaned["units_consumed"] = normalize_number(raw_data.get("units_consumed"))

    # Charges list
    raw_charges = raw_data.get("charges")
    valid_charges: List[Dict[str, Any]] = []
    if isinstance(raw_charges, list):
        for item in raw_charges:
            if isinstance(item, dict):
                c_type = item.get("type")
                c_amt = normalize_number(item.get("amount"))
                if c_type in VALID_CHARGE_TYPES and c_amt is not None:
                    # Subsidies must be negative
                    if c_type == "subsidy" and c_amt > 0:
                        c_amt = -c_amt
                    valid_charges.append({"type": c_type, "amount": c_amt})
    cleaned["charges"] = valid_charges

    # Total charges
    cleaned["total_charges"] = normalize_number(raw_data.get("total_charges"))

    # Taxes list
    raw_taxes = raw_data.get("taxes")
    valid_taxes: List[Dict[str, Any]] = []
    if isinstance(raw_taxes, list):
        for item in raw_taxes:
            if isinstance(item, dict):
                t_type = item.get("type")
                t_amt = normalize_number(item.get("amount"))
                if t_type in VALID_TAX_TYPES and t_amt is not None:
                    valid_taxes.append({"type": t_type, "amount": t_amt})
    cleaned["taxes"] = valid_taxes

    # Total taxes
    cleaned["total_taxes"] = normalize_number(raw_data.get("total_taxes"))

    # Amounts
    cleaned["current_bill"] = normalize_number(raw_data.get("current_bill"))
    cleaned["arrears"] = normalize_number(raw_data.get("arrears"))
    cleaned["payable_within_due_date"] = normalize_number(raw_data.get("payable_within_due_date"))
    cleaned["payable_after_due_date"] = normalize_number(raw_data.get("payable_after_due_date"))

    # Ensure all required keys exist
    for key in ALL_JSON_KEYS:
        if key not in cleaned:
            cleaned[key] = [] if key in ("charges", "taxes") else None

    return True, cleaned, ""


def call_gemini_extraction(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    max_retries: int = 3,
    delay: float = 4.0,
) -> Dict[str, Any]:
    """
    Sends bill image to Gemini API with Level 1 extraction prompt.
    Retries up to max_retries with exponential backoff on errors/rate limits.
    """
    schema_definition = {
        "type": "OBJECT",
        "properties": {
            "provider": {"type": "STRING", "enum": ["KE", "LESCO", "IESCO"], "nullable": True},
            "tariff": {"type": "STRING", "nullable": True},
            "sanctioned_load_kw": {"type": "NUMBER", "nullable": True},
            "bill_month": {"type": "STRING", "nullable": True},
            "reading_date": {"type": "STRING", "nullable": True},
            "issue_date": {"type": "STRING", "nullable": True},
            "due_date": {"type": "STRING", "nullable": True},
            "previous_reading": {"type": "NUMBER", "nullable": True},
            "current_reading": {"type": "NUMBER", "nullable": True},
            "units_consumed": {"type": "NUMBER", "nullable": True},
            "charges": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "type": {
                            "type": "STRING",
                            "enum": [
                                "energy",
                                "fixed",
                                "fpa",
                                "quarterly_adjustment",
                                "surcharge",
                                "meter_rent",
                                "subsidy",
                                "other",
                            ],
                        },
                        "amount": {"type": "NUMBER"},
                    },
                    "required": ["type", "amount"],
                },
            },
            "total_charges": {"type": "NUMBER", "nullable": True},
            "taxes": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "type": {
                            "type": "STRING",
                            "enum": [
                                "gst",
                                "electricity_duty",
                                "income_tax",
                                "municipal_tax",
                                "other_tax",
                            ],
                        },
                        "amount": {"type": "NUMBER"},
                    },
                    "required": ["type", "amount"],
                },
            },
            "total_taxes": {"type": "NUMBER", "nullable": True},
            "current_bill": {"type": "NUMBER", "nullable": True},
            "arrears": {"type": "NUMBER", "nullable": True},
            "payable_within_due_date": {"type": "NUMBER", "nullable": True},
            "payable_after_due_date": {"type": "NUMBER", "nullable": True},
        },
        "required": ALL_JSON_KEYS,
    }

    config = types.GenerateContentConfig(
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        response_mime_type="application/json",
        response_schema=schema_definition,
    )

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        "Extract all billing details from this electricity bill image according to the rules and output strictly conforming JSON.",
    ]

    backoff = delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config,
            )

            response_text = response.text.strip() if response.text else "{}"
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

            parsed = json.loads(response_text)
            is_valid, cleaned, err = validate_and_clean_json(parsed)
            if is_valid:
                return cleaned
            else:
                print(f"Validation issue on attempt {attempt}: {err}", file=sys.stderr)
        except Exception as e:
            print(f"API call error on attempt {attempt}/{max_retries}: {e}", file=sys.stderr)

        if attempt < max_retries:
            print(f"Backing off for {backoff:.1f}s before retry...", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2.0

    print("All retries exhausted. Returning fallback null JSON object.", file=sys.stderr)
    return get_default_null_json()


def call_gemini_questions(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    questions: List[Tuple[str, str]],
    max_retries: int = 3,
    delay: float = 4.0,
) -> Dict[str, str]:
    """
    Sends bill image and all questions for that bill to Gemini in ONE API call.
    Returns a dictionary mapping question_id -> text answer.
    """
    question_ids = [qid for qid, _ in questions]
    properties_schema = {qid: {"type": "STRING"} for qid in question_ids}

    schema_definition = {
        "type": "OBJECT",
        "properties": {
            "answers": {
                "type": "OBJECT",
                "properties": properties_schema,
                "required": question_ids,
            }
        },
        "required": ["answers"],
    }

    config = types.GenerateContentConfig(
        system_instruction=QA_SYSTEM_PROMPT,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        response_mime_type="application/json",
        response_schema=schema_definition,
    )

    questions_prompt_lines = ["Answer all of the following questions grounded strictly in this bill:"]
    for qid, qtext in questions:
        questions_prompt_lines.append(f"[{qid}] {qtext}")

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        "\n".join(questions_prompt_lines),
    ]

    backoff = delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config,
            )

            response_text = response.text.strip() if response.text else "{}"
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

            parsed = json.loads(response_text)
            answers_dict = parsed.get("answers", {})
            if isinstance(answers_dict, dict):
                # Ensure all requested question_ids have non-empty answers
                cleaned_answers: Dict[str, str] = {}
                for qid in question_ids:
                    ans = answers_dict.get(qid)
                    if ans and isinstance(ans, str) and ans.strip():
                        cleaned_answers[qid] = ans.strip()
                    else:
                        cleaned_answers[qid] = FALLBACK_QA_ANSWER
                return cleaned_answers
        except Exception as e:
            print(f"API call error on attempt {attempt}/{max_retries} for questions: {e}", file=sys.stderr)

        if attempt < max_retries:
            print(f"Backing off for {backoff:.1f}s before retry...", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2.0

    print("All retries exhausted for questions. Returning fallback answers.", file=sys.stderr)
    return {qid: FALLBACK_QA_ANSWER for qid in question_ids}


def find_image_path(bills_dir: str, bill_id: str) -> Optional[str]:
    """Helper to find an image file matching a bill_id."""
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        candidate = os.path.join(bills_dir, f"{bill_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def read_image_bytes_and_mime(image_path: str) -> Tuple[bytes, str]:
    """Reads image bytes and infers MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    with open(image_path, "rb") as img_f:
        return img_f.read(), mime_type


def process_level1(
    client: genai.Client,
    bills_dir: str,
    out_csv_path: str,
    template_path: Optional[str] = None,
    delay: float = 4.0,
    limit: Optional[int] = None,
) -> None:
    """Processes bill images for Level 1 extraction and writes output CSV."""
    bill_ids_ordered: List[str] = []
    if template_path and os.path.exists(template_path):
        print(f"Reading Level 1 bill order from template: {template_path}")
        with open(template_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    bill_ids_ordered.append(row[0].strip())
    else:
        if not os.path.isdir(bills_dir):
            print(f"ERROR: Bills directory '{bills_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)
        valid_exts = {".png", ".jpg", ".jpeg"}
        filenames = sorted([
            f for f in os.listdir(bills_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        bill_ids_ordered = [os.path.splitext(f)[0] for f in filenames]

    if limit is not None and limit > 0:
        bill_ids_ordered = bill_ids_ordered[:limit]

    print(f"Total bills for Level 1: {len(bill_ids_ordered)}")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)

    results: List[Tuple[str, str]] = []
    for idx, bill_id in enumerate(bill_ids_ordered, start=1):
        print(f"[Level 1 {idx}/{len(bill_ids_ordered)}] Processing bill: {bill_id} ...")
        image_path = find_image_path(bills_dir, bill_id)

        if not image_path:
            print(f"Warning: Image file for bill_id '{bill_id}' not found. Writing nulls.", file=sys.stderr)
            bill_json = get_default_null_json()
        else:
            image_bytes, mime_type = read_image_bytes_and_mime(image_path)
            bill_json = call_gemini_extraction(
                client=client,
                image_bytes=image_bytes,
                mime_type=mime_type,
                max_retries=3,
                delay=delay,
            )

        single_line_json = json.dumps(bill_json, ensure_ascii=False, separators=(",", ":"))
        results.append((bill_id, single_line_json))

        if idx < len(bill_ids_ordered):
            time.sleep(delay)

    with open(out_csv_path, mode="w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["bill_id", "json"])
        for bill_id, json_str in results:
            writer.writerow([bill_id, json_str])

    print(f"Done Level 1! Successfully wrote {len(results)} rows to {out_csv_path}")


def process_level2(
    client: genai.Client,
    bills_dir: str,
    questions_csv_path: str,
    out2_csv_path: str,
    delay: float = 4.0,
    limit: Optional[int] = None,
) -> None:
    """Processes bill questions for Level 2 and writes output CSV."""
    if not os.path.exists(questions_csv_path):
        print(f"ERROR: Level 2 questions template '{questions_csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    template_rows: List[Dict[str, str]] = []
    bill_questions_map: Dict[str, List[Tuple[str, str]]] = {}
    bill_order: List[str] = []

    with open(questions_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            template_rows.append(row)
            b_id = row.get("bill_id", "").strip()
            q_id = row.get("question_id", "").strip()
            q_text = row.get("question", "").strip()
            if b_id:
                if b_id not in bill_questions_map:
                    bill_questions_map[b_id] = []
                    bill_order.append(b_id)
                bill_questions_map[b_id].append((q_id, q_text))

    if limit is not None and limit > 0:
        bill_order = bill_order[:limit]
        template_rows = [r for r in template_rows if r.get("bill_id") in bill_order]

    print(f"Total bills for Level 2: {len(bill_order)} (Total questions rows: {len(template_rows)})")
    os.makedirs(os.path.dirname(os.path.abspath(out2_csv_path)), exist_ok=True)

    # For each bill, make exactly ONE call sending the image + all questions
    bill_answers_cache: Dict[str, Dict[str, str]] = {}
    for idx, b_id in enumerate(bill_order, start=1):
        print(f"[Level 2 {idx}/{len(bill_order)}] Answering questions for bill: {b_id} ...")
        q_list = bill_questions_map[b_id]
        image_path = find_image_path(bills_dir, b_id)

        if not image_path:
            print(f"Warning: Image file for bill_id '{b_id}' not found. Using fallback answer.", file=sys.stderr)
            bill_answers_cache[b_id] = {qid: FALLBACK_QA_ANSWER for qid, _ in q_list}
        else:
            image_bytes, mime_type = read_image_bytes_and_mime(image_path)
            answers = call_gemini_questions(
                client=client,
                image_bytes=image_bytes,
                mime_type=mime_type,
                questions=q_list,
                max_retries=3,
                delay=delay,
            )
            bill_answers_cache[b_id] = answers

        if idx < len(bill_order):
            time.sleep(delay)

    # Write output CSV preserving original row order and columns
    with open(out2_csv_path, mode="w", encoding="utf-8", newline="") as out_f:
        fieldnames = ["bill_id", "question_id", "question", "answer"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in template_rows:
            b_id = row.get("bill_id", "").strip()
            q_id = row.get("question_id", "").strip()
            q_text = row.get("question", "")

            ans = bill_answers_cache.get(b_id, {}).get(q_id, FALLBACK_QA_ANSWER)
            if not ans or not ans.strip():
                ans = FALLBACK_QA_ANSWER

            writer.writerow({
                "bill_id": b_id,
                "question_id": q_id,
                "question": q_text,
                "answer": ans,
            })

    print(f"Done Level 2! Successfully wrote {len(template_rows)} rows to {out2_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract electricity bill details and answer questions using Google Gemini API."
    )
    # Level 1 Arguments
    parser.add_argument(
        "--bills",
        required=True,
        help="Path to directory containing bill images (.png, .jpg).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV file path for Level 1 extraction (e.g. output/level1.csv).",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template CSV path for Level 1 (e.g. level1.csv) to preserve row order.",
    )
    # Level 2 Arguments
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to level2.csv template containing customer questions.",
    )
    parser.add_argument(
        "--out2",
        default=None,
        help="Output CSV file path for Level 2 answers (e.g. output/level2.csv).",
    )
    # Shared Arguments
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Delay in seconds between requests to prevent rate limit issues (default: 4.0s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of bills to process (useful for testing).",
    )

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Check for .env file in parent directories or current directory
        env_candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        ]
        for env_path in env_candidates:
            if os.path.isfile(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("GEMINI_API_KEY="):
                                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                if api_key:
                                    os.environ["GEMINI_API_KEY"] = api_key
                                    break
                except Exception:
                    pass
            if api_key:
                break

    if not api_key:
        print("ERROR: Environment variable GEMINI_API_KEY is not set (and not found in .env).", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Process Level 1 if --out is specified
    if args.out:
        process_level1(
            client=client,
            bills_dir=args.bills,
            out_csv_path=args.out,
            template_path=args.template,
            delay=args.delay,
            limit=args.limit,
        )

    # Process Level 2 if --questions is specified
    if args.questions:
        if not args.out2:
            print("ERROR: --out2 must be specified when --questions is provided.", file=sys.stderr)
            sys.exit(1)
        process_level2(
            client=client,
            bills_dir=args.bills,
            questions_csv_path=args.questions,
            out2_csv_path=args.out2,
            delay=args.delay,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
