#!/usr/bin/env python3
"""
Interactive manual bill tester.
Allows you to test any bill image with:
1. Level 1: Full structured JSON extraction printed in pretty format.
2. Level 2: Custom question answering - ask any question interactively!
"""

import os
import sys
import json
import argparse
import mimetypes

from google import genai
from source.extract import (
    MODEL_ID,
    call_gemini_extraction,
    call_gemini_questions,
    read_image_bytes_and_mime,
)

def load_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        ]
        for env_path in env_candidates:
            if os.path.isfile(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("GEMINI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            if api_key:
                break
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment or .env file.")
        sys.exit(1)
    return genai.Client(api_key=api_key)

def test_bill(image_path: str, custom_question: str = None):
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    client = load_gemini_client()
    image_bytes, mime_type = read_image_bytes_and_mime(image_path)
    bill_filename = os.path.basename(image_path)

    print("\n" + "=" * 65)
    print(f" TESTING BILL: {bill_filename}")
    print(f" Model: {MODEL_ID}")
    print("=" * 65)

    # 1. Level 1 Extraction
    print("\n[1] Running Level 1 JSON Extraction...")
    extracted_json = call_gemini_extraction(
        client=client,
        image_bytes=image_bytes,
        mime_type=mime_type,
        max_retries=3,
        delay=1.0,
    )

    print("\n--- LEVEL 1 EXTRACTED DATA ---")
    print(f"Provider:                {extracted_json.get('provider')}")
    print(f"Tariff:                  {extracted_json.get('tariff')}")
    print(f"Sanctioned Load (kW):    {extracted_json.get('sanctioned_load_kw')}")
    print(f"Bill Month:              {extracted_json.get('bill_month')}")
    print(f"Due Date:                {extracted_json.get('due_date')}")
    print(f"Units Consumed:          {extracted_json.get('units_consumed')}")
    print(f"Total Charges:           {extracted_json.get('total_charges')}")
    print(f"Total Taxes:             {extracted_json.get('total_taxes')}")
    print(f"Current Bill:            {extracted_json.get('current_bill')}")
    print(f"Payable Within Due Date: {extracted_json.get('payable_within_due_date')}")
    print(f"Payable After Due Date:  {extracted_json.get('payable_after_due_date')}")

    print("\nCharges Breakdown:")
    for ch in extracted_json.get("charges", []):
        print(f"  - {ch.get('type'):22}: {ch.get('amount')}")

    print("\nTaxes Breakdown:")
    for tx in extracted_json.get("taxes", []):
        print(f"  - {tx.get('type'):22}: {tx.get('amount')}")

    # 2. Level 2 Question Answering
    questions_to_ask = []
    if custom_question:
        questions_to_ask.append(("CUSTOM_Q", custom_question))
    else:
        # Default interesting questions
        questions_to_ask = [
            ("Q1", "What is my total payable amount after the due date and what is the late surcharge?"),
            ("Q2", "Which month had the highest electricity consumption according to the billing history?"),
        ]

    print("\n" + "-" * 65)
    print("[2] Running Level 2 Grounded Question Answering...")
    answers = call_gemini_questions(
        client=client,
        image_bytes=image_bytes,
        mime_type=mime_type,
        questions=questions_to_ask,
        max_retries=3,
        delay=1.0,
    )

    print("\n--- LEVEL 2 ANSWERS ---")
    for qid, qtext in questions_to_ask:
        print(f"\nQuestion: {qtext}")
        print(f"Answer:   {answers.get(qid)}")

    print("\n" + "=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually test any bill image.")
    parser.add_argument("image", help="Path to the bill image (e.g. test/bills/KESC_0002.png)")
    parser.add_argument("-q", "--question", default=None, help="Optional custom question to ask about the bill")
    args = parser.parse_args()

    test_bill(args.image, args.question)
