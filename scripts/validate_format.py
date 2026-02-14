#!/usr/bin/env python3
"""
Phase 5.2: Format validation for questions and answers
"""

import json
import argparse
import os
from typing import Dict, List, Tuple
from pathlib import Path

def validate_question(q: Dict) -> Tuple[bool, List[str]]:
    """Validate question format and return (is_valid, errors)"""
    errors = []

    # Check required fields
    required_fields = ["topic", "question", "choices", "answer", "explanation"]
    for field in required_fields:
        if field not in q:
            errors.append(f"Missing field: {field}")

    # Check choices format
    if "choices" in q:
        if not isinstance(q["choices"], list) or len(q["choices"]) != 4:
            errors.append(f"Choices must be a list of 4 items, got {len(q.get('choices', []))}")
        else:
            for i, choice in enumerate(q["choices"]):
                expected_prefix = chr(65 + i)  # A, B, C, D
                if not isinstance(choice, str) or not choice.startswith(f"{expected_prefix})"):
                    errors.append(f"Choice {i} should start with '{expected_prefix})'")

    # Check answer format (single letter A-D)
    if "answer" in q:
        if not isinstance(q["answer"], str) or q["answer"] not in ["A", "B", "C", "D"]:
            errors.append(f"Answer must be A, B, C, or D, got: {q['answer']}")

    # Check explanation is string
    if "explanation" in q:
        if not isinstance(q["explanation"], str):
            errors.append("Explanation must be a string")
        elif len(q["explanation"]) > 500:
            errors.append(f"Explanation too long: {len(q['explanation'])} chars (max 500)")

    # Check topic
    if "topic" in q:
        valid_topics = ["Syllogisms", "Seating Arrangements", "Blood Relations", "Alphanumeric Series"]
        if q["topic"] not in valid_topics:
            errors.append(f"Invalid topic: {q['topic']}")

    return len(errors) == 0, errors

def validate_answer(a: Dict) -> Tuple[bool, List[str]]:
    """Validate answer format"""
    errors = []

    # Check required fields
    required_fields = ["answer", "reasoning"]
    for field in required_fields:
        if field not in a:
            errors.append(f"Missing field: {field}")

    # Check answer format
    if "answer" in a:
        if not isinstance(a["answer"], str) or a["answer"] not in ["A", "B", "C", "D"]:
            errors.append(f"Answer must be A, B, C, or D, got: {a['answer']}")

    # Check reasoning
    if "reasoning" in a:
        if not isinstance(a["reasoning"], str):
            errors.append("Reasoning must be a string")
        elif len(a["reasoning"]) > 300:
            errors.append(f"Reasoning too long: {len(a['reasoning'])} chars (max 300)")

    return len(errors) == 0, errors

def main():
    parser = argparse.ArgumentParser(description="Validate format of questions and answers")
    parser.add_argument("--questions", help="Path to questions file")
    parser.add_argument("--answers", help="Path to answers file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed errors")
    args = parser.parse_args()

    if args.questions:
        print(f"\n[*] Validating questions: {args.questions}")
        if not os.path.exists(args.questions):
            print(f"[ERROR] File not found: {args.questions}")
            return

        try:
            with open(args.questions) as f:
                questions = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {args.questions}: {e}")
            return
        except Exception as e:
            print(f"[ERROR] Error reading file: {e}")
            return

        valid_count = 0
        invalid_count = 0

        for idx, q in enumerate(questions):
            is_valid, errors = validate_question(q)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if args.verbose:
                    print(f"\n[ERROR] Question {idx}:")
                    for error in errors:
                        print(f"   - {error}")

        print(f"\n[OK] Valid: {valid_count}/{len(questions)} ({valid_count/len(questions)*100:.1f}%)")
        print(f"[ERROR] Invalid: {invalid_count}/{len(questions)}")

        if valid_count / len(questions) < 0.5:
            print("[WARNING]  WARNING: Less than 50% valid questions! This will cause disqualification.")

    if args.answers:
        print(f"\n[*] Validating answers: {args.answers}")
        if not os.path.exists(args.answers):
            print(f"[ERROR] File not found: {args.answers}")
            return

        try:
            with open(args.answers) as f:
                answers = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {args.answers}: {e}")
            return
        except Exception as e:
            print(f"[ERROR] Error reading file: {e}")
            return

        valid_count = 0
        invalid_count = 0

        for idx, a in enumerate(answers):
            is_valid, errors = validate_answer(a)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if args.verbose:
                    print(f"\n[ERROR] Answer {idx}:")
                    for error in errors:
                        print(f"   - {error}")

        print(f"\n[OK] Valid: {valid_count}/{len(answers)} ({valid_count/len(answers)*100:.1f}%)")
        print(f"[ERROR] Invalid: {invalid_count}/{len(answers)}")

if __name__ == "__main__":
    main()
