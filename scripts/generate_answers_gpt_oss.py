#!/usr/bin/env python3
"""
Phase 2.3: Generate answers for questions using GPT-OSS-120B via vLLM
"""

import requests
import json
import argparse
import os
from tqdm import tqdm
from typing import List, Dict

VLLM_BASE_URL = "http://localhost:8001/v1"

def generate_answers_for_questions(questions_file: str, output_file: str) -> List[Dict]:
    """Generate answers using GPT-OSS-120B"""

    try:
        with open(questions_file) as f:
            questions = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load questions file: {e}")
        return []

    if not questions:
        print("[ERROR] No questions loaded from file")
        return []

    system_prompt = """You are an expert problem solver with deep understanding of logical reasoning.

For each MCQ question:
1. Think through all options carefully
2. Identify the correct answer
3. Provide reasoning in 50-100 words

Return ONLY valid JSON:
{
  "answer": "A",
  "reasoning": "..."
}"""

    answers = []

    for idx, q in tqdm(enumerate(questions), total=len(questions), desc="Generating answers"):
        try:
            if "question" not in q or "choices" not in q:
                continue

            prompt = f"""Question: {q['question']}

Options:
{chr(10).join(q['choices'])}

Provide your answer and reasoning."""

            response = requests.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": "gptopenai/gpt-oss-120b-instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.1,  # Lower for consistent answers
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0].get("message", {}).get("content", "")

                        try:
                            answer = json.loads(content)
                            if "answer" in answer and "reasoning" in answer:
                                answers.append({
                                    "question_id": idx,
                                    **answer
                                })
                        except json.JSONDecodeError:
                            pass
                except (json.JSONDecodeError, KeyError):
                    pass
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            pass

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"\n[OK] Generated {len(answers)} valid answers to {output_file}")
    print(f"   Success rate: {len(answers) / len(questions) * 100:.1f}%")
    return answers

def main():
    parser = argparse.ArgumentParser(description="Generate answers using GPT-OSS-120B via vLLM")
    parser.add_argument("--input", default="data/final/questions_training.json", help="Input questions file")
    parser.add_argument("--output", default="data/final/answers_training.json", help="Output answers file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return

    generate_answers_for_questions(args.input, args.output)

if __name__ == "__main__":
    main()
