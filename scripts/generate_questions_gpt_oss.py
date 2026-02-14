#!/usr/bin/env python3
"""
Phase 2.2: Generate questions using GPT-OSS-120B via vLLM
Targets: 500 questions per topic (Syllogisms, Seating, Blood Relations, Series)
"""

import requests
import json
import argparse
from tqdm import tqdm
from typing import List, Dict
import time

VLLM_BASE_URL = "http://localhost:8001/v1"

def generate_batch_questions(topic: str, num_questions: int = 500) -> List[Dict]:
    """Generate questions in batches using vLLM"""

    if not topic:
        print("[ERROR] Topic cannot be empty")
        return []

    system_prompt = f"""You are an expert-level examiner creating extremely difficult MCQ questions about {topic}.

CRITICAL RULES:
1. Generate ONLY ONE question per response (not multiple)
2. Topic must be strictly: {topic}
3. For Seating Arrangements: NEVER numeric questions like "how many permutations"
4. Make questions genuinely hard - trick 50%+ of experts
5. Return ONLY valid JSON (no other text)

FORMAT (must be exact):
{{
  "topic": "{topic}",
  "question": "...",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "A",
  "explanation": "Brief explanation under 100 words"
}}"""

    questions = []
    failed_count = 0

    for i in tqdm(range(num_questions), desc=f"Generating {topic}"):
        try:
            response = requests.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": "gptopenai/gpt-oss-120b-instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Generate one extremely difficult MCQ question. Return ONLY valid JSON."}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.8,
                    "top_p": 0.95,
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0].get("message", {}).get("content", "")

                        # Parse JSON
                        try:
                            question = json.loads(content)
                            # Validate required fields
                            if all(k in question for k in ["topic", "question", "choices", "answer", "explanation"]):
                                questions.append(question)
                            else:
                                failed_count += 1
                        except json.JSONDecodeError:
                            failed_count += 1
                    else:
                        failed_count += 1
                except (json.JSONDecodeError, KeyError):
                    failed_count += 1
            else:
                failed_count += 1
        except requests.exceptions.Timeout:
            failed_count += 1
        except requests.exceptions.ConnectionError:
            failed_count += 1
        except Exception as e:
            failed_count += 1

    return questions

def main():
    parser = argparse.ArgumentParser(description="Generate questions using GPT-OSS-120B via vLLM")
    parser.add_argument("--output", default="data/final/questions_training.json", help="Output file path")
    parser.add_argument("--per-topic", type=int, default=500, help="Number of questions per topic")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for vLLM server check (seconds)")
    args = parser.parse_args()

    if args.per_topic <= 0:
        print("[ERROR] per-topic must be greater than 0")
        return

    # Wait for vLLM server to be ready
    print(f"[*] Waiting for vLLM server at {VLLM_BASE_URL}...")
    print(f"   Timeout: {args.timeout}s")
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < args.timeout:
        try:
            response = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
            if response.status_code == 200:
                server_ready = True
                print("[OK] vLLM server is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[WARNING]  Server check error: {type(e).__name__}")
        time.sleep(2)

    if not server_ready:
        print(f"[ERROR] vLLM server did not respond within {args.timeout}s")
        print("   Make sure vLLM is running:")
        print(f"   vllm serve gptopenai/gpt-oss-120b-instruct --port 8001")
        return

    topics = {
        "Syllogisms": args.per_topic,
        "Seating Arrangements": args.per_topic,
        "Blood Relations": args.per_topic,
        "Alphanumeric Series": args.per_topic
    }

    all_questions = []

    for topic, count in topics.items():
        qs = generate_batch_questions(topic, count)
        all_questions.extend(qs)
        print(f"[OK] Generated {len(qs)}/{count} {topic}")

    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save raw questions
    with open(args.output, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n[OK] Total saved: {len(all_questions)} questions to {args.output}")
    print(f"   Success rate: {len(all_questions) / (args.per_topic * len(topics)) * 100:.1f}%")

if __name__ == "__main__":
    main()
