# AAIPL Competition - Final Implementation Plan
## SFT + RL Hybrid Strategy with Qwen2.5-14B & GPT-OSS-120B

**Status:** Ready for execution
**Timeline:** 9-11 hours total
**Buffer:** 13-15 hours for debugging/iteration
**Expected Gain:** 60-120% improvement over baseline

---

## Executive Summary

| Component | Selection | Rationale |
|-----------|-----------|-----------|
| **Teacher Model** | GPT-OSS-120B (vLLM) | 120B reasoning >> 70B, 4-6x faster with tensor-parallel |
| **Final Agents** | Qwen2.5-14B-Instruct | Whitelisted [OK], superior reasoning, meets time constraints [OK] |
| **Training Method** | SFT + RL (Hybrid) | Warm-start (SFT) + Direct optimization (RL) = Stable + Effective |
| **Data Gen Time** | 1.5-2 hours | vs 4-5 hours with Llama (4-6x speedup) |
| **Training Time** | 6-7 hours | SFT: 2-3 hrs, RL: 4-5 hrs |
| **Total Time** | ~9-11 hours | 3-4 hours saved vs pure distillation |
| **Expected Performance** | 60-120% gain | Better data (120B) + Better model (14B) + RL optimization |

---

## Phase 1: Model Verification (30 minutes)

### Pre-Flight Checks

```bash
# 1. Verify GPT-OSS-120B available
huggingface-cli model-info gptopenai/gpt-oss-120b-instruct

# 2. Verify Qwen2.5-14B available
huggingface-cli model-info Qwen/Qwen2.5-14B-Instruct

# 3. Check GPU capacity
nvidia-smi  # or rocm-smi for AMD
# Expected: ~192GB HBM3 available on MI300X
```

### Action Items

- [ ] Both models downloadable
- [ ] vLLM compatible with GPT-OSS-120B
- [ ] Qwen2.5-14B inference speed test <3s per 300 tokens

---

## Phase 2: Fast Data Generation via vLLM (1.5-2 hours)

### 2.1: Start vLLM Server with Tensor Parallelism (5-10 minutes)

```bash
# Start GPT-OSS-120B with tensor parallelism (2 GPUs)
vllm serve gptopenai/gpt-oss-120b-instruct \
  --port 8001 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2 \
  --enable-prefix-caching \
  --dtype bfloat16 \
  --max-num-seqs 32

# Verify startup (in another terminal)
sleep 30
curl http://localhost:8001/v1/models
```

**Performance Target:**
- Tokens/sec: 400-600 (with tensor parallelism)
- Memory: Distributed across 2 GPUs
- Latency: ~1-2ms per token

### 2.2: Generate Questions (45-60 minutes)

**Target:** 1,500-2,500 questions across 4 topics

**File:** `scripts/generate_questions_gpt_oss.py`

```python
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

VLLM_BASE_URL = "http://localhost:8001/v1"

def generate_batch_questions(topic: str, num_questions: int = 500) -> List[Dict]:
    """Generate questions in batches using vLLM"""

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

    for i in tqdm(range(num_questions), desc=f"Generating {topic}"):
        try:
            response = requests.post(
                f"{VLLM_BASE_URL}/completions",
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
                content = response.json()["choices"][0]["message"]["content"]

                # Parse JSON
                try:
                    question = json.loads(content)
                    # Validate required fields
                    if all(k in question for k in ["topic", "question", "choices", "answer", "explanation"]):
                        questions.append(question)
                except json.JSONDecodeError:
                    pass  # Skip malformed JSON
        except Exception as e:
            pass  # Skip failed requests

    return questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/final/questions_training.json")
    parser.add_argument("--per-topic", type=int, default=500)
    args = parser.parse_args()

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

    # Save raw questions
    with open(args.output, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n[OK] Total saved: {len(all_questions)} questions to {args.output}")

if __name__ == "__main__":
    main()
```

**Expected Results:**
- Total generated: ~2000 questions
- Valid format: ~1600-1800 (80-90%)
- Time: 45-60 minutes

### 2.3: Generate Answers (30-45 minutes)

**File:** `scripts/generate_answers_gpt_oss.py`

```python
#!/usr/bin/env python3
"""
Phase 2.3: Generate answers for questions using GPT-OSS-120B via vLLM
"""

import requests
import json
import argparse
from tqdm import tqdm
from typing import List, Dict

VLLM_BASE_URL = "http://localhost:8001/v1"

def generate_answers_for_questions(questions_file: str, output_file: str) -> List[Dict]:
    """Generate answers using GPT-OSS-120B"""

    with open(questions_file) as f:
        questions = json.load(f)

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

    for q in tqdm(questions, desc="Generating answers"):
        try:
            prompt = f"""Question: {q['question']}

Options:
{chr(10).join(q['choices'])}

Provide your answer and reasoning."""

            response = requests.post(
                f"{VLLM_BASE_URL}/completions",
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
                content = response.json()["choices"][0]["message"]["content"]

                try:
                    answer = json.loads(content)
                    if "answer" in answer and "reasoning" in answer:
                        answers.append({
                            "question_id": questions.index(q),
                            **answer
                        })
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            pass

    with open(output_file, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"\n[OK] Generated {len(answers)} valid answers to {output_file}")
    return answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final/questions_training.json")
    parser.add_argument("--output", default="data/final/answers_training.json")
    args = parser.parse_args()

    generate_answers_for_questions(args.input, args.output)

if __name__ == "__main__":
    main()
```

**Expected Results:**
- Answers generated: ~1600-1800
- Valid format: ~1400-1600 (85-90%)
- Time: 30-45 minutes

### 2.4: Shutdown vLLM (1 minute)

```bash
pkill -f vllm

# Verify freed memory
nvidia-smi  # or rocm-smi
```

---

## Phase 3: Supervised Fine-Tuning (SFT) - Warm Start (2-3 hours)

**Goal:** Warm-start Qwen2.5-14B with 3 epochs of SFT training on generated data

### 3.1: Q-Agent Fine-Tuning

**File:** `agents/train_question_model_qwen.py`

```python
#!/usr/bin/env python3
"""
Phase 3.1: SFT training for Q-Agent (Qwen2.5-14B) using Unsloth
"""

import json
import torch
from unsloth import FastLanguageModel, get_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def format_prompts(examples):
    """Format questions for SFT training"""
    formatted = []
    for q in examples['questions']:
        text = f"""### Topic
{q['topic']}

### Question
{q['question']}

### Choices
{chr(10).join(q['choices'])}

### Answer
{q['answer']}

### Explanation
{q['explanation']}"""
        formatted.append(text)
    return {'text': formatted}

def main():
    # Load model with Unsloth (2x faster, uses gradient checkpointing)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # Use full precision on MI300X
    )

    # Apply LoRA (Parameter-efficient fine-tuning)
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,  # Rank: Higher for larger model
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # Load training data
    with open("data/final/questions_training.json") as f:
        questions = json.load(f)

    dataset = Dataset.from_dict({
        "questions": questions
    }).map(
        lambda examples: {'text': [format_prompts({'questions': [q]})['text'][0] for q in examples['questions']]},
        batched=True,
        remove_columns=['questions']
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir="hf_models/qwen-2.5-14b-qagent-lora",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,  # Effective batch size: 32
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        bf16=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=2048,
    )

    trainer.train()

    # Save adapters
    model.save_pretrained("hf_models/qwen-2.5-14b-qagent-lora")
    tokenizer.save_pretrained("hf_models/qwen-2.5-14b-qagent-lora")
    print("[OK] Q-Agent training complete!")

if __name__ == "__main__":
    main()
```

**Configuration:**
- Rank: 128 (larger model = larger rank)
- Batch size: 32 (effective)
- Epochs: 3
- Learning rate: 2e-4
- **Time: 2-3 hours**

### 3.2: A-Agent Fine-Tuning

**File:** `agents/train_answer_model_qwen.py`

Similar to Q-Agent, but:
- Max sequence length: 1024 (answers shorter)
- Learning rate: 3e-4 (can be higher)
- Dataset: `data/final/answers_training.json`
- Output: `hf_models/qwen-2.5-14b-aagent-lora`
- **Time: 2-3 hours**

---

## Phase 4: Reinforcement Learning Optimization (4-5 hours)

**Goal:** Use RL to directly optimize for win metrics (answer accuracy)

### 4.1: Self-Play Loop

**File:** `agents/rl_train_answer.py`

**Strategy:**
1. Generate 500-1000 questions with fine-tuned Q-Agent
2. Have A-Agent answer them
3. Score answers (correct = +1, incorrect = -1)
4. Use DPO (Direct Preference Optimization) to maximize accuracy

```bash
# Run RL training
python agents/rl_train_answer.py \
  --model_name "hf_models/qwen-2.5-14b-aagent-lora" \
  --num_questions 500 \
  --output_dir "hf_models/qwen-2.5-14b-aagent-rl"

# Expected: 1-2% improvement per epoch for 3-4 epochs
```

**Expected Results:**
- A-Agent accuracy improvement: 3-8%
- Time: 4-5 hours
- Total gain: Combined SFT (50-70% gain) + RL (3-8% incremental)

---

## Phase 5: Testing & Validation (1-2 hours)

### Critical Tests

#### 5.1: Time Constraint Validation
```bash
# Test 100 questions to verify <1300s total
python -m agents.question_agent --num_questions 100 --batch_size 5 --verbose

# Test 100 answers to verify <900s total
python -m agents.answer_agent --input_file outputs/filtered_questions.json --batch_size 5 --verbose
```

**Pass Criteria:**
- Q-Agent: Average <13s per question
- A-Agent: Average <9s per answer
- ≥50% format-valid questions

#### 5.2: Format Validation
```bash
python scripts/validate_format.py \
  --questions outputs/filtered_questions.json \
  --answers outputs/filtered_answers.json
```

#### 5.3: Quality Spot Check
- Manually review 20 random questions
- Verify difficulty level (genuinely hard)
- Check answer correctness
- Ensure no adversarial edge cases

---

## Phase 6: Update Agent Model Loaders (30 minutes)

### 6.1: Update Question Model Loader

**File:** `agents/question_model.py`

```python
# Change from: "Qwen/Qwen3-4B"
# To:
import torch
from unsloth import FastLanguageModel

class QAgent:
    def __init__(self, **kwargs):
        model_name = "Qwen/Qwen2.5-14B-Instruct"
        adapter_path = "hf_models/qwen-2.5-14b-qagent-lora"

        # Load with LoRA adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=False,
            adapters=[adapter_path],
        )

        # Prepare for inference
        self.model = FastLanguageModel.for_inference(model)
        self.tokenizer = tokenizer
```

### 6.2: Update Answer Model Loader

**File:** `agents/answer_model.py`

Same pattern, but with:
- `adapter_path = "hf_models/qwen-2.5-14b-aagent-rl"`

---

## Phase 7: Final Integration & Submission (30 minutes)

```bash
# 1. Final test run
python -m agents.question_agent --num_questions 50
python -m agents.answer_agent --input_file outputs/filtered_questions.json

# 2. Create submission package
mkdir AAIPL_your_team_name
cp -r agents/ AAIPL_your_team_name/
cp -r hf_models/qwen-2.5-14b-*-lora/ AAIPL_your_team_name/hf_models/
cp -r hf_models/qwen-2.5-14b-*-rl/ AAIPL_your_team_name/hf_models/
cp qgen.yaml AAIPL_your_team_name/
cp agen.yaml AAIPL_your_team_name/

# 3. Verify submission
ls -la AAIPL_your_team_name/
```

---

## Timeline Breakdown (9-11 hours)

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1 | Model verification | 30 min | Pre-flight checks |
| 2.1 | Start vLLM | 5 min | Tensor parallel setup |
| 2.2 | Generate questions | 45-60 min | 1500-2500 questions |
| 2.3 | Generate answers | 30-45 min | 1500-2500 answers |
| 2.4 | Shutdown vLLM | 1 min | Free GPU memory |
| 3.1 | Train Q-Agent | 2-3 hours | SFT warm-start |
| 3.2 | Train A-Agent | 2-3 hours | SFT warm-start |
| 4.1 | RL optimization | 4-5 hours | DPO for A-Agent (optional: can run in parallel with Phase 3) |
| 5 | Testing | 1-2 hours | Validation & spot checks |
| 6 | Update loaders | 30 min | Integrate fine-tuned models |
| 7 | Final integration | 30 min | Submission prep |
| **TOTAL** | | **~9-11h** | Buffer: 13-15 hours |

---

## Success Criteria

[OK] **Q-Agent:**
- ≥80% format-valid questions
- Average generation: <8 seconds (vs 13s limit)
- Difficulty: Genuinely hard (tricks 50%+ of opponents)

[OK] **A-Agent:**
- ≥70% accuracy on generated questions (after SFT)
- ≥78%+ accuracy after RL (target)
- Average generation: <6 seconds (vs 9s limit)
- Reasoning quality: Detailed and coherent

[OK] **Competition:**
- Advance past elimination round
- Win ≥60% of matches
- No time constraint violations

---

## Alternative Paths (If Time-Constrained)

### Fast Track: SFT Only (6-8 hours)
- Skip Phase 4 (RL)
- Expected gain: 50-70%
- Lower risk than rushing RL

### Ultra-Fast: SFT + Light RL (7-9 hours)
- Run RL on smaller dataset (100-200 questions)
- 2 epochs max instead of 3-4
- Still get 3-5% RL improvement

---

## Files to Create

```
scripts/
  ├── generate_questions_gpt_oss.py
  ├── generate_answers_gpt_oss.py
  └── validate_format.py

agents/
  ├── train_question_model_qwen.py (NEW)
  ├── train_answer_model_qwen.py (NEW)
  ├── rl_train_answer.py (NEW - optional)
  ├── question_model.py (UPDATE)
  └── answer_model.py (UPDATE)

data/final/
  ├── questions_training.json (GENERATED)
  └── answers_training.json (GENERATED)

hf_models/
  ├── qwen-2.5-14b-qagent-lora/ (GENERATED)
  ├── qwen-2.5-14b-aagent-lora/ (GENERATED)
  └── qwen-2.5-14b-aagent-rl/ (GENERATED - optional)
```

---

## Why This Strategy Wins

1. **GPT-OSS-120B Teacher:** 120B reasoning >> 70B, 4-6x faster with tensor parallelism
2. **Qwen2.5-14B Agent:** Better reasoning than Mistral-7B, still meets time constraints
3. **Hybrid SFT + RL:** Warm-start stability + direct optimization
4. **Massive Training Data:** 1500-2500 examples vs competitors' likely <500
5. **Multi-Epoch Training:** 3 epochs extracts maximum performance
6. **AMD MI300X Advantage:** 192GB allows batch size 32+ = fast training
7. **Proven Techniques:** SFT + RL = battle-tested combination

**Expected Gain:** 60-120% improvement over baseline

---

## Quick Start

```bash
# Terminal 1: Start vLLM server
vllm serve gptopenai/gpt-oss-120b-instruct --port 8001 \
  --tensor-parallel-size 2 --max-model-len 32768

# Terminal 2 (wait 30s): Generate data
python scripts/generate_questions_gpt_oss.py --per-topic 500
python scripts/generate_answers_gpt_oss.py

# Terminal 1: Kill vLLM
pkill -f vllm

# Terminal 2: Train models
python agents/train_question_model_qwen.py &
python agents/train_answer_model_qwen.py

# (Optional) While training: RL optimization
python agents/rl_train_answer.py --num_questions 500

# After training: Test
python -m agents.question_agent --num_questions 100 --verbose
python -m agents.answer_agent --input_file outputs/filtered_questions.json

# Submit
mkdir AAIPL_submission
cp -r agents hf_models qgen.yaml agen.yaml AAIPL_submission/
```

---

**Ready to execute!** 

Next step: Execute Phase 1-2 now.
