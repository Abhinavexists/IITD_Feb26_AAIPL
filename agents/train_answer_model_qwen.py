#!/usr/bin/env python3
"""
Phase 3.2: SFT training for A-Agent (Qwen2.5-14B) using Unsloth
Warm-start with 3 epochs of supervised fine-tuning on generated answers
"""

import json
import torch
import os
from typing import List, Dict

try:
    from unsloth import FastLanguageModel, get_peft_model_state_dict
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("[ERROR] ERROR: Unsloth not installed!")
    print("   Please install: pip install unsloth")
    UNSLOTH_AVAILABLE = False

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

def format_prompts(examples):
    """Format answers for SFT training"""
    formatted = []
    for q, a in zip(examples['questions'], examples['answers']):
        text = f"""### Question
{q['question']}

### Choices
{chr(10).join(q['choices'])}

### Answer
{a['answer']}

### Reasoning
{a['reasoning']}"""
        formatted.append(text)
    return {'text': formatted}

def main():
    if not UNSLOTH_AVAILABLE:
        print("[ERROR] Cannot proceed without Unsloth")
        return

    print(">> Phase 3.2: Starting A-Agent SFT Training\n")

    # Load model with Unsloth
    print("[*] Loading Qwen2.5-14B-Instruct with Unsloth...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-14B-Instruct",
            max_seq_length=1024,  # Shorter for answers
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print("[OK] Model loaded!\n")

    # Apply LoRA
    print("[*] Applying LoRA adapters...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=128,
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
    except Exception as e:
        print(f"[ERROR] Failed to apply LoRA: {e}")
        return

    print("[OK] LoRA applied!\n")

    # Load training data
    print("[*] Loading training data...")
    questions_path = "data/final/questions_training.json"
    answers_path = "data/final/answers_training.json"

    if not os.path.exists(questions_path) or not os.path.exists(answers_path):
        print(f"[ERROR] Training data not found")
        print(f"   Questions: {os.path.exists(questions_path)}")
        print(f"   Answers: {os.path.exists(answers_path)}")
        print("   Please run Phase 2 (data generation) first")
        return

    with open(questions_path) as f:
        questions = json.load(f)
    with open(answers_path) as f:
        answers_data = json.load(f)

    print(f"[OK] Loaded {len(questions)} questions and {len(answers_data)} answers\n")

    # Pair questions with answers by ID
    answers_by_id = {}
    for i, a in enumerate(answers_data):
        q_id = a.get('question_id', i)
        answers_by_id[q_id] = a

    paired_data = {
        'questions': [],
        'answers': []
    }

    for idx, q in enumerate(questions):
        if idx in answers_by_id:
            paired_data['questions'].append(q)
            paired_data['answers'].append(answers_by_id[idx])

    print(f"[OK] Paired {len(paired_data['questions'])} question-answer pairs\n")

    dataset = Dataset.from_dict(paired_data).map(
        lambda examples: {'text': [format_prompts({'questions': [q], 'answers': [a]})['text'][0]
                                   for q, a in zip(examples['questions'], examples['answers'])]},
        batched=True,
        remove_columns=['questions', 'answers']
    )

    # Training configuration
    print("[*]  Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="hf_models/qwen-2.5-14b-aagent-lora",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,  # Effective batch size: 32
        learning_rate=3e-4,  # Slightly higher than Q-Agent
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        bf16=True,
        remove_unused_columns=False,
    )

    # Train
    print("[*] Starting training (3 epochs)...\n")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=1024,
    )

    trainer.train()

    # Save adapters
    print("\n[*] Saving model adapters...")
    model.save_pretrained("hf_models/qwen-2.5-14b-aagent-lora")
    tokenizer.save_pretrained("hf_models/qwen-2.5-14b-aagent-lora")
    print("[OK] A-Agent training complete!")
    print("   Saved to: hf_models/qwen-2.5-14b-aagent-lora\n")

if __name__ == "__main__":
    main()
