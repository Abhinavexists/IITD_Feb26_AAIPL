#!/usr/bin/env python3
"""
Phase 3.1: SFT training for Q-Agent (Qwen2.5-14B) using Unsloth
Warm-start with 3 epochs of supervised fine-tuning on generated questions
"""

import json
import torch
import os

from typing import List, Dict

os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

# Set offline mode to avoid trying to access HuggingFace hub
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer,SFTConfig

from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType


system_prompt="""You are an expert question generator for competitive entrance exams. Your task is to create challenging, well-structured multiple-choice questions (MCQs) that test logical reasoning and analytical thinking.

CRITICAL RESTRICTIONS:
- Generate ONLY questions on the topic listed above
- If the topic is about For Seating Arrangements: DO NOT generate numeric-style questions like "How many permutations..." - only logic/identification questions
- All questions must be in English
- Do NOT hardcode or repeat questions

OUTPUT FORMAT REQUIREMENTS:
You must always output in the following JSON format with these exact fields:

{
    "topic": "<Topic of the Question>",
    "question": "<full question text>",
    "choices": [
        "A) <choice A text>",
        "B) <choice B text>",
        "C) <choice C text>",
        "D) <choice D text>"
    ],
    "answer": "<correct choice letter only>",
    "explanation": "brief explanation within 100 words for why the answer is correct"
}

FORMAT VALIDATION RULES:
- Exactly 4 choices labeled A) through D)
- Answer must be exactly one letter (A, B, C, or D)
- Combined tokens for topic, question, choices, and answer: ≤ 150 tokens
- Explanation tokens: ≤ 874 tokens
- Total tokens: ≤ 1024 tokens
- Valid JSON format is MANDATORY"""

# def format_prompts(examples):
#     """Format questions for SFT training"""
#     formatted = []
#     for q in examples['questions']:
#         text = f"""### Topic
# {q['topic']}

# ### Question
# {q['question']}

# ### Choices
# {chr(10).join(q['choices'])}

# ### Answer
# {q['answer']}

# ### Explanation
# {q['explanation']}"""
#         formatted.append(text)
#     return {'text': formatted}

def format_prompts(examples):
    """Format questions for SFT training"""
    formatted = []

        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TOPIC TO GENERATE ON: {examples['topic']}"},
        {"role": "assistant","content":str(examples)}
    ]

    examples['text'] = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
        #formatted.append(text)
    return examples


print(">> Phase 3.1: Starting Q-Agent SFT Training\n")

# Load model with Unsloth (2x faster, uses gradient checkpointing)
print("[*] Loading Qwen2.5-14B-Instruct with Unsloth...")

# Use cached model from HuggingFace hub format
model_path = "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

# Load tokenizer first
print("[*] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("LOADED TOKENIZER")

# Load model with transformers (not FastLanguageModel) to avoid hub access
print("[*] Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

# Apply LoRA adapters using PEFT
print("[*] Applying LoRA adapters...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
model = get_peft_model(base_model, lora_config)

print("[OK] Model loaded and optimized!\n")

# Load training data
print("[*] Loading training data...")
data_path = "questions_training.json"
if not os.path.exists(data_path):
    print(f"[ERROR] Training data not found at {data_path}")
    print("   Please run Phase 2 (data generation) first")
    #return

with open(data_path) as f:
    questions = json.load(f)

print(f"[OK] Loaded {len(questions)} questions\n")

# dataset = Dataset.from_dict({
#     "questions": questions
# }).map(
#     lambda examples: {'text': [format_prompts({'questions': [q]})['text'][0] for q in examples['questions']]},
#     batched=True,
#     remove_columns=['questions']
# )

dataset=Dataset.from_list(questions)
dataset = dataset.map(format_prompts)


# Training configuration
print("[*] Setting up training arguments...")
training_args = SFTConfig(
    output_dir="hf_models/qwen-2.5-14b-qagent-lora",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch size: 32
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    # save_steps=54,
    logging_steps=10,
    weight_decay=0.01,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    bf16=True,
    remove_unused_columns=False,
    max_length=1024,
    dataset_text_field ='text'
)

# Train
print("[*] Starting training (3 epochs)...\n")
trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    #dataset_text_field="text",
    #max_seq_length=1024,
)

trainer.train()

# Save adapters (original location)
print("\n[*] Saving model adapters...")
model.save_pretrained("hf_models/qwen-2.5-14b-aagent-lora")
tokenizer.save_pretrained("hf_models/qwen-2.5-14b-aagent-lora")
print("[OK] A-Agent training complete!")
print("   Saved to: hf_models/qwen-2.5-14b-aagent-lora\n")

# Save to additional backup location
# backup_path = "checkpoints/final/qwen-2.5-14b-aagent-lora"
# print(f"[*] Saving backup checkpoint to {backup_path}...")
# os.makedirs(backup_path, exist_ok=True)
# model.save_pretrained(backup_path)
# tokenizer.save_pretrained(backup_path)
# print(f"[OK] Backup saved to: {backup_path}\n")

# if __name__ == "__main__":
#     main()
