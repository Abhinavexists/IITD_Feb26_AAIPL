# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AMD AI Premier League (AAIPL) competition** at IIT Delhi for building AI agents that compete in generating and answering multiple-choice questions (MCQs) for competitive exams. The competition uses a cricket-style 1v1 format where teams compete with their Q-Agent (question generation) and A-Agent (answer answering) models.

The project is part of a tournament where team performance depends on:

- **Q-Agent Score**: How many questions the opponent's A-Agent gets wrong
- **A-Agent Score**: How many questions the opponent's Q-Agent the model answers correctly

## Architecture Overview

### Core Components

1. **Question Agent (Q-Agent)**: Generates difficult MCQs on specified topics
   - **Entry**: `python -m agents.question_agent`
   - **Model**: `agents/question_model.py` (implements actual generation logic)
   - **Runner**: `agents/question_agent.py` (orchestrates batch generation, filtering, format validation)
   - **Config**: `qgen.yaml` (max_tokens=1024, temperature=0.7, top_p=0.9)
   - **Output**: `outputs/questions.json` (raw) + `outputs/filtered_questions.json` (format-validated)
   - **Constraint**: Must generate ≥50% of questions in valid format to avoid disqualification

2. **Answer Agent (A-Agent)**: Answers MCQs from Q-Agent
   - **Entry**: `python -m agents.answer_agent`
   - **Model**: `agents/answer_model.py` (implements actual answering logic)
   - **Runner**: `agents/answer_agent.py` (processes questions in batches, validates answers)
   - **Config**: `agen.yaml` (max_tokens=512, temperature=0.1, top_p=0.9)
   - **Output**: `outputs/answers.json` (raw) + `outputs/filtered_answers.json` (format-validated)

3. **Utility Functions**: `utils/build_prompt.py`
   - `option_extractor_prompt()`: Extracts single letter answer from text
   - `auto_json()`: Fixes malformed JSON responses

### Base Model

Both agents use **Qwen3-4B** from HuggingFace:

- Loaded via `transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")`
- Can be fine-tuned using Unsloth (see `agents/question_model_llama.py` and `agents/answer_model_llama.py` for examples)
- Models must come from `/root/.cache/huggingface/hub` (read-only) or copied to `hf_models/`

## Common Development Commands

### Generate Questions

```bash
python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
```

### Generate Answers

```bash
python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
```

### Test Full Pipeline

```bash
python -m agents.question_agent --num_questions 10 --output_file outputs/questions.json --verbose && \
python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --verbose
```

## Key Output Formats

### Question Format (Required)

```json
{
    "topic": "Logical Reasoning/Syllogisms",
    "question": "Full question text?",
    "choices": ["A) Choice 1", "B) Choice 2", "C) Choice 3", "D) Choice 4"],
    "answer": "A",
    "explanation": "Brief explanation (≤100 words)"
}
```

### Answer Format (Required)

```json
{
    "answer": "A",
    "reasoning": "Brief reasoning (≤100 words)"
}
```

### Format Validation Rules

**Questions**:

- 4 choices labeled A) through D)
- Answer must be single letter (A, B, C, or D)
- **Combined tokens** (question + choices + answer): < 150 tokens (Qwen tokenizer)
- **Explanation tokens**: ≤ 874 tokens
- **Total**: ≤ 1024 tokens
- Valid JSON format required

**Answers**:

- Answer must be single letter (A, B, C, or D)
- Answer tokens < 50
- Total tokens (answer + reasoning) < 512

## Topics (AAIPL Competition)

Questions must be generated for these **FOUR** topics only:

1. **Syllogisms**: Logical reasoning based on premises and conclusions
2. **Seating Arrangements**:
   - Linear and Circular seating
   - **⚠️ NO numeric-style questions** (e.g., "how many permutations?")
3. **Blood Relations and Family Tree**: Relationship inference and family structure logic
4. **Mixed Series (Alphanumeric)**: Pattern recognition in letter/number sequences

See `assets/topics.json` for current topics and `assets/topics_example.json` for example questions.

## Restrictions & Important Notes

### Critical Restrictions

1. **No RAG (Retrieval Augmented Generation)**: All questions must be generated from model knowledge only
2. **No Adversarial Approaches**: Don't try to make opponent's A-Agent hallucinate or fail
3. **English Only**: Both Q-Agent and A-Agent must produce English-only content
4. **Disqualification Risks**:
   - Hardcoding questions/answers
   - Q-Agent generating <50% valid format questions
   - Using unauthorized models
   - Non-English content
   - Exceeding time limits

### Approved WhiteList Models

**You MUST use one of these for final submission:**

1. Qwen/Qwen3-4B
2. Qwen/Qwen2.5-14B-Instruct
3. Unsloth/Llama-3.1-8B-Instruct
4. mistralai/Mistral-7B-Instruct-v0.3
5. microsoft/Phi-4-mini-instruct
6. google/gemma-3-12b-it

**Important**: For training/distillation, you can use ANY teacher model, but final agents MUST use a whitelisted model.

### Seating Arrangement Restriction

**⚠️ DO NOT generate numeric-style seating arrangement questions** such as:

- "How many permutations of such arrangements are possible?"
- Questions asking for counting/enumeration of arrangements
- Any numeric calculation-based seating questions

Only generate identification/logic-based seating arrangement questions.

### Token & Time Limits (CRITICAL)

**Q-Agent:**

- **Combined tokens** for `topic`, `question`, `choices`, `answer`: ≤ 150 tokens
- **Explanation tokens**: ≤ 874 tokens (within 1024 total)
- **Time per question**: ≤ 13 seconds
- **Total for 100 questions**: ≤ 1300 seconds (~21 minutes)

**A-Agent:**

- **Time per answer**: ≤ 9 seconds
- **Total for 100 answers**: ≤ 900 seconds (~15 minutes)

**⚠️ If time limits are exceeded, your submission will be DISQUALIFIED.**

### Hyperparameter Configuration

- **Do NOT Modify** `max_tokens` in `agen.yaml` (512) and `qgen.yaml` (1024)
- Other hyperparameters can be tuned (temperature, top_p, etc.)

## Submission Requirements

**Folder Structure:**

```
AAIPL_your_IP/        # Replace 'your_IP' with your IPv4 (dots → underscores)
├── agents/
│   ├── question_model.py
│   ├── question_agent.py
│   ├── answer_model.py
│   ├── answer_agent.py
├── outputs/
│   ├── questions.json
│   ├── answers.json
└── [supporting files]
```

**Example**: If your IP is `192.168.1.100`, folder should be named `AAIPL_192_168_1_100`.

## Training & Model Improvements

See `tutorial.ipynb` for comprehensive guide on:

- Synthetic data generation using vLLM
- Using Unsloth for efficient fine-tuning
- Training strategies for logical reasoning
- Chain-of-thought (CoT) data generation

Example implementations:

- `agents/question_model_llama.py`: Fine-tuned Q-Agent using Llama + Unsloth
- `agents/answer_model_llama.py`: Fine-tuned A-Agent using Llama + Unsloth

Training timeline estimate: 6-8 hours per model with optimizations during training.

## Scoring Logic

For each match:

- **Team Q-Agent Score** = (# questions A-Agent gets wrong) / N × 100
- **Team A-Agent Score** = (# questions A-Agent gets correct) / N × 100

Where N = number of valid format questions generated by Q-Agent.

## Directory Structure

```
.
├── agents/
│   ├── question_model.py          # ← Modify this for Q-Agent improvements
│   ├── question_agent.py           # Runner (don't modify)
│   ├── answer_model.py             # ← Modify this for A-Agent improvements
│   ├── answer_agent.py             # Runner (don't modify)
│   ├── question_model_llama.py     # Example: fine-tuned Q-Agent
│   └── answer_model_llama.py       # Example: fine-tuned A-Agent
├── assets/
│   ├── topics.json                 # Topics for generation
│   ├── topics_example.json         # Example questions per topic
│   ├── sample_question.json        # Expected question format
│   └── sample_answer.json          # Expected answer format
├── outputs/
│   ├── questions.json              # Raw generated questions
│   ├── filtered_questions.json     # Format-validated questions
│   ├── answers.json                # Raw generated answers
│   └── filtered_answers.json       # Format-validated answers
├── utils/
│   └── build_prompt.py             # JSON/option extraction utilities
├── hf_models/                      # Copy models here for modification
├── qgen.yaml                       # Q-Agent hyperparameters
├── agen.yaml                       # A-Agent hyperparameters
└── tutorial.ipynb                  # Fine-tuning tutorial
```

## Primary Files to Modify

To improve agent performance, focus on:

1. **`agents/question_model.py`**: Enhance Q-Agent generation logic and prompts
2. **`agents/answer_model.py`**: Enhance A-Agent answering logic and prompts
3. **`qgen.yaml` / `agen.yaml`**: Tune hyperparameters (except max_tokens)
4. **Fine-tune models**: Copy to `hf_models/` and use Unsloth (see tutorial)

## Debugging & Common Issues

1. **Invalid JSON in output**: Check filtering logic in question_agent.py line 211-268
2. **Low answer accuracy**: Adjust system prompt and generation parameters in answer_agent.py line 23-32
3. **Format validation failures**: Ensure output matches expected format and token limits
4. **Token count issues**: Use `count_tokens_q()` and `count_tokens_a()` to debug
5. **Model loading errors**: Ensure model is in `/root/.cache/huggingface/hub` or copied to `hf_models/`
6. **Time limit exceeded**: Profile with `tgps_show=True` flag (see qgen.yaml/agen.yaml)

## Additional Context

- See `/Projects/AMD/instruction.md` for complete hackathon guidelines extracted from AAIPL presentation
- Tutorial: `tutorial_config.yaml` contains vLLM server config for synthetic data generation
- Competition documentation: `README.ipynb` contains detailed instructions and scoring examples
