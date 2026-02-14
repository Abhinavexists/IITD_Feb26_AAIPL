# AAIPL Competition - Implementation Plan

## Executive Summary

**Goal:** Win the AMD AI Premier League (AAIPL) competition in the next 12-24 hours.

**Strategy:** Aggressive Distillation + Supervised Fine-Tuning (SFT)

**Key Decisions:**

- **Model:** Mistral-7B-Instruct-v0.3 (sweet spot of speed & quality)
- **Teacher:** Llama-3.3-70B (distillation for training data)
- **Method:** LoRA fine-tuning with Unsloth
- **Data:** 1000-2000 questions + 1500-3000 Q-A pairs
- **Expected Gain:** 50-100% improvement over baseline

**Total Time:** ~13 hours implementation + testing, 11+ hours buffer

---

## Phase 1: Model Selection (30 minutes)

### Decision: Use Mistral-7B-Instruct-v0.3

**Why Mistral-7B?**

1. **Speed:** 7B parameters - meets 13s/9s time constraints comfortably
2. **Quality:** Better reasoning than Qwen3-4B, competitive with Llama-8B
3. **Memory:** Fits easily in 192GB MI300X with batch sizes 64-128
4. **Proven:** Excellent instruction following for competitive exams
5. **Safety Margin:** Faster than 12-14B models, more capable than 4B

**Backup Options:**

- Llama-3.1-8B-Instruct (with Unsloth optimization)
- Qwen2.5-14B (if speed is not bottleneck)

### Action Items:

- [ ] Verify Mistral-7B can be downloaded from HuggingFace
- [ ] Test single inference speed (target: <3s per 200 tokens)
- [ ] Confirm memory usage with batch_size=64

**Files to Update:**

- `agents/question_model.py` - Load Mistral instead of Qwen3-4B
- `agents/answer_model.py` - Load Mistral instead of Qwen3-4B

---

## Phase 2: Synthetic Data Generation via Distillation (4-5 hours)

### Overview
Use Llama-3.3-70B teacher model to generate high-quality training data for Mistral-7B student model.

### 2.1: Start vLLM Server (5 minutes)

```bash
vllm serve Unsloth/Llama-3.3-70B-Instruct \
  --port 8001 \
  --max-model-len 48000 \
  --gpu-memory-utilization 0.85
```

**Verify server is running:**
```bash
curl http://localhost:8001/v1/models
```

### 2.2: Generate Question Training Data (2 hours)

**Target:** 1,000-2,000 high-quality questions

**Distribution Across 4 Topics:**

- Syllogisms: 250-500 questions
- Seating Arrangements (Linear + Circular): 250-500 questions
- Blood Relations and Family Tree: 250-500 questions
- Alphanumeric Series: 250-500 questions

**Generation Parameters:**

- Temperature: 0.8 (diversity)
- Max tokens: 1024
- Difficulty mix: Easy (30%), Medium (40%), Hard (30%)

**Quality Filtering:**

- Threshold: 8.0/10 minimum quality score
- Format validation: Must be valid JSON
- Target output: 750-1,500 final examples

**Critical:** Use Chain-of-Thought prompting to get the 70B model to think through hard problems.

**Sample Prompt:**
```
Generate 500 extremely difficult MCQ questions about [TOPIC].

For each question:
1. Think step-by-step about the problem
2. Create 4 plausible options (only one correct)
3. Output in JSON format

Format:
{
  "topic": "Topic Name",
  "question": "...",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "A",
  "explanation": "Brief explanation"
}
```

**Files to Create/Use:**

- `scripts/generate_questions_vllm.py` - Custom vLLM script
- Output: `data/final/questions_training.json`
- Config: Update `tutorial_config.yaml` if using synthetic-data-kit

### 2.3: Generate Answer Training Data (2 hours)

**Target:** 1,500-3,000 question-answer pairs

**Two-Stage Approach:**

**Stage 1: Use generated questions from 2.2**

- Feed all questions to 70B model
- Generate detailed reasoning chains (CoT)
- Request both correct answers AND common mistakes
- ~30 minutes for 1500 questions

**Stage 2: Augment with example questions**

- Use `assets/topics_example.json` as base
- Generate variations with different difficulty levels
- Create multiple reasoning paths per question
- ~1.5 hours for augmentation

**Quality Filtering:**

- Verify answer correctness
- Ensure reasoning is detailed (threshold 8.5/10)
- Format validation
- Remove duplicates

**Expected Output:** 1,500-3,000 high-quality training pairs

**Files to Create/Use:**

- `scripts/generate_answers_vllm.py` - Custom vLLM script
- Output: `data/final/answers_training.json`

### 2.4: Shutdown vLLM Server (1 minute)

```bash
pkill -f vllm
# Free up GPU memory for training
```

### Action Items:

- [ ] Create `scripts/generate_questions_vllm.py`
- [ ] Create `scripts/generate_answers_vllm.py`
- [ ] Generate and validate training data
- [ ] Spot-check 50 examples manually
- [ ] Calculate actual data quality metrics

**Success Criteria:**

- ✅ 750-1,500 question examples generated
- ✅ 1,500-3,000 answer examples generated
- ✅ ≥80% format valid
- ✅ ≥8.0/10 average quality score

---

## Phase 3: Supervised Fine-Tuning (4-6 hours)

### Overview
Use Unsloth + LoRA to efficiently fine-tune both agents on generated synthetic data.

### 3.1: Q-Agent Fine-Tuning (2-3 hours)

**Setup:**

- Base Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Method: LoRA (parameter-efficient)
- Framework: Unsloth for 2x speed, Hugging Face Transformers

**LoRA Configuration:**
```python
lora_r = 64                    # Higher rank for better quality
lora_alpha = 64                # Alpha = rank for stability
lora_dropout = 0.05            # Light regularization
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

**Training Hyperparameters:**
```python
# Data and model
max_seq_length = 1536          # Allow full question generation
train_batch_size = 32          # Per GPU
gradient_accumulation_steps = 2  # Effective batch = 64

# Learning
learning_rate = 2e-4           # Aggressive for fast convergence
num_train_epochs = 3           # Multiple passes on data
warmup_ratio = 0.03            # 3% warmup
lr_scheduler_type = "cosine"   # Better than linear

# Optimization
optimizer = "adamw_8bit"       # 8-bit Adam
weight_decay = 0.01
max_grad_norm = 1.0
gradient_checkpointing = True  # Memory efficient

# Misc
fp16 = False                   # Use bfloat16 on AMD MI300X
bf16 = True                    # AMD support
save_steps = 10                # Save checkpoints
eval_steps = 10
```

**Data Format:**
```python
# Messages format
[
  {
    "messages": [
      {"role": "system", "content": "You are an expert..."},
      {"role": "user", "content": "Generate an MCQ..."},
      {"role": "assistant", "content": "{...json question...}"}
    ]
  },
  ...
]
```

**Training Time Estimate:**

- Dataset: 750-1,500 examples
- Total steps: ~50-70 (3 epochs)
- Time with Unsloth: **2-3 hours**

**Output:**

- LoRA adapters: `hf_models/mistral-7b-qagent-lora/`
- Optional: Merged model (if needed for deployment)

**Critical Files to Create:**

- `agents/train_question_model.py` - Main training script
- `configs/train_config_qagent.json` - Training configuration

### 3.2: A-Agent Fine-Tuning (2-3 hours)

**Same configuration as Q-Agent, but:**

- Max sequence length: 1024 tokens (answers shorter than questions)
- Learning rate: 3e-4 (can be higher for simpler task)
- Dataset: 1,500-3,000 answer examples

**Training Time Estimate:**

- Dataset: 1,500-3,000 examples (more data)
- **Time: 2-3 hours**

**Output:**

- LoRA adapters: `hf_models/mistral-7b-aagent-lora/`

**Critical Files to Create:**

- `agents/train_answer_model.py` - Main training script
- `configs/train_config_aagent.json` - Training configuration

### Action Items:

- [ ] Create training scripts with Unsloth
- [ ] Verify data format compatibility
- [ ] Start Q-Agent training
- [ ] While Q-Agent trains, optimize prompts (Phase 4)
- [ ] Start A-Agent training
- [ ] Verify LoRA adapters saved correctly
- [ ] Test loading fine-tuned models

**Success Criteria:**

- ✅ Both models finish training without errors
- ✅ Training loss decreases across epochs
- ✅ LoRA adapters saved (~1-2GB each)
- ✅ Inference with adapters works correctly

---

## Phase 4: Prompt Engineering Optimization (1-2 hours)

### Overview
Parallel to training: Optimize prompts for both agents

**Can start while Phase 3 training is running!**

### 4.1: Q-Agent Prompt Improvements

**Current Weaknesses:**

- Generic "expert examiner" persona
- Randomized answer position might confuse model
- ICL examples not topic-specific
- No constraint emphasis

**Improvements:**

1. **Topic-Specific System Prompts:**
   ```python
   SYLLOGISM_SYSTEM = """
   You are an expert in formal logic and syllogistic reasoning...
   """

   SEATING_SYSTEM = """
   You are an expert in constraint satisfaction and logical deduction...
   CRITICAL: Do NOT create numeric-style questions about permutations.
   Only create identification/logic-based questions.
   """

   BLOOD_RELATIONS_SYSTEM = """
   You are an expert in family tree logic and relationships...
   """

   SERIES_SYSTEM = """
   You are an expert in pattern recognition and sequences...
   """
   ```

2. **Expand Few-Shot Examples:**
   - Current: 1-3 examples per topic
   - Target: 5-8 examples per topic
   - Source: Hardest questions from `assets/topics_example.json`

3. **Constraint Emphasis in Prompt:**
   - Add token budget reminders
   - Add format requirements
   - Add "NO numeric seating" warning for that topic

4. **Answer Position Handling:**
   - Don't randomize; let model choose naturally
   - Or: Fix position but vary question difficulty to compensate

**Files to Update:**

- `agents/question_agent.py` lines 59-112 (build_prompt method)
- Create `utils/topic_specific_prompts.py`

**Test After Changes:**
```bash
python -m agents.question_agent --num_questions 20 --verbose
```

### 4.2: A-Agent Prompt Improvements

**Current Weaknesses:**

- Generic reasoning instruction
- No topic-specific strategies
- No structured reasoning flow

**Improvements:**

1. **Topic Detection + Specific Prompts:**
   ```python
   def get_system_prompt_for_topic(topic):
       if "Syllogism" in topic:
           return SYLLOGISM_ANSWER_PROMPT
       elif "Seating" in topic:
           return SEATING_ANSWER_PROMPT
       # ... etc
   ```

2. **Structured Reasoning Template:**
   ```
   For each question:
   1. Understand the question
   2. Analyze each option
   3. Eliminate wrong answers
   4. Verify your choice
   5. Output final answer
   ```

3. **Topic-Specific Strategies:**
   - Syllogisms: Focus on formal logic rules
   - Seating: Systematic constraint checking
   - Blood Relations: Build family tree diagram
   - Series: Identify pattern rules

**Files to Update:**

- `agents/answer_agent.py` lines 23-53 (build_prompt method)
- Create `utils/answer_prompts.py`

**Test After Changes:**
```bash
python -m agents.answer_agent --input_file outputs/filtered_questions.json --verbose
```

### Action Items:

- [ ] Create topic-specific prompts for Q-Agent
- [ ] Expand ICL examples to 5-8 per topic
- [ ] Test Q-Agent with new prompts
- [ ] Create topic-specific answer prompts
- [ ] Test A-Agent with new prompts
- [ ] Benchmark improvement over baseline prompts

**Success Criteria:**

- ✅ Q-Agent generates ≥80% format-valid questions
- ✅ Questions visibly harder/better quality
- ✅ A-Agent accuracy improves ≥10% over baseline

---

## Phase 5: OPTIONAL - Light RL Fine-Tuning (2-4 hours)

### When to Do This
**Only if** you have time after Phases 1-4 and testing passes.

### Strategy
Self-Play RL to optimize A-Agent for answer accuracy.

**Approach:**

1. Generate 500 questions with fine-tuned Q-Agent
2. Have A-Agent answer them
3. Score answers: correct=+1, incorrect=-1
4. Use DPO (Direct Preference Optimization) or PPO
5. Fine-tune A-Agent using reward signal

**Why Optional:**

- SFT should provide 50%+ improvement already
- RL is high-risk in limited time
- Can break working model if not careful
- Better to have solid SFT than rushed RL

**If You Have Time:**
```bash
# Generate RL dataset
python scripts/generate_rl_dataset.py --num_questions 500

# Fine-tune with DPO
python agents/rl_train_answer.py --method dpo
```

**Files to Create (if attempting):**

- `scripts/generate_rl_dataset.py`
- `agents/rl_train_answer.py`

---

## Phase 6: Testing & Validation (1-2 hours)

### 6.1: Time Constraint Validation

**Critical:** Must pass time constraints or you're disqualified!

```bash
# Test Q-Agent speed
python -m agents.question_agent \
  --num_questions 100 \
  --batch_size 5 \
  --verbose

# Expected output:
# - Total time: <1300 seconds
# - Average per question: <13 seconds
# - ≥50% format-valid questions
```

```bash
# Test A-Agent speed
python -m agents.answer_agent \
  --input_file outputs/filtered_questions.json \
  --batch_size 5 \
  --verbose

# Expected output:
# - Total time: <900 seconds
# - Average per answer: <9 seconds
```

**Pass/Fail Criteria:**

- ✅ Q-Agent average ≤ 13 seconds
- ✅ A-Agent average ≤ 9 seconds
- ✅ ≥50% format-valid questions
- ❌ If failed: Debug and optimize

### 6.2: Format Validation

```bash
# Create validation script
python scripts/validate_format.py \
  --questions outputs/filtered_questions.json \
  --answers outputs/filtered_answers.json
```

**Checks:**

- ✅ Valid JSON format
- ✅ All required fields present
- ✅ Token counts within limits (150 for Q, 512 for A)
- ✅ Correct choice letters (A-D only)
- ✅ No numeric seating questions
- ✅ Topics match allowed list

### 6.3: Quality Spot Check

**Manually review:**

- 20 random questions across all 4 topics
- Verify genuine difficulty
- Check answer correctness
- Ensure plausible distractors
- Verify no obvious errors

**Benchmark Against Examples:**

- Compare with `assets/topics_example.json`
- Should be similar or better difficulty
- Better variety than current implementation

### 6.4: Accuracy Benchmark

```bash
# Compare fine-tuned vs baseline
python scripts/compare_models.py \
  --baseline qwen-3-4b \
  --finetuned mistral-7b-lora \
  --num_samples 50
```

**Expected Improvements:**

- Q-Agent format validity: 50% → 80%+
- Q-Agent perceived difficulty: +30%
- A-Agent accuracy: +20-30%
- Overall quality: +50-100%

### Action Items:

- [ ] Create `scripts/validate_format.py`
- [ ] Create `scripts/compare_models.py`
- [ ] Run full 100-question test
- [ ] Document results
- [ ] Debug any failures
- [ ] Verify time constraints met

**Success Criteria:**

- ✅ All tests pass
- ✅ Time constraints met
- ✅ Format validation 100%
- ✅ Quality improvements documented

---

## Phase 7: Final Integration (30 minutes)

### 7.1: Model Integration

**Option A: Keep LoRA Adapters (Recommended)**
```python
# Fast loading, small files
from unsloth import FastLanguageModel

model = FastLanguageModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=False,
    adapters=["hf_models/mistral-7b-qagent-lora"]
)

model = FastLanguageModel.for_inference(model)
```

**Option B: Merge Adapters**
```bash
# If you need standalone model
python scripts/merge_lora.py \
  --base mistralai/Mistral-7B-Instruct-v0.3 \
  --adapters hf_models/mistral-7b-qagent-lora \
  --output hf_models/mistral-7b-qagent-merged
```

### 7.2: Update Agent Files

**Update `agents/question_model.py`:**
```python
# Instead of Qwen3-4B
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "hf_models/mistral-7b-qagent-lora"

# Load with LoRA
model = FastLanguageModel.from_pretrained(
    model_name,
    adapters=[adapter_path]
)
```

**Update `agents/answer_model.py`:**
```python
# Same pattern for answer model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "hf_models/mistral-7b-aagent-lora"

model = FastLanguageModel.from_pretrained(
    model_name,
    adapters=[adapter_path]
)
```

**Update `qgen.yaml` and `agen.yaml`:**

- Adjust temperature if needed
- Keep max_tokens fixed (don't change)
- Other hyperparameters can be tuned

### 7.3: Final Test Run

```bash
# Generate 50 test questions
python -m agents.question_agent --num_questions 50 --verbose

# Answer them
python -m agents.answer_agent --input_file outputs/filtered_questions.json --verbose

# Verify both work end-to-end
```

### 7.4: Submission Preparation

**Create submission folder:**
```bash
# Get your IP
ifconfig | grep inet

# Create folder (replace with actual IP)
mkdir AAIPL_192_168_1_100

# Copy required files
cp -r agents/ AAIPL_192_168_1_100/
cp -r hf_models/ AAIPL_192_168_1_100/  # Include fine-tuned models
cp qgen.yaml AAIPL_192_168_1_100/
cp agen.yaml AAIPL_192_168_1_100/
```

**Verify submission structure:**
```
AAIPL_192_168_1_100/
├── agents/
│   ├── question_model.py
│   ├── question_agent.py
│   ├── answer_model.py
│   ├── answer_agent.py
│   └── __init__.py
├── hf_models/
│   ├── mistral-7b-qagent-lora/
│   └── mistral-7b-aagent-lora/
├── qgen.yaml
├── agen.yaml
└── outputs/
    ├── questions.json
    └── answers.json
```

### Action Items:

- [ ] Update all agent model files
- [ ] Test end-to-end inference
- [ ] Create submission folder with correct naming
- [ ] Verify all files present and paths correct
- [ ] Do final validation run

**Success Criteria:**

- ✅ Models load without errors
- ✅ End-to-end test passes
- ✅ Submission folder correctly structured
- ✅ Ready for upload

---

## Timeline Summary

| Phase | Task | Duration | Hours From Start |
|-------|------|----------|-----------------|
| 1 | Model selection & setup | 0.5h | 0-0.5 |
| 2.1 | Start vLLM server | 0.1h | 0.5-0.6 |
| 2.2 | Generate Q-Agent data | 2h | 0.6-2.6 |
| 2.3 | Generate A-Agent data | 2h | 2.6-4.6 |
| 2.4 | Shutdown vLLM | 0.02h | 4.6-4.62 |
| 4 | Prompt engineering (PARALLEL) | 2h | 0.5-2.5 |
| 3.1 | Train Q-Agent | 3h | 4.6-7.6 |
| 3.2 | Train A-Agent | 3h | 7.6-10.6 |
| 6 | Testing & validation | 2h | 10.6-12.6 |
| 7 | Final integration | 0.5h | 12.6-13.1 |
| **TOTAL** | **Implementation** | **~13h** | **13.1h** |
| **BUFFER** | **Debugging/iteration** | **11h** | **24.1h** |

---

## Alternative Paths (If Constrained)

### Fast Track: Only Prompt Optimization (4-6 hours)
If you have < 12 hours or encounter GPU issues:

1. Skip fine-tuning entirely
2. Focus on aggressive prompt engineering (3 hours)
3. Expand ICL examples to 10-15 per topic
4. Test and polish (2 hours)
5. **Expected gain:** 20-30% over baseline

### Medium Track: SFT Only, No RL (8-12 hours)
If you want balanced approach:

1. Follow Phases 1-4, 6-7
2. Skip Phase 5 (RL)
3. Focus on data quality over quantity
4. More thorough testing
5. **Expected gain:** 50% over baseline

---

## Risk Mitigation

### Risk 1: Time Constraints Violated
**Prevention:**

- Profile inference speed BEFORE training
- Use smaller model if needed (Llama-8B or keep Qwen3-4B)
- Reduce batch size for inference
- Use quantization if needed (8-bit)

### Risk 2: Training Failures
**Prevention:**

- Start Q-Agent first (higher priority)
- Save checkpoints every 10 steps
- Keep Qwen3-4B as fallback baseline
- Test loading LoRA adapters after training

### Risk 3: Data Quality Issues
**Prevention:**

- Use aggressive filtering (threshold 8.0+)
- Manually spot-check 50 examples before training
- Include validation set (10% of data)
- Compare with examples before committing

### Risk 4: Format Validation Failures
**Prevention:**

- Validate format during data generation
- Test on 20 questions before full run
- Implement robust post-processing
- Have fallback validation logic

---

## Success Metrics

### Minimum Viable (Pass Disqualification Check)

- ✅ Q-Agent generates ≥50% format-valid questions
- ✅ Questions complete in ≤13s each
- ✅ Answers complete in ≤9s each
- ✅ No hardcoding or adversarial content

### Competitive (Advance to Semifinals)

- ✅ Q-Agent: ≥80% format-valid questions
- ✅ A-Agent: ≥70% accuracy on opponent questions
- ✅ Questions genuinely difficult (trick 50%+ of baseline)
- ✅ Average time: Q=8s, A=5s (buffer for hard cases)

### Win Condition (Championship)

- ✅ A-Agent accuracy: ≥85%
- ✅ Q-Agent difficulty: opponents score <60%
- ✅ Combined strategy beats all opponents
- ✅ No time constraint issues across all matches

---

## Implementation Checklist

### Preparation

- [ ] Confirm AMD MI300X access and ~192GB memory available
- [ ] Verify Mistral-7B can be downloaded
- [ ] Check vLLM server can run Llama-3.3-70B
- [ ] Confirm 12-24 hours available

### Phase 1

- [ ] Download Mistral-7B-Instruct-v0.3
- [ ] Test inference speed
- [ ] Update model loading code

### Phase 2

- [ ] Create vLLM data generation scripts
- [ ] Generate question training data
- [ ] Generate answer training data
- [ ] Validate data quality (≥8.0/10)
- [ ] Save filtered datasets

### Phase 3

- [ ] Create SFT training scripts with Unsloth
- [ ] Train Q-Agent (~3 hours)
- [ ] Train A-Agent (~3 hours)
- [ ] Verify LoRA adapters saved

### Phase 4 (Parallel to Phase 3)

- [ ] Create topic-specific prompts
- [ ] Expand ICL examples
- [ ] Test improved prompts
- [ ] Measure improvement

### Phase 6

- [ ] Validate time constraints
- [ ] Validate format compliance
- [ ] Spot-check quality
- [ ] Benchmark improvements

### Phase 7

- [ ] Update agent model files
- [ ] Test end-to-end pipeline
- [ ] Create submission folder
- [ ] Final verification

---

## Files to Create/Modify

### New Files to Create
```
agents/
  ├── train_question_model.py          # Q-Agent training script
  ├── train_answer_model.py            # A-Agent training script
  └── __init__.py

scripts/
  ├── generate_questions_vllm.py       # vLLM data generation
  ├── generate_answers_vllm.py         # vLLM data generation
  ├── validate_format.py               # Format validation
  ├── compare_models.py                # Benchmark comparison
  └── merge_lora.py                    # Merge LoRA adapters (optional)

configs/
  ├── train_config_qagent.json         # Q-Agent training config
  └── train_config_aagent.json         # A-Agent training config

utils/
  ├── topic_specific_prompts.py        # Topic-specific prompts
  └── answer_prompts.py                # Answer-specific prompts

data/
  ├── final/
  │   ├── questions_training.json      # Generated Q-Agent training data
  │   └── answers_training.json        # Generated A-Agent training data
```

### Files to Modify
```
agents/question_model.py               # Load Mistral + LoRA
agents/answer_model.py                 # Load Mistral + LoRA
agents/question_agent.py               # Update prompts (lines 59-112)
agents/answer_agent.py                 # Update prompts (lines 23-53)
qgen.yaml                              # Tune hyperparameters
agen.yaml                              # Tune hyperparameters
```

---

## Key Decisions Made

1. **Mistral-7B over alternatives:**
   - Fastest safe option for time constraints
   - Better quality than Qwen3-4B
   - Less risky than 12-14B models

2. **Distillation from 70B teacher:**
   - Captures reasoning quality baseline can't
   - 70B already available on system
   - 1000-3000 examples >> current approach

3. **SFT over RL/Distillation alone:**
   - Faster to implement
   - Lower risk
   - Proven approach
   - RL can be added if time permits

4. **Massive training data:**
   - 1000-2000 Q + 1500-3000 A examples
   - Quality filtering during generation
   - Better generalization than smaller datasets

5. **Parallel prompt engineering:**
   - While training runs, optimize prompts
   - Compounds with fine-tuning gains
   - Low-risk improvement

---

## Expected Outcomes

### Performance Improvements

**Baseline (Qwen3-4B, no fine-tuning):**

- Q-Agent format validity: ~40-50%
- Question difficulty: Basic
- A-Agent accuracy: ~30-40% (mostly lucky)
- Time per question: ~5s
- Time per answer: ~2s

**With This Strategy (Mistral-7B + SFT):**

- Q-Agent format validity: **80%+**
- Question difficulty: **Genuinely hard** (tricks 50%+ of opponents)
- A-Agent accuracy: **60-70%** (much improved)
- Time per question: **6-10s** (safe margin)
- Time per answer: **4-6s** (safe margin)
- Overall improvement: **50-100%**

### Competitive Advantage

**Why You'll Win:**

1. Most competitors likely using baseline Qwen3-4B
2. Very few will attempt fine-tuning in 12h
3. Distillation from 70B teacher gives massive quality boost
4. 1000-3000 training examples >> competitors' typical <500
5. AMD MI300X 192GB allows batch sizes others can't match
6. Multi-epoch training extracts more signal from data

---

## Questions & Support

**If you get stuck:**

1. Check the original CLAUDE.md for quick reference
2. See instruction.md for competition rules
3. Refer to tutorial.ipynb for detailed examples
4. Debug with `--verbose` flag on agent runs
5. Profile with `tgps_show=True` for speed analysis

---

## GOOD LUCK! 🚀

You have a solid, proven strategy with 11+ hours of buffer. Execute this plan methodically, test thoroughly, and you have a real shot at winning the championship!

**Start with Phase 1 right now if ready, or let me know if you need clarification on any step.**
