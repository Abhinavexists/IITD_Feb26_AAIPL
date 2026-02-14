# Setup Guide for AAIPL Competition

## System Requirements

- **Hardware**: AMD MI300X GPU (192GB HBM3) or equivalent
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.10+
- **VRAM**: Minimum 192GB for full tensor parallelism

## Installation Steps

### 1. Create Virtual Environment (uv venv recommended)

```bash
# Using uv (faster)
uv venv venv
source venv/bin/activate

# Or using conda
conda create -n aaipl python=3.11
conda activate aaipl
```

### 2. Install Core Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install other dependencies
pip install -r requirements.txt
```

### 3. Install Unsloth (for optimized training)

```bash
# Clone and install Unsloth
git clone https://github.com/unslothai/unsloth.git
cd unsloth
pip install -e .
cd ..
```

### 4. Login to HuggingFace

```bash
# For downloading models
huggingface-cli login
```

### 5. Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU
python -c "import torch; print(f'CUDA/ROCm available: {torch.cuda.is_available()}')"

# Check Unsloth
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"

# Check transformers
python -c "from transformers import AutoTokenizer; print('Transformers: OK')"
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'unsloth'`

**Solution:**
```bash
pip install unsloth
# Or install from GitHub:
pip install git+https://github.com/unslothai/unsloth.git
```

### Issue: `CUDA out of memory`

**Solution:**
- Reduce batch size in training scripts
- Reduce model sequence length
- Enable gradient checkpointing (already enabled)

### Issue: `vLLM server won't start`

**Solution:**
```bash
# Check GPU
rocm-smi

# Try with less parallelism
vllm serve gptopenai/gpt-oss-120b-instruct --port 8001 --tensor-parallel-size 1
```

### Issue: `Cannot find model on HuggingFace`

**Solution:**
```bash
# Verify login
huggingface-cli whoami

# Download manually
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir ./hf_models/Qwen2.5-14B-Instruct
```

## Quick Start

After installation, run:

```bash
# Terminal 1: Start vLLM
vllm serve gptopenai/gpt-oss-120b-instruct --port 8001

# Terminal 2: Generate data (wait 30s for vLLM to start)
python scripts/generate_questions_gpt_oss.py --per-topic 500
python scripts/generate_answers_gpt_oss.py

# Train models
python agents/train_question_model_qwen.py &
python agents/train_answer_model_qwen.py

# Test
python -m agents.question_agent --num_questions 10 --verbose
```

## Environment Variables

Optional optimizations:

```bash
# Enable Flash Attention
export FLASH_ATTENTION=1

# Set number of CUDA threads
export CUDA_LAUNCH_BLOCKING=1

# ROCm specific
export HIP_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

## File Structure After Setup

```
/project/
├── agents/
│   ├── question_model.py
│   ├── answer_model.py
│   ├── train_question_model_qwen.py
│   ├── train_answer_model_qwen.py
│   └── rl_train_answer.py
├── scripts/
│   ├── generate_questions_gpt_oss.py
│   ├── generate_answers_gpt_oss.py
│   └── validate_format.py
├── hf_models/  (will be created)
│   ├── qwen-2.5-14b-qagent-lora/
│   ├── qwen-2.5-14b-aagent-lora/
│   └── qwen-2.5-14b-aagent-rl/
├── data/final/  (will be created)
│   ├── questions_training.json
│   └── answers_training.json
├── requirements.txt
└── EXECUTION_GUIDE.md
```

## Next Steps

1. Verify all installations: `bash verify_setup.sh`
2. Run Phase 1 verification
3. Follow EXECUTION_GUIDE.md for phases 2-7
