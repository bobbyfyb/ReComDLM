<h1 align="center">ReComDLM</h1>

<p align="center">
Retrieval-augmented Commonsense Reasoning with Diffusion Language Models
</p>

## Overview

**ReComDLM** is a novel framework that integrates retrieval-augmented generation (RAG) with diffusion language models (DLMs) to enhance commonsense reasoning capabilities. This project provides training and inference code, along with datasets and evaluation scripts for commonsense reasoning benchmarks.

Built upon the [dLLM](https://github.com/ZHZisZZ/dllm) foundation, ReComDLM demonstrates how recommendation and retrieval systems can augment diffusion-based language models to achieve superior performance on complex reasoning tasks.

## Quick Start

### Prerequisites
- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Installation with uv

1. **Clone the repository**
   ```bash
   git clone https://github.com/bobbyfyb/ReComDLM.git
   cd ReComDLM
   ```

2. **Initialize Python environment with uv**
   ```bash
   # Create and sync the environment
   uv sync
   ```

3. **Install the package in development mode**
   ```bash
   # This is automatically handled by uv sync, but you can also explicitly run:
   uv pip install -e .
   ```

4. **(Optional) Install evaluation dependencies**
   ```bash
   uv sync --extra eval
   ```

5. **(Optional) Initialize evaluation submodule**
   ```bash
   git submodule update --init --recursive
   uv pip install -e "lm-evaluation-harness[ifeval,math]"
   ```

### Running with uv
To run any Python script or command in the environment:
```bash
# Run a script
uv run python run_dllm_inference.py

# Or directly with uv
uv run python -m dllm.core.samplers
```

### Available Extra Dependencies
```bash
# Optional dependencies
uv sync --extra optional    # Install bitsandbytes, vllm
uv sync --extra rl          # Install RL dependencies (trl, math_verify)
uv sync --extra eval        # Install evaluation dependencies (nltk, scikit-learn, etc.)

# Multiple extras
uv sync --extra optional --extra rl --extra eval
```

## Training and Inference

This project provides convenient shell scripts for training and inference with both diffusion language models (DLMs) and traditional autoregressive language models (LLMs).

### Supported Models

**Diffusion Language Models (DLMs):**
- `LLaDA` (8B): `GSAI-ML/LLaDA-8B-Instruct`
- `Dream` (7B): `Dream-org/Dream-v0-Instruct-7B`

**Autoregressive Language Models (LLMs):**
- `Qwen3` (4B): `unsloth/Qwen3-4B-Instruct-2507`
- `LLaMA3` (8B): Local path or HuggingFace model ID

### DLM Training and Inference

#### Fine-tuning a DLM

```bash
# Fine-tune LLaDA on commongen dataset
bash scripts/sft.dllm.sh

# Fine-tune Dream with custom settings
DLLM=dream \
DATASET_NAME=dimongen \
LEARNING_RATE=5e-5 \
NUM_PROCESSES=4 \
bash scripts/sft.dllm.sh
```

**Available environment variables for `sft.dllm.sh`:**
- `DLLM`: Model to use (`llada` or `dream`, default: `llada`)
- `DATASET_NAME`: Dataset name (default: `commongen`)
- `TOP_K`: Top-K augmented samples (default: `5`)
- `RERANKED`: Whether to use reranked data (`True` or `False`, default: `False`)
- `MASK_PROMPT_LOSS`: Mask prompt tokens in loss computation (default: `True`)
- `NUM_PROCESSES`: Number of processes for distributed training (default: `2`)
- `EXTRA_TAG`: Custom tag for the run (optional)
- `LLADA_MODEL`: Model path/ID for LLaDA
- `DREAM_MODEL`: Model path/ID for Dream

#### Inference with a DLM

```bash
# Run inference with LLaDA on test set
bash scripts/infer.dllm.sh

# Run inference with Dream, generating multiple results
DLLM=dream \
DATASET_NAME=dimongen \
NUM_RESULTS=5 \
TEMPERATURE=0.7 \
bash scripts/infer.dllm.sh
```

**Available environment variables for `infer.dllm.sh`:**
- `DLLM`: Model to use (`llada` or `dream`, default: `llada`)
- `DATASET_NAME`: Dataset name (default: `commongen`)
- `TOP_K`: Top-K augmented samples (default: `10`)
- `RERANKED`: Whether data is reranked (default: `False`)
- `BATCH_SIZE`: Inference batch size (default: `32`)
- `NUM_RESULTS`: Number of results to generate per input (default: `1`)
- `TEMPERATURE`: Sampling temperature (auto-set based on `NUM_RESULTS` if not specified)
- `SAMPLING_TOP_P`: Top-P for nucleus sampling (default: `0.95`, Dream only)
- `SAMPLING_TOP_K`: Top-K for sampling (default: `50`, Dream only)
- `CFG_SCALE`: Classifier-free guidance scale (default: `1.0`)
- `EXTRA_TAG`: Custom tag for run identification (optional)

### LLM Training and Inference

#### Fine-tuning an LLM

```bash
# Fine-tune Qwen3 on commongen dataset
bash scripts/sft.llm.sh

# Fine-tune LLaMA3 on dimongen with custom hyperparameters
LLM_MODEL=llama3 \
DATASET_NAME=dimongen \
LEARNING_RATE=2e-5 \
NUM_TRAIN_EPOCHS=3 \
bash scripts/sft.llm.sh
```

**Available environment variables for `sft.llm.sh`:**
- `LLM_MODEL`: Model to use (`qwen3` or `llama3`, default: `qwen3`)
- `DATASET_NAME`: Dataset name (default: `commongen`)
- `NUM_TRAIN_EPOCHS`: Number of training epochs (default: `1`)
- `LEARNING_RATE`: Learning rate (default: `2e-5`)
- `EXTRA_TAG`: Custom tag for the run (optional)
- `QWEN3_MODEL`: Model path/ID for Qwen3
- `LLAMA3_MODEL`: Model path/ID for LLaMA3

#### Inference with an LLM

```bash
# Run inference with Qwen3 on test set
bash scripts/infer.llm.sh

# Generate 5 results per input with LLaMA3
LLM_MODEL=llama3 \
DATASET_NAME=dimongen \
NUM_RESULTS=5 \
TEMPERATURE=0.7 \
bash scripts/infer.llm.sh
```

**Available environment variables for `infer.llm.sh`:**
- `LLM_MODEL`: Model to use (`qwen3` or `llama3`, default: `qwen3`)
- `DATASET_NAME`: Dataset name (default: `commongen`)
- `DATA_TAG`: Data suffix for alternative test sets (optional)
- `BATCH_SIZE`: Inference batch size (default: `32`)
- `NUM_RESULTS`: Number of results per input (default: `1`)
- `TEMPERATURE`: Sampling temperature (auto-set based on `NUM_RESULTS` if not specified)
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: `128`)
- `EXTRA_TAG`: Custom tag for run identification (optional)
- `ADAPTER_PATH`: Path to LoRA adapter checkpoint (auto-configured if not specified)

### Example Workflows

**Complete pipeline: Fine-tune and evaluate LLaDA on CommonGen**

```bash
# Step 1: Fine-tune LLaDA
DLLM=llada DATASET_NAME=commongen bash scripts/sft.dllm.sh

# Step 2: Run inference
DLLM=llada DATASET_NAME=commongen bash scripts/infer.dllm.sh

# Step 3: Evaluate results (if evaluation scripts are available)
# python run_evaluate.py ...
```

**Multiple sampling with Qwen3**

```bash
# Generate 1 result (greedy decoding)
LLM_MODEL=qwen3 DATASET_NAME=commongen NUM_RESULTS=1 bash scripts/infer.llm.sh

# Generate 5 results (temperature sampling)
LLM_MODEL=qwen3 DATASET_NAME=commongen NUM_RESULTS=5 TEMPERATURE=0.8 bash scripts/infer.llm.sh
```

**Custom dataset and output paths**

```bash
# Fine-tune with custom dataset paths
bash scripts/sft.dllm.sh \
  DATASET_NAME=custom_data \
  TRAIN_JSONL_PATH=/path/to/train.jsonl \
  VALID_JSONL_PATH=/path/to/valid.jsonl \
  OUTPUT_DIR=/custom/output/path
```
```diff
- #SBATCH --partition=mllm_safety # Note: adjust this for your cluster
- #SBATCH --quotatype=spot        # Note: adjust this for your cluster
+ #SBATCH --partition=YOUR_PARTITION
+ #SBATCH --quotatype=YOUR_QUOTATYPE
```
Next, create a directory for your job logs:
```shell
mkdir logs
```
This folder will store the log files generated by your sbatch jobs.
