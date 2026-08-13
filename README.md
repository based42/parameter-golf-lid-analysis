# Analyzing and Intervening on Local Intrinsic Dimensions during Pre-Training of Small Transformer Models

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager

## Setup

Clone the repository and install python dependencies with uv:

```bash
git clone https://github.com/based42/parameter-golf-lid-analysis.git
cd parameter-golf-lid-analysis
uv sync
```

## Usage

### 1. Download the dataset

```bash
uv run python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This downloads the first 10 training shards along with populating `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/` using the script provided by the [parameter-golf](https://github.com/openai/parameter-golf) repository.

### 2. Train the model

Start a pre-training run:

```bash
uv run python train_gpt.py
```

Checkpoints are saved every `x` steps determined by `CHECKPOINT_EVERY`.

### 3. Compute local intrinsic dimension estimates

Run the LID analysis based on the checkpoints saved from `train_gpt.py`:

```bash
uv run python estimate_analyze_lid.py \
  --num-sampled-sequences 8 \
  --num-sampled-token-vectors 512 \
  --neighborhood-size 32 \
  --run-id <RUN_ID>
```

Outputs the results to `lid.csv`.