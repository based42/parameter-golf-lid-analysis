# Analyzing and Intervening on Local Intrinsic Dimensions during Pre-Training of Small Transformer Models

## Setup

Clone the repository, create a new python environment and install the dependencies:

```bash
git clone https://github.com/based42/parameter-golf-lid-analysis.git
cd parameter-golf-lid-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Download the dataset

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This downloads the first 10 training shards along with populating `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/` using the script provided by the [parameter-golf](https://github.com/openai/parameter-golf) repository.

### 2. Train the model

Start a pre-training run:

```bash
python train_gpt.py
```

Checkpoints are saved every `x` steps determined by `CHECKPOINT_EVERY`.

### 3. Compute local intrinsic dimension estimates

Run the LID analysis based on the checkpoints saved from `train_gpt.py`:

```bash
python estimate_analyze_lid.py
```

Outputs the results to `lid.csv`.