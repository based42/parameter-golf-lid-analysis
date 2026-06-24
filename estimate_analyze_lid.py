import torch
import glob
import numpy as np
from measure_variance_ratio import load_module_from_path
from measure_variance_ratio import build_model
from train_gpt import load_data_shard

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = load_module_from_path("train_gpt.py")
    args = mod.Hyperparameters
    model = build_model(mod, device, "standard")


    train_files = sorted(glob.glob(args.train_files))
    val_files = sorted(glob.glob(args.val_files))

    train_tokens = load_data_shard(train_files[0])
    val_tokens = load_data_shard(val_files[0])

    # probe size
    train_tokens = train_tokens[:8192]
    val_tokens = val_tokens[:8192]

if __name__ == "__main__":
    main()