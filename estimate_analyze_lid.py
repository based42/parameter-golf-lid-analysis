import torch
from measure_variance_ratio import load_module_from_path
from measure_variance_ratio import build_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = load_module_from_path("train_gpt.py")
    args = mod.Hyperparameters
    model = build_model(mod, device, "standard")

if __name__ == "__main__":
    main()