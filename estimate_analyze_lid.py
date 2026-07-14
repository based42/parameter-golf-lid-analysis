import torch
import glob
import re
import skdim
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from measure_variance_ratio import load_module_from_path
from measure_variance_ratio import build_model
from train_gpt import load_data_shard

def get_representations(model, input_ids):
    with torch.inference_mode():
        return model.forward_representations(input_ids)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = load_module_from_path("train_gpt.py")
    args = mod.Hyperparameters()
    model = build_model(mod, device, "standard")


    train_files = sorted(glob.glob(args.train_files))
    val_files = sorted(glob.glob(args.val_files))

    train_tokens = load_data_shard(Path(train_files[0])).long().to(device)
    val_tokens = load_data_shard(Path(val_files[0])).long().to(device)

    # probe size
    train_tokens = train_tokens[:8192]
    val_tokens = val_tokens[:8192]


    checkpoints = glob.glob("checkpoints/model_step_*.pt")
    checkpoints.sort(key=lambda f: int(re.search(r"model_step_(\d+)\.pt", f).group(1)))

    results = np.zeros((len(checkpoints), 3))

    count = 0
    for checkpoint in checkpoints:
        count += 1
        step = args.save_checkpoint_every * (count - 1)

        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        estimator = skdim.id.TwoNN(discard_fraction=0.1)

        train_hidden_states = get_representations(model, train_tokens)
        train_hidden_states_np = train_hidden_states.float().cpu().numpy()
        if count == 1:
            train_hidden_states_np = np.unique(train_hidden_states_np, axis=0)

        estimator.fit_pw(train_hidden_states_np)
        train_lid = np.mean(estimator.dimension_pw_)

        val_hidden_states = get_representations(model, val_tokens)
        val_hidden_states_np = val_hidden_states.float().cpu().numpy()
        if count == 1:
            val_hidden_states_np = np.unique(val_hidden_states_np, axis=0)

        estimator.fit_pw(val_hidden_states_np)
        val_lid = np.mean(estimator.dimension_pw_)

        results[(count - 1), 0] = step
        results[(count - 1), 1] = train_lid
        results[(count - 1), 2] = val_lid
        print(f"Step {step}, Train LID: {train_lid}, Val LID: {val_lid}")
    
    np.savetxt('lid.csv', results, delimiter=",", fmt="%f")

if __name__ == "__main__":
    main()