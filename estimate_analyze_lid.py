import argparse
import glob
import re
from pathlib import Path
import numpy as np
import skdim
import torch
from measure_variance_ratio import load_module_from_path
from measure_variance_ratio import build_model
from train_gpt import load_data_shard


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

def neighborhood_size(value):
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return parsed

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate pointwise local intrinsic dimensions of final-layer token " +
            "representations from model checkpoints"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_argument_group("sequence sampling")
    data_group.add_argument(
        "--num-sampled-sequences",
        metavar="M",
        type=positive_int,
        required=True,
        help=(
            "Number of sequences sampled (M)"
        ),
    )
    data_group.add_argument(
        "--sequence-length",
        type=positive_int,
        default=1024,
        help=(
            "Number of tokens per sequence"
        ),
    )
    data_group.add_argument(
        "--sequence-sampling-seed",
        type=int,
        default=42,
    )

    estimator_group = parser.add_argument_group("token sampling and LID estimation")
    estimator_group.add_argument(
        "--num-tokens-sampled",
        metavar="N",
        type=positive_int,
        required=True,
        help=(
            "Number of token representation vectors sampled (N)"
        ),
    )
    estimator_group.add_argument(
        "--token-sampling-seed",
        type=int,
        default=42,
    )
    estimator_group.add_argument(
        "--neighborhood-size",
        metavar="L",
        type=neighborhood_size,
        required=True,
        help=(
            "Number of nearest neighbors used for each pointwise localized TwoNN estimate (L)"
        ),
    )
    return parser

def get_representations(model, input_ids):
    with torch.inference_mode():
        return model.forward_representations(input_ids)

def deduplicate_representations(representations):
    _, indices = np.unique(representations, axis=0, return_index=True)
    return representations[np.sort(indices)]

def main(cli_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = load_module_from_path("train_gpt.py")
    training_args = mod.Hyperparameters()
    model = build_model(mod, device, "standard")


    train_files = sorted(glob.glob(training_args.train_files))
    val_files = sorted(glob.glob(training_args.val_files))

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
        step = training_args.save_checkpoint_every * (count - 1)

        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        estimator = skdim.id.TwoNN(discard_fraction=0.1)

        train_hidden_states = get_representations(model, train_tokens)
        train_hidden_states_np = deduplicate_representations(train_hidden_states.float().cpu().numpy())

        estimator.fit_pw(train_hidden_states_np)
        train_lid = np.mean(estimator.dimension_pw_)

        val_hidden_states = get_representations(model, val_tokens)
        val_hidden_states_np = deduplicate_representations(val_hidden_states.float().cpu().numpy())

        estimator.fit_pw(val_hidden_states_np)
        val_lid = np.mean(estimator.dimension_pw_)

        results[(count - 1), 0] = step
        results[(count - 1), 1] = train_lid
        results[(count - 1), 2] = val_lid
        print(f"Step {step}, Train LID: {train_lid}, Val LID: {val_lid}")
    
    np.savetxt('lid.csv', results, delimiter=",", fmt="%f")

if __name__ == "__main__":
    main(build_parser().parse_args())
