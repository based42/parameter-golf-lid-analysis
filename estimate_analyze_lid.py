import argparse
import glob
import re
import datetime
import json
import sys
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
        "--num-sampled-token-vectors",
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

    logging_group = parser.add_argument_group("logging, output")
    logging_group.add_argument(
        "--run-id",
        type=str,
        required=True,
        help=(
            "Identifier of training run this analysis will be run on"
        )
    )
    logging_group.add_argument(
        "--analysis-id",
        required=False,
        help=("Identifier for this analysis. If not specified, generates a random ID")
    )
    logging_group.add_argument(
        "--loss-batch-size",
        type=positive_int,
        default=32,
        help="Number of sequences per batch when evaluating zeroth-shard loss",
    )
    return parser

def get_representations(model, input_ids, device, batch_size=16):
    model.eval()
    chunks = []

    with torch.inference_mode():
        for token_batch in input_ids.split(batch_size):
            representations = model.forward_representations(token_batch.to(device))
            chunks.append(representations.float().cpu())

    return torch.cat(chunks, dim=0)

def deduplicate_representations(representations):
    _, indices = np.unique(representations, axis=0, return_index=True)
    return representations[np.sort(indices)]

def sample_representations(representations, num_representations, seed):
    available_vectors = representations.shape[0]

    actual_sample_size = min(num_representations, available_vectors)

    sampled_indices = np.random.default_rng(seed).choice(
        available_vectors, size=actual_sample_size, replace=False)

    return representations[sampled_indices]

def sample_sequences(path, num_sequences, sequence_length, seed):
    path = Path(path)
    num_shard_tokens = load_data_shard(path)
    num_windows = num_shard_tokens.numel() // sequence_length

    if num_sequences > num_windows:
        raise ValueError(
            f"Number of sampled sequences ({num_sequences}) "
            f"can not be bigger than available windows ({num_windows})")

    usable_tokens = num_shard_tokens[:num_windows * sequence_length]
    windows = usable_tokens.reshape(num_windows, sequence_length)

    sampled_indices = np.random.default_rng(seed).choice(num_windows, size=num_sequences, replace=False)

    return windows.long()[torch.from_numpy(sampled_indices)]

def filter_special_token_vectors(input_ids, hidden_states):
    flat_ids = input_ids.reshape(-1)
    special_token_mask = (
            (flat_ids != 0) &
            (flat_ids != 1) &
            (flat_ids != 2)
            )
    hidden_states = hidden_states[special_token_mask]
    return hidden_states

def compute_loss_on_sequences(input_ids, model, device, batch_size):
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = 0

    model.eval()
    with torch.inference_mode():
        for batch in input_ids.split(batch_size):
            x = batch[:, :-1].to(device=device, dtype=torch.int64)
            y = batch[:, 1:].to(device=device, dtype=torch.int64)

            batch_loss = model(x, y).detach()
            batch_token_count = y.numel()
            loss_sum += batch_loss.to(torch.float64) * batch_token_count
            token_count += batch_token_count

    return float((loss_sum / token_count).item())


def main(cli_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cli_args.analysis_id is None:
        cli_args.analysis_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(f"runs/{cli_args.run_id}")
    analysis_dir = run_dir / "analysis" / cli_args.analysis_id

    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    logfile = analysis_dir / "log.txt"
    def log0(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with open(logfile, "a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0("=" * 100, console=False)

    mod = load_module_from_path("train_gpt.py")
    training_args = mod.Hyperparameters()
    model = build_model(mod, device, "standard")


    train_files = sorted(glob.glob(training_args.train_files))
    val_files = sorted(glob.glob(training_args.val_files))

    train_tokens = sample_sequences(
        train_files[0],
        cli_args.num_sampled_sequences,
        cli_args.sequence_length,
        cli_args.sequence_sampling_seed
    )

    val_tokens = sample_sequences(
        val_files[0],
        cli_args.num_sampled_sequences,
        cli_args.sequence_length,
        cli_args.sequence_sampling_seed
    )


    checkpoints = glob.glob(str(run_dir / "checkpoints" / "model_step_*.pt"))
    checkpoints.sort(key=lambda f: int(re.search(r"model_step_(\d+)\.pt", f).group(1)))

    results = np.zeros((len(checkpoints), 3))
    zeroth_shard_sequences_losses = np.zeros((len(checkpoints), 2))

    for result_index, checkpoint in enumerate(checkpoints):
        step = int(re.search(r"model_step_(\d+)\.pt", checkpoint).group(1))

        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        estimator = skdim.id.TwoNN(discard_fraction=0.1)

        train_hidden_states = get_representations(model, train_tokens, device, batch_size=16)

        train_hidden_states = filter_special_token_vectors(train_tokens, train_hidden_states)

        train_hidden_states_np = deduplicate_representations(train_hidden_states.float().cpu().numpy())

        train_hidden_states_np = sample_representations(
            train_hidden_states_np,
            cli_args.num_sampled_token_vectors,
            cli_args.token_sampling_seed)

        estimator.fit_pw(X=train_hidden_states_np,
                         n_neighbors=cli_args.neighborhood_size)
        train_lid = np.mean(estimator.dimension_pw_)

        val_hidden_states = get_representations(model, val_tokens, device, batch_size=16)

        val_hidden_states = filter_special_token_vectors(val_tokens, val_hidden_states)

        val_hidden_states_np = deduplicate_representations(val_hidden_states.float().cpu().numpy())

        val_hidden_states_np = sample_representations(
                    val_hidden_states_np,
                    cli_args.num_sampled_token_vectors,
                    cli_args.token_sampling_seed)

        estimator.fit_pw(X=val_hidden_states_np,
                         n_neighbors=cli_args.neighborhood_size)
        val_lid = np.mean(estimator.dimension_pw_)

        results[result_index, 0] = step
        results[result_index, 1] = train_lid
        results[result_index, 2] = val_lid

        zeroth_shard_sequences_loss = compute_loss_on_sequences(
            train_tokens,
            model,
            device,
            cli_args.loss_batch_size)
        zeroth_shard_sequences_losses[result_index, 0] = step
        zeroth_shard_sequences_losses[result_index, 1] = zeroth_shard_sequences_loss

        log0(f"step:{step} train_lid:{train_lid:.4f} val_lid:{val_lid:.4f}")
        log0(f"zeroth_shard_sequences_loss:{zeroth_shard_sequences_loss:.4f}")

    with open(analysis_dir / "config.json", "w") as f:
            json.dump(vars(cli_args), f, indent=4)

    np.savetxt(f"{analysis_dir}/lid.csv", results, delimiter=",", fmt="%f", header="step,train_lid,val_lid", comments="")
    np.savetxt(f"{analysis_dir}/zeroth_shard_sequences_loss.csv",
               zeroth_shard_sequences_losses,
               delimiter=",",
               fmt="%f",
               header="step,zeroth_shard_sequences_loss",
               comments="")
    

if __name__ == "__main__":
    main(build_parser().parse_args())
