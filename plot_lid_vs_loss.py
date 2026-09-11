import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def read_by_step(path, value_column, start_step):
    with path.open(newline="", encoding="utf-8") as file:
        return {
            int(float(row["step"])): float(row[value_column])
            for row in csv.DictReader(file)
            if float(row["step"]) >= start_step
        }


def aggregate_pairs(loss_by_run, lid_by_run):
    all_series = [*loss_by_run, *lid_by_run]
    common_steps = sorted(set.intersection(*(set(series) for series in all_series)))
    if not common_steps:
        raise ValueError("The runs have no loss and LID measurement steps in common")

    losses = np.asarray(
        [[series[step] for step in common_steps] for series in loss_by_run]
    )
    lids = np.asarray(
        [[series[step] for step in common_steps] for series in lid_by_run]
    )
    return np.asarray(common_steps), losses.mean(axis=0), lids.mean(axis=0)


def add_panel(axis, steps, mean_loss, mean_lid, title, color_norm):
    axis.plot(
        mean_loss,
        mean_lid,
        color="#999999",
        linewidth=0.9,
        alpha=0.65,
        zorder=1,
    )
    points = axis.scatter(
        mean_loss,
        mean_lid,
        c=steps,
        cmap="viridis",
        norm=color_norm,
        s=35,
        edgecolors="none",
        zorder=2,
    )
    axis.set_title(title)
    axis.set_xlabel("Cross-entropy loss")
    axis.grid(alpha=0.25)
    return points


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot side-by-side mean training and validation LID against mean "
            "fixed-sample loss across seeds"
        )
    )
    parser.add_argument(
        "--run-id",
        nargs="+",
        required=True,
        help="Run IDs, in the same order as --analysis-id",
    )
    parser.add_argument(
        "--analysis-id",
        nargs="+",
        required=True,
        help="One analysis ID per run ID",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=200,
        help="First step shown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output path (default: lid_vs_loss.png in the first "
            "analysis directory)"
        ),
    )
    args = parser.parse_args()

    if len(args.run_id) != len(args.analysis_id):
        parser.error("--run-id and --analysis-id must contain the same number of IDs")

    train_losses = []
    val_losses = []
    train_lids = []
    val_lids = []
    lid_parameters = set()

    for run_id, analysis_id in zip(args.run_id, args.analysis_id, strict=True):
        run_dir = Path("runs") / run_id
        analysis_dir = run_dir / "analysis" / analysis_id

        train_losses.append(
            read_by_step(
                analysis_dir / "zeroth_shard_sequences_loss.csv",
                "zeroth_shard_sequences_loss",
                args.start_step,
            )
        )
        val_losses.append(
            read_by_step(
                analysis_dir / "val_shard_sequences_loss.csv",
                "val_shard_sequences_loss",
                args.start_step,
            )
        )
        train_lids.append(
            read_by_step(analysis_dir / "lid.csv", "train_lid", args.start_step)
        )
        val_lids.append(
            read_by_step(analysis_dir / "lid.csv", "val_lid", args.start_step)
        )

        with (analysis_dir / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)
        lid_parameters.add(
            (
                config["num_sampled_sequences"],
                config["num_sampled_token_vectors"],
                config["neighborhood_size"],
            )
        )

    if len(lid_parameters) != 1:
        raise ValueError(
            "All analyses must use the same (M, N, L) parameters, found "
            f"{sorted(lid_parameters)}"
        )

    train_steps, mean_train_loss, mean_train_lid = aggregate_pairs(
        train_losses, train_lids
    )
    val_steps, mean_val_loss, mean_val_lid = aggregate_pairs(val_losses, val_lids)

    color_norm = Normalize(
        vmin=float(min(train_steps.min(), val_steps.min())),
        vmax=float(max(train_steps.max(), val_steps.max())),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    points = add_panel(
        axes[0],
        train_steps,
        mean_train_loss,
        mean_train_lid,
        "Fixed zeroth training shard sample",
        color_norm,
    )
    add_panel(
        axes[1],
        val_steps,
        mean_val_loss,
        mean_val_lid,
        "Fixed validation shard sample",
        color_norm,
    )
    axes[0].set_ylabel("Mean LID")

    colorbar = figure.colorbar(points, ax=axes, pad=0.02)
    colorbar.set_label("Training step")

    first_analysis_dir = (
        Path("runs") / args.run_id[0] / "analysis" / args.analysis_id[0]
    )
    output_path = args.output or first_analysis_dir / "lid_vs_loss.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
