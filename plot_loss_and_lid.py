import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_series(path, columns, start_step):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        return {
            column: {
                float(row["step"]): float(row[column])
                for row in rows
                if float(row["step"]) >= start_step
            }
            for column in columns
        }


def aggregate(series_by_run):
    common_steps = sorted(
        set.intersection(*(set(series) for series in series_by_run))
    )
    if not common_steps:
        raise ValueError("The runs have no measurement steps in common")

    values = np.asarray(
        [[series[step] for step in common_steps] for series in series_by_run]
    )
    ddof = 1 if len(series_by_run) > 1 else 0
    return np.asarray(common_steps), values.mean(axis=0), values.std(axis=0, ddof=ddof)


def plot_mean_with_std(axis, series_by_run, color, label, linestyle="-"):
    steps, mean, std = aggregate(series_by_run)
    axis.fill_between(
        steps,
        mean - std,
        mean + std,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(
        steps,
        mean,
        color=color,
        linewidth=1.6,
        linestyle=linestyle,
        marker="o",
        markersize=2.5,
        label=label,
    )


def add_panel(axis, lid_series, loss_series, title, lid_label):
    lid_color = "#677feb"
    loss_axis = axis.twinx()

    plot_mean_with_std(axis, lid_series, lid_color, lid_label)
    for series, color, label, linestyle in loss_series:
        plot_mean_with_std(loss_axis, series, color, label, linestyle)

    axis.set_title(title)
    axis.set_ylabel("Mean LID")
    loss_axis.set_ylabel("Cross-entropy loss")
    axis.tick_params(axis="y")
    loss_axis.tick_params(axis="y")
    axis.grid(alpha=0.25)

    lid_handles, lid_labels = axis.get_legend_handles_labels()
    loss_handles, loss_labels = loss_axis.get_legend_handles_labels()
    axis.legend(
        lid_handles + loss_handles,
        lid_labels + loss_labels,
        loc="center right",
    )
    return loss_axis


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean training/validation loss and LID with standard deviations"
    )
    parser.add_argument(
        "--run-id",
        nargs="+",
        required=True,
        help="One or more run IDs, in the same order as --analysis-id",
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
        help="Output image path (default: the first run's analysis directory)",
    )
    args = parser.parse_args()

    if len(args.run_id) != len(args.analysis_id):
        parser.error("--run-id and --analysis-id must contain the same number of IDs")

    run_analyses = list(zip(args.run_id, args.analysis_id))
    train_losses = []
    zeroth_train_shard_seq_losses = []
    val_losses = []
    val_shard_seq_losses = []
    train_lids = []
    val_lids = []
    lid_parameters = set()

    for run_id, analysis_id in run_analyses:
        run_dir = Path("runs") / run_id
        analysis_dir = run_dir / "analysis" / analysis_id

        train_losses.append(
            read_series(
                run_dir / "metrics_train.csv", ("train_loss",), args.start_step
            )["train_loss"]
        )
        zeroth_train_shard_seq_losses.append(
            read_series(
                analysis_dir / "zeroth_shard_sequences_loss.csv",
                ("zeroth_shard_sequences_loss",),
                args.start_step,
            )["zeroth_shard_sequences_loss"]
        )
        val_losses.append(
            read_series(run_dir / "metrics_val.csv", ("val_loss",), args.start_step)[
                "val_loss"
            ]
        )
        val_shard_seq_losses.append(
            read_series(
                analysis_dir / "val_shard_sequences_loss.csv",
                ("val_shard_sequences_loss",),
                args.start_step,
            )["val_shard_sequences_loss"]
        )
        lid_series = read_series(
            analysis_dir / "lid.csv", ("train_lid", "val_lid"), args.start_step
        )
        train_lids.append(lid_series["train_lid"])
        val_lids.append(lid_series["val_lid"])

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
            "All analyses must use the same (M, N, L) parameters; found "
            f"{sorted(lid_parameters)}"
        )

    first_run_dir = Path("runs") / args.run_id[0]
    first_analysis_dir = first_run_dir / "analysis" / args.analysis_id[0]
    output_path = args.output or first_analysis_dir / "loss_and_lid.png"

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, sharey=True)

    train_loss_axis = add_panel(
        axes[0],
        train_lids,
        [
            (zeroth_train_shard_seq_losses, "#df6a85", "Fixed training sample loss", "-"),
            (train_losses, "#8ced7a", "Current batch training loss", "--"),
        ],
        "Zeroth training shard",
        "Mean LID",
    )
    val_loss_axis = add_panel(
        axes[1],
        val_lids,
        [
            (val_shard_seq_losses, "#df6a85", "Fixed validation sample loss", "-"),
            (val_losses, "#e9ca6d", "Full validation shard loss", "--"),
        ],
        "Validation shard",
        "Mean LID",
    )

    axes[1].set_xlabel("Training step")

    loss_min = min(train_loss_axis.get_ylim()[0], val_loss_axis.get_ylim()[0])
    loss_max = max(train_loss_axis.get_ylim()[1], val_loss_axis.get_ylim()[1])
    train_loss_axis.set_ylim(loss_min, loss_max)
    val_loss_axis.set_ylim(loss_min, loss_max)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
