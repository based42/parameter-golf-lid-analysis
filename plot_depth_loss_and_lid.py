"""Plot depth comparisons without input validation.

Supply run IDs and analysis IDs in matching order. Inputs are assumed to use
compatible analysis settings. Only checkpoints shared by all inputs are plotted.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_loss_and_lid import read_series


COLORS = {9: "#677feb", 13: "#df6a85", 17: "#8ced7a", 21: "#e9ca6d", 25:"#9b72cf"}
METRICS = [
    ("zeroth_shard_sequences_loss.csv", "zeroth_shard_sequences_loss"),
    ("val_shard_sequences_loss.csv", "val_shard_sequences_loss"),
    ("lid.csv", "train_lid"),
    ("lid.csv", "val_lid"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--analysis-id", nargs="+", required=True)
    parser.add_argument("--start-step", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("fig/depth_loss_and_lid.pdf"))
    args = parser.parse_args()

    # Group the four metric series by model depth and training seed.
    groups = {}
    for index, run_id in enumerate(args.run_id):
        run_dir = args.runs_dir / run_id
        config = json.loads((run_dir / "config.json").read_text())
        analysis_dir = run_dir / "analysis" / args.analysis_id[index]
        metrics = [
            read_series(analysis_dir / filename, (column,), args.start_step)[column]
            for filename, column in METRICS
        ]
        groups.setdefault(config["num_layers"], {})[config["seed"]] = metrics

    all_series = [series for runs in groups.values() for metrics in runs.values() for series in metrics]
    steps = sorted(set.intersection(*(set(series) for series in all_series)))

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    for depth, runs in sorted(groups.items()):
        color = COLORS[depth]
        label = f"{depth} layers" + (" (baseline)" if depth == 9 else "")

        for metric_index, axis in enumerate(axes.flat):
            values = np.array([
                [metrics[metric_index][step] for step in steps]
                for metrics in runs.values()
            ])
            mean = values.mean(axis=0)
            axis.plot(steps, mean, color=color, linewidth=1.8, label=label)
            if len(runs) > 1:
                std = values.std(axis=0, ddof=1)
                axis.fill_between(steps, mean - std, mean + std,
                                  color=color, alpha=0.3, linewidth=0)

    axes[0, 0].set_title("Fixed zeroth shard training sample")
    axes[0, 1].set_title("Fixed validation sample")
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[1, 0].set_ylabel("Mean LID")
    for axis in axes[1]:
        axis.set_xlabel("Training step")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
