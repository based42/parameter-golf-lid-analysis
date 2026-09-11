import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

from scipy.stats import kendalltau


def compute_kendall_tau(lid_values, loss_values):
    return float(kendalltau(lid_values, loss_values, variant="b").statistic)


def read_by_step(path, column, min_step, max_step):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            int(float(row["step"])): float(row[column])
            for row in csv.DictReader(file)
            if min_step <= float(row["step"]) <= max_step
        }


def read_measurements(run_ids, analysis_ids, splits, min_step, max_step):
    groups = {split: {} for split in splits}
    for run_id, analysis_id in zip(run_ids, analysis_ids, strict=True):
        directory = Path("runs") / run_id / "analysis" / analysis_id
        for split in splits:
            lid_column = "train_lid" if split == "train" else "val_lid"
            loss_column = ("zeroth_shard_sequences_loss" if split == "train"
                           else "val_shard_sequences_loss")
            lids = read_by_step(directory / "lid.csv", lid_column, min_step, max_step)
            losses = read_by_step(directory / f"{loss_column}.csv", loss_column,
                                  min_step, max_step)
            groups[split][run_id] = {
                step: (lids[step], losses[step]) for step in lids.keys() & losses.keys()
            }
    return groups


def summarize(groups, aggregation):
    results = []

    def add(split, mode, seed, num_seeds, series):
        steps = sorted(series)
        lids, losses = zip(*(series[step] for step in steps))
        results.append(dict(
            split=split, aggregation=mode, seed=seed, num_seeds=num_seeds,
            n_checkpoints=len(steps), min_step=steps[0], max_step=steps[-1],
            tau_b=compute_kendall_tau(lids, losses),
        ))

    for split, seeds in groups.items():
        if aggregation in {"per-seed", "single seed", "both"}:
            for seed, series in sorted(seeds.items()):
                add(split, "per-seed", seed, 1, series)
        if aggregation in {"mean", "both"}:
            series_list = list(seeds.values())
            means = {
                step: tuple(mean(series[step][i] for series in series_list)
                            for i in (0, 1))
                for step in set.intersection(*(set(series) for series in series_list))
            }
            add(split, "mean", "", len(seeds), means)
    return results


def write_results(file, results):
    writer = csv.DictWriter(file, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", nargs="+", required=True, help="Run IDs in analysis-ID order")
    parser.add_argument("--analysis-id", nargs="+", required=True, help="One analysis ID per run")
    parser.add_argument("--split", choices=("train", "validation", "val", "both"), default="both")
    parser.add_argument("--min-step", type=int, default=200)
    parser.add_argument("--max-step", type=int, default=20000)
    parser.add_argument("--aggregation", choices=("per-seed", "single seed", "mean", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    splits = ["train", "validation"] if args.split == "both" else [
        "validation" if args.split == "val" else args.split]
    groups = read_measurements(args.run_id, args.analysis_id, splits,
                               args.min_step, args.max_step)
    results = summarize(groups, args.aggregation)
    write_results(sys.stdout, results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            write_results(file, results)


if __name__ == "__main__":
    main()
