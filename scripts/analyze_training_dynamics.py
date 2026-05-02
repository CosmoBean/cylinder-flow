import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLORS = {
    "flare": "#1b9e77",
    "transolver": "#d95f02",
    "gnn": "#7570b3",
    "fno": "#e7298a",
    "gnot": "#66a61e",
    "lno": "#e6ab02",
}

LOSS_LINE_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train_loss>[0-9.]+).*"
    r"valid_loss=(?P<valid_loss>[0-9.]+)"
)


def load_history_file(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def find_history_files(root_dir):
    return sorted(Path(root_dir).glob("**/history.json"))


def parse_loss_log(path):
    epochs = []
    train_losses = []
    valid_losses = []

    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOSS_LINE_PATTERN.search(line)
        if not match:
            continue
        epochs.append(int(match.group("epoch")))
        train_losses.append(float(match.group("train_loss")))
        valid_losses.append(float(match.group("valid_loss")))

    if not epochs:
        return None

    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "valid_loss": valid_losses,
    }


def build_run_record(path, payload):
    history = payload.get("history", [])
    run_info = payload.get("run_info", {})
    if not history:
        return None

    best_snapshot = min(history, key=lambda snapshot: snapshot["valid_nrmse"])
    final_snapshot = history[-1]
    model = run_info["model"]
    log_data = parse_loss_log(path.parent / "train.log")

    return {
        "path": str(path),
        "model": model,
        "color": MODEL_COLORS.get(model, None),
        "train_size": run_info["train_size"],
        "valid_size": run_info["valid_size"],
        "num_parameters": run_info["num_parameters"],
        "resolved_hidden_dim": run_info["resolved_hidden_dim"],
        "best_valid_nrmse": best_snapshot["valid_nrmse"],
        "best_valid_rmse": best_snapshot["valid_rmse"],
        "best_epoch": best_snapshot["epoch"],
        "final_elapsed_seconds": final_snapshot["elapsed_seconds"],
        "peak_gpu_memory_mb": final_snapshot["peak_gpu_memory_mb"],
        "history": history,
        "loss_history": log_data,
    }


def keep_best_per_train_size(records):
    best_records = {}
    for record in records:
        key = (record["model"], record["train_size"])
        current = best_records.get(key)
        if current is None or record["best_valid_nrmse"] < current["best_valid_nrmse"]:
            best_records[key] = record
    return list(best_records.values())


def select_full_size_records(records):
    selected = {}
    for record in records:
        model = record["model"]
        current = selected.get(model)
        if current is None:
            selected[model] = record
            continue
        if record["train_size"] > current["train_size"]:
            selected[model] = record
            continue
        if record["train_size"] == current["train_size"] and record["best_valid_nrmse"] < current["best_valid_nrmse"]:
            selected[model] = record
    return list(selected.values())


def plot_metric_curves(records, metric_key, title, ylabel, output_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=180)
    for record in sorted(records, key=lambda item: item["model"]):
        epochs = [snapshot["epoch"] for snapshot in record["history"]]
        values = [snapshot[metric_key] for snapshot in record["history"]]
        ax.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=record["model"].upper(),
            color=record["color"],
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, max(max([s["epoch"] for s in r["history"]]) for r in records) + 1))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(records, loss_key, title, output_path):
    plot_records = [record for record in records if record["loss_history"] is not None]
    if not plot_records:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=180)
    for record in sorted(plot_records, key=lambda item: item["model"]):
        ax.plot(
            record["loss_history"]["epochs"],
            record["loss_history"][loss_key],
            marker="o",
            linewidth=2,
            markersize=4,
            label=record["model"].upper(),
            color=record["color"],
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(range(1, max(max(record["loss_history"]["epochs"]) for record in plot_records) + 1))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(records, x_key, x_label, output_path):
    plt.figure(figsize=(8, 5))
    for record in sorted(records, key=lambda item: item["model"]):
        plt.scatter(record[x_key], record["best_valid_nrmse"], label=record["model"])
        plt.text(record[x_key], record["best_valid_nrmse"], record["model"])
    plt.xlabel(x_label)
    plt.ylabel("Best Validation NRMSE")
    plt.title(f"{x_label} vs Validation NRMSE")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="out")
    parser.add_argument("--output-dir", default="out/training_dynamics")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    history_files = find_history_files(args.root_dir)
    run_records = []
    for history_file in history_files:
        payload = load_history_file(history_file)
        record = build_run_record(history_file, payload)
        if record is not None:
            run_records.append(record)

    if args.models:
        requested_models = set(args.models)
        run_records = [record for record in run_records if record["model"] in requested_models]

    if not run_records:
        raise ValueError("No history.json files found for the requested analysis.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_curve_records = keep_best_per_train_size(run_records)
    full_size_records = select_full_size_records(run_records)

    summary_rows = []
    for record in sorted(learning_curve_records, key=lambda item: (item["model"], item["train_size"])):
        summary_rows.append(
            {
                "model": record["model"],
                "train_size": record["train_size"],
                "valid_size": record["valid_size"],
                "num_parameters": record["num_parameters"],
                "resolved_hidden_dim": record["resolved_hidden_dim"],
                "best_epoch": record["best_epoch"],
                "best_valid_nrmse": record["best_valid_nrmse"],
                "best_valid_rmse": record["best_valid_rmse"],
                "final_elapsed_seconds": record["final_elapsed_seconds"],
                "peak_gpu_memory_mb": record["peak_gpu_memory_mb"],
                "has_loss_log": record["loss_history"] is not None,
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "training_dynamics_summary.csv", index=False)

    plot_metric_curves(full_size_records, "valid_nrmse", "Validation NRMSE by Epoch", "NRMSE", output_dir / "validation_curves.png")
    plot_metric_curves(full_size_records, "valid_nrmse", "Validation NRMSE by Epoch", "NRMSE", output_dir / "validation_nrmse.png")
    plot_metric_curves(full_size_records, "valid_rmse", "Validation RMSE by Epoch", "RMSE", output_dir / "validation_rmse.png")
    plot_metric_curves(full_size_records, "peak_gpu_memory_mb", "Peak GPU Memory by Epoch", "Peak GPU Memory (MB)", output_dir / "peak_gpu_memory_by_epoch.png")

    for record in full_size_records:
        elapsed = [snapshot["elapsed_seconds"] for snapshot in record["history"]]
        epoch_time = [elapsed[0]] + [elapsed[i] - elapsed[i - 1] for i in range(1, len(elapsed))]
        for index, snapshot in enumerate(record["history"]):
            snapshot["epoch_time_seconds"] = epoch_time[index]
    plot_metric_curves(full_size_records, "epoch_time_seconds", "Epoch Time by Epoch", "Epoch Time (s)", output_dir / "epoch_time_by_epoch.png")

    plot_loss_curves(full_size_records, "train_loss", "Training Loss by Epoch", output_dir / "training_loss_curves.png")
    plot_loss_curves(full_size_records, "valid_loss", "Validation Loss by Epoch", output_dir / "validation_loss_curves.png")

    plot_efficiency(full_size_records, "final_elapsed_seconds", "Elapsed Seconds", output_dir / "efficiency_time_vs_nrmse.png")
    plot_efficiency(full_size_records, "peak_gpu_memory_mb", "Peak GPU Memory (MB)", output_dir / "efficiency_memory_vs_nrmse.png")
    plot_efficiency(full_size_records, "num_parameters", "Parameter Count", output_dir / "efficiency_params_vs_nrmse.png")


if __name__ == "__main__":
    main()
