import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_history_file(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def find_history_files(root_dir):
    return sorted(Path(root_dir).glob("**/history.json"))


def build_run_record(path, payload):
    history = payload.get("history", [])
    run_info = payload.get("run_info", {})
    if not history:
        return None

    best_snapshot = min(history, key=lambda snapshot: snapshot["valid_nrmse"])
    final_snapshot = history[-1]
    return {
        "path": str(path),
        "model": run_info["model"],
        "train_size": run_info["train_size"],
        "valid_size": run_info["valid_size"],
        "num_parameters": run_info["num_parameters"],
        "resolved_hidden_dim": run_info["resolved_hidden_dim"],
        "best_valid_nrmse": best_snapshot["valid_nrmse"],
        "best_valid_rmse": best_snapshot["valid_rmse"],
        "final_elapsed_seconds": final_snapshot["elapsed_seconds"],
        "peak_gpu_memory_mb": final_snapshot["peak_gpu_memory_mb"],
        "history": history,
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


def plot_convergence(records, output_path):
    plt.figure(figsize=(8, 5))
    for record in sorted(records, key=lambda item: item["model"]):
        epochs = [snapshot["epoch"] for snapshot in record["history"]]
        values = [snapshot["valid_nrmse"] for snapshot in record["history"]]
        plt.plot(epochs, values, marker="o", label=record["model"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation NRMSE")
    plt.title("Convergence Behavior")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_learning_curves(records, output_path):
    plt.figure(figsize=(8, 5))
    for model in sorted({record["model"] for record in records}):
        model_records = [record for record in records if record["model"] == model]
        model_records.sort(key=lambda item: item["train_size"])
        sizes = [record["train_size"] for record in model_records]
        values = [record["best_valid_nrmse"] for record in model_records]
        plt.plot(sizes, values, marker="o", label=model)
    plt.xlabel("Training Set Size")
    plt.ylabel("Best Validation NRMSE")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
                "best_valid_nrmse": record["best_valid_nrmse"],
                "best_valid_rmse": record["best_valid_rmse"],
                "final_elapsed_seconds": record["final_elapsed_seconds"],
                "peak_gpu_memory_mb": record["peak_gpu_memory_mb"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "training_dynamics_summary.csv", index=False)

    plot_convergence(full_size_records, output_dir / "convergence_nrmse.png")
    plot_learning_curves(learning_curve_records, output_dir / "learning_curves_nrmse.png")
    plot_efficiency(full_size_records, "final_elapsed_seconds", "Elapsed Seconds", output_dir / "efficiency_time_vs_nrmse.png")
    plot_efficiency(full_size_records, "peak_gpu_memory_mb", "Peak GPU Memory (MB)", output_dir / "efficiency_memory_vs_nrmse.png")
    plot_efficiency(full_size_records, "num_parameters", "Parameter Count", output_dir / "efficiency_params_vs_nrmse.png")


if __name__ == "__main__":
    main()
