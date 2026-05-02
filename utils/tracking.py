import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def build_checkpoint_epochs(total_epochs, num_checkpoints):
    if total_epochs < 1:
        raise ValueError("total_epochs must be at least 1")
    if num_checkpoints < 1:
        raise ValueError("num_checkpoints must be at least 1")
    if total_epochs <= num_checkpoints:
        return list(range(1, total_epochs + 1))
    if num_checkpoints == 1:
        return [total_epochs]

    epochs = []
    for index in range(num_checkpoints):
        fraction = index / (num_checkpoints - 1)
        epoch = 1 + round(fraction * (total_epochs - 1))
        if not epochs or epoch != epochs[-1]:
            epochs.append(epoch)

    if epochs[-1] != total_epochs:
        epochs.append(total_epochs)
    return epochs


def get_peak_gpu_memory_mb(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def current_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryLogger:
    def __init__(self, path, run_info, scheduled_epochs):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scheduled_epochs = list(scheduled_epochs)
        self.scheduled_epoch_set = set(self.scheduled_epochs)
        self.data = {
            "run_info": run_info,
            "best_valid_metrics": None,
            "history": [],
        }
        self.write()

    def should_log_epoch(self, epoch):
        return epoch in self.scheduled_epoch_set

    def add_snapshot(
        self,
        epoch,
        model,
        train_size,
        valid_size,
        train_metrics,
        valid_metrics,
        elapsed_seconds,
        peak_gpu_memory_mb,
        num_parameters,
    ):
        snapshot = {
            "model": model,
            "train_size": train_size,
            "valid_size": valid_size,
            "checkpoint_index": len(self.data["history"]) + 1,
            "num_checkpoints": len(self.scheduled_epochs),
            "epoch": epoch,
            "train_nrmse": train_metrics["nrmse"],
            "valid_nrmse": valid_metrics["nrmse"],
            "train_rmse": train_metrics["rmse"],
            "valid_rmse": valid_metrics["rmse"],
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "num_parameters": num_parameters,
        }
        self.data["history"].append(snapshot)
        self.write()

    def update_best(self, epoch, train_metrics, valid_metrics):
        self.data["best_valid_metrics"] = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
        }
        self.write()

    def write(self):
        with open(self.path, "w", encoding="utf-8") as file_handle:
            json.dump(self.data, file_handle, indent=2)
