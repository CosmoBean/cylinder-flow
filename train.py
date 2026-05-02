import argparse
import json
import time
from pathlib import Path

import torch

from datasets import create_dataloader
from metrics import compute_metrics, mean_squared_error
from models import build_model
from utils import HistoryLogger, build_checkpoint_epochs, current_timestamp, ensure_directory, get_peak_gpu_memory_mb, set_seed


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def run_epoch(model, dataloader, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_metrics = {"mse": 0.0, "rmse": 0.0, "nrmse": 0.0}
    num_batches = 0

    for batch in dataloader:
        batch = move_to_device(batch, device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            prediction = model(batch)
            target = batch["targets"]
            loss = mean_squared_error(prediction, target)

            if training:
                loss.backward()
                if optimizer.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer.grad_clip)
                optimizer.step()

        prediction_raw = dataloader.dataset.denormalize_states(prediction.detach())
        target_raw = dataloader.dataset.denormalize_states(target.detach())
        metrics = compute_metrics(prediction_raw, target_raw)

        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "mse": 0.0, "rmse": 0.0, "nrmse": 0.0}

    results = {"loss": total_loss / num_batches}
    for key in total_metrics:
        results[key] = total_metrics[key] / num_batches
    return results


def save_results(path, results):
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_hidden_dim(args, input_dim):
    if args.target_params is None:
        return args.hidden_dim

    candidate_hidden_dims = range(
        args.hidden_dim_min,
        args.hidden_dim_max + 1,
        args.hidden_dim_step,
    )
    best_hidden_dim = None
    best_distance = None

    for hidden_dim in candidate_hidden_dims:
        if args.model in {"transolver", "flare", "gnot", "lno"} and hidden_dim % args.num_heads != 0:
            continue

        model = build_model(
            name=args.model,
            input_dim=input_dim,
            output_steps=args.output_steps,
            hidden_dim=hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_slices=args.num_slices,
            fno_modes=args.fno_modes,
            fno_grid_size=args.fno_grid_size,
        )
        num_parameters = count_parameters(model)
        distance = abs(num_parameters - args.target_params)

        if best_distance is None or distance < best_distance:
            best_hidden_dim = hidden_dim
            best_distance = distance

    if best_hidden_dim is None:
        raise ValueError("No valid hidden_dim candidates found for the requested parameter matching setup.")

    return best_hidden_dim


def train_model(args):
    set_seed(args.seed)
    device_name = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    train_loader = create_dataloader(
        data_dir=args.data_dir,
        split="train",
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        shuffle=True,
        max_samples=args.max_train_samples,
        window_stride=args.window_stride,
    )
    valid_loader = create_dataloader(
        data_dir=args.data_dir,
        split="valid",
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        shuffle=False,
        max_samples=args.max_valid_samples,
        window_stride=args.window_stride,
    )

    resolved_hidden_dim = resolve_hidden_dim(args, train_loader.dataset.node_feature_dim)
    model = build_model(
        name=args.model,
        input_dim=train_loader.dataset.node_feature_dim,
        output_steps=args.output_steps,
        hidden_dim=resolved_hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_slices=args.num_slices,
        fno_modes=args.fno_modes,
        fno_grid_size=args.fno_grid_size,
    ).to(device)
    num_parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.grad_clip = args.grad_clip
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    save_dir = ensure_directory(args.save_dir)
    checkpoint_path = Path(save_dir) / f"{args.model}_best.pt"
    results_path = Path(save_dir) / f"{args.model}_results.json"
    history_path = Path(save_dir) / "history.json"
    checkpoint_epochs = build_checkpoint_epochs(args.epochs, args.num_history_checkpoints)
    train_size = len(train_loader.dataset)
    valid_size = len(valid_loader.dataset)
    run_start_time = time.perf_counter()
    best_valid_nrmse = float("inf")
    best_results = None

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    history_logger = HistoryLogger(
        path=history_path,
        run_info={
            "timestamp": current_timestamp(),
            "model": args.model,
            "device": str(device),
            "num_parameters": num_parameters,
            "resolved_hidden_dim": resolved_hidden_dim,
            "train_size": train_size,
            "valid_size": valid_size,
            "epochs": args.epochs,
            "num_history_checkpoints": args.num_history_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "config": {**vars(args), "resolved_hidden_dim": resolved_hidden_dim},
        },
        scheduled_epochs=checkpoint_epochs,
    )

    print(
        f"device={device} "
        f"model={args.model} "
        f"hidden_dim={resolved_hidden_dim} "
        f"num_parameters={num_parameters} "
        f"train_samples={train_size} "
        f"valid_samples={valid_size}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            training=True,
        )
        valid_metrics = run_epoch(
            model=model,
            dataloader=valid_loader,
            optimizer=None,
            device=device,
            training=False,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_rmse={train_metrics['rmse']:.6f} "
            f"train_nrmse={train_metrics['nrmse']:.6f} "
            f"valid_loss={valid_metrics['loss']:.6f} "
            f"valid_rmse={valid_metrics['rmse']:.6f} "
            f"valid_nrmse={valid_metrics['nrmse']:.6f}"
        )

        if history_logger.should_log_epoch(epoch):
            history_logger.add_snapshot(
                epoch=epoch,
                model=args.model,
                train_size=train_size,
                valid_size=valid_size,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                elapsed_seconds=time.perf_counter() - run_start_time,
                peak_gpu_memory_mb=get_peak_gpu_memory_mb(device),
                num_parameters=num_parameters,
            )

        if valid_metrics["nrmse"] < best_valid_nrmse:
            best_valid_nrmse = valid_metrics["nrmse"]
            best_results = {
                "epoch": epoch,
                "model": args.model,
                "device": str(device),
                "hidden_dim": resolved_hidden_dim,
                "num_parameters": num_parameters,
                "train_samples": train_size,
                "valid_samples": valid_size,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "config": {**vars(args), "resolved_hidden_dim": resolved_hidden_dim},
            }
            history_logger.update_best(
                epoch=epoch,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "node_feature_dim": train_loader.dataset.node_feature_dim,
                    "output_steps": args.output_steps,
                    "hidden_dim": resolved_hidden_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "num_slices": args.num_slices,
                    "fno_modes": args.fno_modes,
                    "fno_grid_size": args.fno_grid_size,
                },
                checkpoint_path,
            )
            save_results(results_path, best_results)
            print(f"saved={checkpoint_path}")

        scheduler.step()

    if best_results is not None:
        print("best_valid_metrics", json.dumps(best_results["valid_metrics"]))
    return best_results


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--model",
        default="gnn",
        choices=["gnn", "transolver", "flare", "gnot", "lno", "fno"],
    )
    parser.add_argument("--input-steps", type=int, default=1)
    parser.add_argument("--output-steps", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-params", type=int, default=None)
    parser.add_argument("--hidden-dim-min", type=int, default=16)
    parser.add_argument("--hidden-dim-max", type=int, default=256)
    parser.add_argument("--hidden-dim-step", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-slices", type=int, default=32)
    parser.add_argument("--fno-modes", type=int, default=12)
    parser.add_argument("--fno-grid-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--num-history-checkpoints", type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
