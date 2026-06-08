#!/usr/bin/env python3
"""Train the Task 1 scalar velocity head from expert annotations."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from task_carla_code.task1_velocity_head import (
    BASIC_COMMANDS,
    FEATURE_NAMES,
    annotation_features,
    build_mlp,
    load_annotations,
    split_annotation_path,
    target_speed_from_annotation,
)


def build_examples(annotation_path: pathlib.Path, future_dt: float, waypoint_gap: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    annotations = load_annotations(annotation_path)
    features: list[np.ndarray] = []
    targets: list[float] = []
    counts = {command: 0 for command in BASIC_COMMANDS}
    skipped = 0
    for sample in annotations.values():
        try:
            command = str(sample.get("command", "")).strip().lower()
            target = target_speed_from_annotation(sample, future_dt=future_dt, waypoint_gap=waypoint_gap)
            if command not in counts or target is None:
                skipped += 1
                continue
            features.append(annotation_features(sample))
            targets.append(float(target))
            counts[command] += 1
        except Exception:
            skipped += 1
    if not features:
        raise RuntimeError(f"No usable velocity examples in {annotation_path}; skipped={skipped}")
    return np.stack(features).astype(np.float32), np.asarray(targets, dtype=np.float32)[:, None], counts


def evaluate(model: torch.nn.Module, loader: DataLoader, target_mean: torch.Tensor, target_std: torch.Tensor, device: str) -> dict[str, float]:
    model.eval()
    losses = []
    abs_errors = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            losses.append(float(loss.item()))
            pred_speed = pred * target_std + target_mean
            true_speed = y * target_std + target_mean
            abs_errors.extend(torch.abs(pred_speed - true_speed).detach().cpu().numpy().reshape(-1).tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mae_mps": float(np.mean(abs_errors)) if abs_errors else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--future-dt", type=float, default=0.4)
    parser.add_argument("--waypoint-gap", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-speed-mps", type=float, default=0.0)
    parser.add_argument("--max-speed-mps", type=float, default=8.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    train_path = split_annotation_path(args.data_root, args.train_split)
    val_path = split_annotation_path(args.data_root, args.val_split)
    train_x, train_y, train_counts = build_examples(train_path, args.future_dt, args.waypoint_gap)
    val_x, val_y, val_counts = build_examples(val_path, args.future_dt, args.waypoint_gap)

    feature_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)
    target_mean = float(train_y.mean())
    target_std = float(max(train_y.std(), 1e-6))

    train_xn = (train_x - feature_mean[None, :]) / feature_std[None, :]
    val_xn = (val_x - feature_mean[None, :]) / feature_std[None, :]
    train_yn = (train_y - target_mean) / target_std
    val_yn = (val_y - target_mean) / target_std

    train_ds = TensorDataset(torch.from_numpy(train_xn), torch.from_numpy(train_yn))
    val_ds = TensorDataset(torch.from_numpy(val_xn), torch.from_numpy(val_yn))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_mlp(train_x.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target_mean_tensor = torch.tensor(target_mean, dtype=torch.float32, device=args.device)
    target_std_tensor = torch.tensor(target_std, dtype=torch.float32, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    best_path = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_metrics = evaluate(model, train_loader, target_mean_tensor, target_std_tensor, args.device)
        val_metrics = evaluate(model, val_loader, target_mean_tensor, target_std_tensor, args.device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "train_eval_loss": train_metrics["loss"],
            "train_mae_mps": train_metrics["mae_mps"],
            "val_loss": val_metrics["loss"],
            "val_mae_mps": val_metrics["mae_mps"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["mae_mps"] < best_val:
            best_val = val_metrics["mae_mps"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "feature_names": list(FEATURE_NAMES),
                        "feature_mean": feature_mean.astype(float).tolist(),
                        "feature_std": feature_std.astype(float).tolist(),
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "commands": list(BASIC_COMMANDS),
                        "future_dt": float(args.future_dt),
                        "waypoint_gap": int(args.waypoint_gap),
                        "min_speed_mps": float(args.min_speed_mps),
                        "max_speed_mps": float(args.max_speed_mps),
                    },
                    "train_counts": train_counts,
                    "val_counts": val_counts,
                    "best_epoch": best_epoch,
                    "best_val_mae_mps": best_val,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    summary = {
        "data_root": str(args.data_root),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_samples": int(len(train_x)),
        "val_samples": int(len(val_x)),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "feature_names": list(FEATURE_NAMES),
        "target_mean_mps": target_mean,
        "target_std_mps": target_std,
        "best_epoch": best_epoch,
        "best_val_mae_mps": best_val,
        "best_checkpoint": str(best_path),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
