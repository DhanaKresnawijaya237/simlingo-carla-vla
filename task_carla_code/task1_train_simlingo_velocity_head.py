#!/usr/bin/env python3
"""Train a Task 1 velocity head from SimLingo closed-loop tick logs.

This head consumes SimLingo outputs, not only ego state:

    predicted speed waypoints + predicted path + command + current speed
        -> corrected scalar target speed

It is intended for stop / speed up / slow down commands while SimLingo or the
map lane route still controls lateral motion.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_carla_code.task1_velocity_head import (
    BASIC_COMMANDS,
    SIMLINGO_SPEED_FEATURE_NAMES,
    SIMLINGO_SPEED_FEATURE_VERSION,
    build_mlp,
    build_simlingo_velocity_features,
    estimate_speed_from_waypoints,
)


SPEED_COMMANDS = ("stop", "speed up", "slow down")


def normalize_command(command: str) -> str:
    normalized = command.strip().lower().replace("_", " ")
    if normalized not in BASIC_COMMANDS:
        raise ValueError(f"Unsupported command {command!r}")
    return normalized


def iter_tick_paths(logs_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in logs_root.rglob("ticks.jsonl")
        if ".merge_backups" not in path.parts
    )


def read_ticks(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def tick_baseline_speed(tick: dict[str, Any]) -> float:
    instant = tick.get("instant_success")
    if isinstance(instant, dict):
        for key in ("baseline_speed_mps", "baseline_speed"):
            if key in instant:
                try:
                    return float(instant[key])
                except Exception:
                    pass
    cached = tick.get("cached_follow")
    if isinstance(cached, dict):
        value = cached.get("cached_model_target_speed_mps")
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return float(tick.get("speed_mps", 0.0))


def stop_schedule_target_speed(
    baseline_speed_mps: float,
    elapsed_sec: float,
    args: argparse.Namespace,
) -> float:
    if args.stop_target_mode == "immediate":
        return float(args.stop_target_speed_mps)

    decel_time = max(1e-3, float(args.stop_decel_time_sec))
    progress = float(np.clip(float(elapsed_sec) / decel_time, 0.0, 1.0))
    target = baseline_speed_mps + (float(args.stop_target_speed_mps) - baseline_speed_mps) * progress
    return float(np.clip(target, float(args.stop_target_speed_mps), float(args.max_speed_mps)))


def semantic_target_speed(
    command: str,
    baseline_speed_mps: float,
    elapsed_sec: float,
    args: argparse.Namespace,
) -> float:
    if command == "stop":
        return stop_schedule_target_speed(baseline_speed_mps, elapsed_sec, args)
    if command == "slow down":
        if args.target_mode == "scaled":
            scaled = baseline_speed_mps * float(args.slowdown_ratio)
            min_delta = baseline_speed_mps - float(args.slowdown_min_delta_mps)
            return float(
                np.clip(
                    min(scaled, min_delta),
                    float(args.slowdown_min_speed_mps),
                    float(args.max_speed_mps),
                )
            )
        return float(
            np.clip(
                baseline_speed_mps - float(args.slowdown_delta_mps),
                float(args.slowdown_min_speed_mps),
                float(args.max_speed_mps),
            )
        )
    if command == "speed up":
        if args.target_mode == "scaled":
            scaled = baseline_speed_mps * float(args.speedup_ratio)
            min_delta = baseline_speed_mps + float(args.speedup_min_delta_mps)
            return float(
                np.clip(
                    max(scaled, min_delta),
                    float(args.min_speed_mps),
                    float(args.speedup_max_speed_mps),
                )
            )
        return float(
            np.clip(
                baseline_speed_mps + float(args.speedup_delta_mps),
                float(args.min_speed_mps),
                float(args.speedup_max_speed_mps),
            )
        )
    raise ValueError(f"Not a speed command: {command}")


def tick_command_elapsed_sec(tick: dict[str, Any]) -> float:
    try:
        return max(0.0, float(tick.get("time_sec", 0.0)) - float(tick.get("warmup_sec", 0.0)))
    except Exception:
        return 0.0


def tick_to_example(
    tick: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, str] | None:
    command = normalize_command(str(tick.get("execution_command") or tick.get("command") or ""))
    if command not in SPEED_COMMANDS:
        return None
    if str(tick.get("speed_test_phase", "command")) != "command":
        return None
    if args.require_fresh_velocity_head_source and not bool(tick.get("velocity_head_fresh", False)):
        return None
    speed_waypoints = tick.get("predicted_speed_waypoints")
    if speed_waypoints is None:
        return None
    route = tick.get("predicted_route")
    cached = tick.get("cached_follow") if isinstance(tick.get("cached_follow"), dict) else {}
    model_target = cached.get("cached_model_target_speed_mps")
    if model_target is None:
        model_target = estimate_speed_from_waypoints(speed_waypoints)
    previous_steer = float(tick.get("steer", 0.0))
    previous_throttle = float(tick.get("throttle", 0.0))
    previous_brake = float(tick.get("brake", 0.0))
    current_speed = float(tick.get("speed_mps", 0.0))
    baseline = tick_baseline_speed(tick)
    elapsed = tick_command_elapsed_sec(tick)
    target = semantic_target_speed(command, baseline, elapsed, args)
    features = build_simlingo_velocity_features(
        command=command,
        current_speed_mps=current_speed,
        previous_steer=previous_steer,
        previous_throttle=previous_throttle,
        previous_brake=previous_brake,
        baseline_speed_mps=baseline,
        command_elapsed_sec=elapsed,
        predicted_speed_waypoints=speed_waypoints,
        predicted_route=route,
        model_target_speed_mps=model_target,
    )
    return features, target, command


def tick_to_stop_augmented_examples(
    tick: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, float, str]]:
    """Add stop examples for intermediate speeds not present in abrupt-stop logs."""
    if int(args.stop_augment_samples_per_tick) <= 0:
        return []
    command = normalize_command(str(tick.get("execution_command") or tick.get("command") or ""))
    if command != "stop":
        return []
    if str(tick.get("speed_test_phase", "command")) != "command":
        return []
    speed_waypoints = tick.get("predicted_speed_waypoints")
    if speed_waypoints is None:
        return []
    route = tick.get("predicted_route")
    cached = tick.get("cached_follow") if isinstance(tick.get("cached_follow"), dict) else {}
    model_target = cached.get("cached_model_target_speed_mps")
    if model_target is None:
        model_target = estimate_speed_from_waypoints(speed_waypoints)
    baseline = tick_baseline_speed(tick)
    if baseline <= 0.1:
        return []
    previous_steer = float(tick.get("steer", 0.0))
    previous_brake = float(args.stop_augment_previous_brake)
    previous_throttle = 0.0
    elapsed = tick_command_elapsed_sec(tick)
    count = int(args.stop_augment_samples_per_tick)
    low = float(args.stop_augment_min_current_speed_mps)
    high = min(float(args.stop_augment_max_current_speed_mps), float(baseline))
    if high <= low:
        return []
    speeds = np.linspace(low, high, count + 2, dtype=np.float32)[1:-1]
    out: list[tuple[np.ndarray, float, str]] = []
    for current_speed in speeds:
        target = semantic_target_speed("stop", baseline, elapsed, args)
        features = build_simlingo_velocity_features(
            command="stop",
            current_speed_mps=float(current_speed),
            previous_steer=previous_steer,
            previous_throttle=previous_throttle,
            previous_brake=previous_brake,
            baseline_speed_mps=baseline,
            command_elapsed_sec=elapsed,
            predicted_speed_waypoints=speed_waypoints,
            predicted_route=route,
            model_target_speed_mps=model_target,
        )
        out.append((features, target, "stop"))
    return out


def load_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    counts = {command: 0 for command in SPEED_COMMANDS}
    augmented_counts = {command: 0 for command in SPEED_COMMANDS}
    skipped = 0
    for ticks_path in iter_tick_paths(args.logs_root):
        rows = read_ticks(ticks_path)
        if args.max_ticks_per_trial > 0:
            rows = rows[:: max(1, int(args.sample_stride))][: int(args.max_ticks_per_trial)]
        else:
            rows = rows[:: max(1, int(args.sample_stride))]
        trial_id = str(ticks_path.parent.relative_to(args.logs_root))
        for tick in rows:
            try:
                item = tick_to_example(tick, args)
            except Exception:
                item = None
            if item is None:
                skipped += 1
                continue
            features, target, command = item
            examples.append(
                {
                    "features": features,
                    "target": float(target),
                    "command": command,
                    "trial_id": trial_id,
                    "ticks_path": str(ticks_path),
                }
            )
            counts[command] += 1
            for aug_features, aug_target, aug_command in tick_to_stop_augmented_examples(tick, args):
                examples.append(
                    {
                        "features": aug_features,
                        "target": float(aug_target),
                        "command": aug_command,
                        "trial_id": trial_id,
                        "ticks_path": str(ticks_path),
                        "augmented": True,
                    }
                )
                counts[aug_command] += 1
                augmented_counts[aug_command] += 1
    if not examples:
        raise RuntimeError(f"No usable SimLingo velocity examples under {args.logs_root}; skipped={skipped}")
    print(json.dumps({"loaded_examples": len(examples), "augmented_counts": augmented_counts}, indent=2), flush=True)
    return examples, counts


def split_by_trial(
    examples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_trial[str(example["trial_id"])].append(example)
    trial_ids = sorted(by_trial)
    rng = random.Random(seed)
    rng.shuffle(trial_ids)
    val_count = max(1, int(round(len(trial_ids) * float(val_ratio))))
    val_ids = set(trial_ids[:val_count])
    train = []
    val = []
    for trial_id, trial_examples in by_trial.items():
        if trial_id in val_ids:
            val.extend(trial_examples)
        else:
            train.extend(trial_examples)
    if not train or not val:
        raise RuntimeError(f"Bad split: train={len(train)} val={len(val)} trials={len(trial_ids)}")
    return train, val


def stack_examples(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    x = np.stack([example["features"] for example in examples]).astype(np.float32)
    y = np.asarray([example["target"] for example in examples], dtype=np.float32)[:, None]
    counts = {command: 0 for command in SPEED_COMMANDS}
    for example in examples:
        counts[str(example["command"])] += 1
    return x, y, counts


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses = []
    abs_errors = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(float(torch.nn.functional.mse_loss(pred, y).item()))
            pred_speed = pred * target_std + target_mean
            true_speed = y * target_std + target_mean
            abs_errors.extend(torch.abs(pred_speed - true_speed).detach().cpu().numpy().reshape(-1).tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mae_mps": float(np.mean(abs_errors)) if abs_errors else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-ticks-per-trial", type=int, default=0)
    parser.add_argument("--require-fresh-velocity-head-source", action="store_true")
    parser.add_argument(
        "--target-mode",
        choices=("fixed-delta", "scaled"),
        default="fixed-delta",
        help=(
            "fixed-delta uses baseline +/- constant deltas. "
            "scaled uses baseline ratios with min deltas, which is better when warmup speeds vary."
        ),
    )
    parser.add_argument("--stop-target-speed-mps", type=float, default=0.0)
    parser.add_argument(
        "--stop-target-mode",
        choices=("immediate", "linear-decel"),
        default="linear-decel",
        help="immediate labels stop as 0 m/s immediately; linear-decel ramps from baseline to stop target.",
    )
    parser.add_argument("--stop-decel-time-sec", type=float, default=1.5)
    parser.add_argument("--slowdown-delta-mps", type=float, default=2.0)
    parser.add_argument("--speedup-delta-mps", type=float, default=1.5)
    parser.add_argument("--slowdown-ratio", type=float, default=0.60)
    parser.add_argument("--speedup-ratio", type=float, default=1.22)
    parser.add_argument("--slowdown-min-delta-mps", type=float, default=0.8)
    parser.add_argument("--speedup-min-delta-mps", type=float, default=0.9)
    parser.add_argument("--slowdown-min-speed-mps", type=float, default=1.0)
    parser.add_argument("--speedup-max-speed-mps", type=float, default=7.5)
    parser.add_argument("--stop-augment-samples-per-tick", type=int, default=0)
    parser.add_argument("--stop-augment-min-current-speed-mps", type=float, default=0.0)
    parser.add_argument("--stop-augment-max-current-speed-mps", type=float, default=7.0)
    parser.add_argument("--stop-augment-previous-brake", type=float, default=0.6)
    parser.add_argument("--min-speed-mps", type=float, default=0.0)
    parser.add_argument("--max-speed-mps", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    examples, all_counts = load_examples(args)
    train_examples, val_examples = split_by_trial(examples, args.val_ratio, args.seed)
    train_x, train_y, train_counts = stack_examples(train_examples)
    val_x, val_y, val_counts = stack_examples(val_examples)

    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    target_mean = float(train_y.mean())
    target_std = float(max(train_y.std(), 1e-6))

    train_xn = (train_x - feature_mean[None, :]) / feature_std[None, :]
    val_xn = (val_x - feature_mean[None, :]) / feature_std[None, :]
    train_yn = (train_y - target_mean) / target_std
    val_yn = (val_y - target_mean) / target_std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_xn), torch.from_numpy(train_yn)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_xn), torch.from_numpy(val_yn)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_mlp(train_x.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target_mean_tensor = torch.tensor(target_mean, dtype=torch.float32, device=args.device)
    target_std_tensor = torch.tensor(target_std, dtype=torch.float32, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

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
                        "feature_version": SIMLINGO_SPEED_FEATURE_VERSION,
                        "feature_names": list(SIMLINGO_SPEED_FEATURE_NAMES),
                        "feature_mean": feature_mean.astype(float).tolist(),
                        "feature_std": feature_std.astype(float).tolist(),
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "commands": list(BASIC_COMMANDS),
                        "future_dt": 0.0,
                        "waypoint_gap": 0,
                        "label_mode": f"semantic_relative_to_baseline_{args.target_mode}",
                        "stop_target_speed_mps": float(args.stop_target_speed_mps),
                        "stop_target_mode": str(args.stop_target_mode),
                        "stop_decel_time_sec": float(args.stop_decel_time_sec),
                        "slowdown_delta_mps": float(args.slowdown_delta_mps),
                        "speedup_delta_mps": float(args.speedup_delta_mps),
                        "slowdown_ratio": float(args.slowdown_ratio),
                        "speedup_ratio": float(args.speedup_ratio),
                        "slowdown_min_delta_mps": float(args.slowdown_min_delta_mps),
                        "speedup_min_delta_mps": float(args.speedup_min_delta_mps),
                        "slowdown_min_speed_mps": float(args.slowdown_min_speed_mps),
                        "speedup_max_speed_mps": float(args.speedup_max_speed_mps),
                        "stop_augment_samples_per_tick": int(args.stop_augment_samples_per_tick),
                        "stop_augment_min_current_speed_mps": float(args.stop_augment_min_current_speed_mps),
                        "stop_augment_max_current_speed_mps": float(args.stop_augment_max_current_speed_mps),
                        "stop_augment_previous_brake": float(args.stop_augment_previous_brake),
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
        "logs_root": str(args.logs_root),
        "output_dir": str(args.output_dir),
        "feature_version": SIMLINGO_SPEED_FEATURE_VERSION,
        "feature_names": list(SIMLINGO_SPEED_FEATURE_NAMES),
        "total_samples": int(len(examples)),
        "train_samples": int(len(train_x)),
        "val_samples": int(len(val_x)),
        "all_counts": all_counts,
        "train_counts": train_counts,
        "val_counts": val_counts,
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
