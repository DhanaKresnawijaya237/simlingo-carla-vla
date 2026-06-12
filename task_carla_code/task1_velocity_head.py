#!/usr/bin/env python3
"""Small scalar velocity head for Task 1 explicit speed commands."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np


BASIC_COMMANDS = ("turn left", "turn right", "straight", "stop", "speed up", "slow down")
FEATURE_NAMES_V1 = (
    "command_turn_left",
    "command_turn_right",
    "command_straight",
    "command_stop",
    "command_speed_up",
    "command_slow_down",
    "current_speed_mps",
    "previous_steer",
    "previous_throttle",
    "previous_brake",
)
FEATURE_NAMES_V2 = FEATURE_NAMES_V1 + (
    "baseline_speed_mps",
    "command_elapsed_sec",
)
FEATURE_NAMES = FEATURE_NAMES_V2
SIMLINGO_SPEED_MAX_POINTS = 8
SIMLINGO_ROUTE_MAX_POINTS = 12
SIMLINGO_SPEED_FEATURE_VERSION = "simlingo_speed_v1"


def _simlingo_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(FEATURE_NAMES_V2)
    names.extend(
        (
            "model_target_speed_mps",
            "speed_wp_count",
            "speed_wp_path_length_m",
            "speed_wp_mean_step_m",
            "speed_wp_final_x_forward",
            "speed_wp_final_y_right",
            "speed_wp_mean_abs_y_right",
            "route_wp_count",
            "route_path_length_m",
            "route_mean_step_m",
            "route_final_x_forward",
            "route_final_y_right",
            "route_mean_abs_y_right",
            "route_max_abs_y_right",
        )
    )
    for idx in range(SIMLINGO_SPEED_MAX_POINTS):
        names.append(f"speed_wp_{idx}_x_forward")
        names.append(f"speed_wp_{idx}_y_right")
    for idx in range(SIMLINGO_ROUTE_MAX_POINTS):
        names.append(f"route_wp_{idx}_x_forward")
        names.append(f"route_wp_{idx}_y_right")
    return tuple(names)


SIMLINGO_SPEED_FEATURE_NAMES = _simlingo_feature_names()


def command_to_index(command: str) -> int:
    normalized = command.strip().lower().replace("_", " ")
    if normalized not in BASIC_COMMANDS:
        raise ValueError(f"Unsupported command {command!r}")
    return BASIC_COMMANDS.index(normalized)


def build_velocity_features(
    command: str,
    current_speed_mps: float,
    previous_steer: float = 0.0,
    previous_throttle: float = 0.0,
    previous_brake: float = 0.0,
    baseline_speed_mps: float | None = None,
    command_elapsed_sec: float = 0.0,
    feature_names: tuple[str, ...] | list[str] = FEATURE_NAMES,
) -> np.ndarray:
    normalized = command.strip().lower().replace("_", " ")
    values = {name: 0.0 for name in feature_names}
    command_name = f"command_{normalized.replace(' ', '_')}"
    if command_name in values:
        values[command_name] = 1.0
    values["current_speed_mps"] = float(current_speed_mps)
    values["previous_steer"] = float(previous_steer)
    values["previous_throttle"] = float(previous_throttle)
    values["previous_brake"] = float(previous_brake)
    if "baseline_speed_mps" in values:
        values["baseline_speed_mps"] = float(current_speed_mps if baseline_speed_mps is None else baseline_speed_mps)
    if "command_elapsed_sec" in values:
        values["command_elapsed_sec"] = float(command_elapsed_sec)
    features = np.asarray([values[name] for name in feature_names], dtype=np.float32)
    return features


def extract_xy_array(raw: Any, max_points: int | None = None) -> np.ndarray:
    if raw is None:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr = arr[:, :2].astype(np.float32, copy=True)
    if max_points is not None:
        arr = arr[: int(max_points)]
    return arr


def estimate_speed_from_waypoints(raw_speed_waypoints: Any) -> float | None:
    points = extract_xy_array(raw_speed_waypoints)
    if len(points) < 3:
        return None
    speed = float(np.linalg.norm(points[0] - points[2]) * 2.0)
    if not math.isfinite(speed):
        return None
    return max(0.0, speed)


def _path_summary(points: np.ndarray) -> dict[str, float]:
    if points.size == 0:
        return {
            "count": 0.0,
            "path_length_m": 0.0,
            "mean_step_m": 0.0,
            "final_x_forward": 0.0,
            "final_y_right": 0.0,
            "mean_abs_y_right": 0.0,
            "max_abs_y_right": 0.0,
        }
    if len(points) > 1:
        steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
        path_length = float(np.sum(steps))
        mean_step = float(np.mean(steps))
    else:
        path_length = 0.0
        mean_step = 0.0
    y = points[:, 1]
    return {
        "count": float(len(points)),
        "path_length_m": path_length,
        "mean_step_m": mean_step,
        "final_x_forward": float(points[-1, 0]),
        "final_y_right": float(points[-1, 1]),
        "mean_abs_y_right": float(np.mean(np.abs(y))),
        "max_abs_y_right": float(np.max(np.abs(y))),
    }


def build_simlingo_velocity_features(
    command: str,
    current_speed_mps: float,
    previous_steer: float = 0.0,
    previous_throttle: float = 0.0,
    previous_brake: float = 0.0,
    baseline_speed_mps: float | None = None,
    command_elapsed_sec: float = 0.0,
    predicted_speed_waypoints: Any = None,
    predicted_route: Any = None,
    model_target_speed_mps: float | None = None,
    feature_names: tuple[str, ...] | list[str] = SIMLINGO_SPEED_FEATURE_NAMES,
) -> np.ndarray:
    values = {name: 0.0 for name in feature_names}
    base_features = build_velocity_features(
        command,
        current_speed_mps,
        previous_steer=previous_steer,
        previous_throttle=previous_throttle,
        previous_brake=previous_brake,
        baseline_speed_mps=baseline_speed_mps,
        command_elapsed_sec=command_elapsed_sec,
        feature_names=[name for name in FEATURE_NAMES_V2 if name in values],
    )
    for name, value in zip([name for name in FEATURE_NAMES_V2 if name in values], base_features):
        values[name] = float(value)

    speed_points = extract_xy_array(predicted_speed_waypoints, SIMLINGO_SPEED_MAX_POINTS)
    route_points = extract_xy_array(predicted_route, SIMLINGO_ROUTE_MAX_POINTS)
    if model_target_speed_mps is None:
        model_target_speed_mps = estimate_speed_from_waypoints(predicted_speed_waypoints)
    values["model_target_speed_mps"] = float(0.0 if model_target_speed_mps is None else model_target_speed_mps)

    speed_summary = _path_summary(speed_points)
    route_summary = _path_summary(route_points)
    speed_summary_names = {
        "count": "speed_wp_count",
        "path_length_m": "speed_wp_path_length_m",
        "mean_step_m": "speed_wp_mean_step_m",
        "final_x_forward": "speed_wp_final_x_forward",
        "final_y_right": "speed_wp_final_y_right",
        "mean_abs_y_right": "speed_wp_mean_abs_y_right",
    }
    route_summary_names = {
        "count": "route_wp_count",
        "path_length_m": "route_path_length_m",
        "mean_step_m": "route_mean_step_m",
        "final_x_forward": "route_final_x_forward",
        "final_y_right": "route_final_y_right",
        "mean_abs_y_right": "route_mean_abs_y_right",
        "max_abs_y_right": "route_max_abs_y_right",
    }
    for key, name in speed_summary_names.items():
        if name in values:
            values[name] = float(speed_summary[key])
    for key, name in route_summary_names.items():
        if name in values:
            values[name] = float(route_summary[key])

    for idx, point in enumerate(speed_points):
        values[f"speed_wp_{idx}_x_forward"] = float(point[0])
        values[f"speed_wp_{idx}_y_right"] = float(point[1])
    for idx, point in enumerate(route_points):
        values[f"route_wp_{idx}_x_forward"] = float(point[0])
        values[f"route_wp_{idx}_y_right"] = float(point[1])

    return np.asarray([values[name] for name in feature_names], dtype=np.float32)


def annotation_features(sample: dict[str, Any]) -> np.ndarray:
    ego_state = sample.get("ego_state") or {}
    control = sample.get("control") or {}
    speed = float(ego_state.get("speed_mps", 0.0))
    return build_velocity_features(
        str(sample.get("command", "")),
        speed,
        float(control.get("steer", 0.0)),
        float(control.get("throttle", 0.0)),
        float(control.get("brake", 0.0)),
        baseline_speed_mps=float(sample.get("baseline_speed_mps", speed)),
        command_elapsed_sec=float(sample.get("command_elapsed_sec", 0.0)),
    )


def target_speed_from_annotation(sample: dict[str, Any], future_dt: float = 0.4, waypoint_gap: int = 2) -> float | None:
    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) <= waypoint_gap:
        return None
    try:
        p0 = np.asarray(trajectory[0][:2], dtype=np.float32)
        p1 = np.asarray(trajectory[waypoint_gap][:2], dtype=np.float32)
    except Exception:
        return None
    dt = max(1e-6, float(future_dt) * float(waypoint_gap))
    speed = float(np.linalg.norm(p1 - p0) / dt)
    if not math.isfinite(speed):
        return None
    return max(0.0, speed)


def load_annotations(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict annotations at {path}")
    return data


def split_annotation_path(data_root: pathlib.Path, split: str) -> pathlib.Path:
    direct = data_root / split / "annotations.json"
    if direct.exists():
        return direct
    fallback = data_root / "annotations.json"
    if split == "train" and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find annotations for split {split!r} under {data_root}")


@dataclass
class VelocityHeadConfig:
    feature_version: str
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: float
    target_std: float
    commands: list[str]
    future_dt: float
    waypoint_gap: int
    min_speed_mps: float
    max_speed_mps: float


class VelocityHeadRuntime:
    def __init__(self, checkpoint: pathlib.Path, device: str = "cpu"):
        import torch

        self.torch = torch
        payload = torch.load(checkpoint, map_location=device)
        config = payload["model_config"]
        self.config = VelocityHeadConfig(
            feature_version=str(config.get("feature_version", "task1_velocity_v2")),
            feature_names=list(config["feature_names"]),
            feature_mean=[float(x) for x in config["feature_mean"]],
            feature_std=[float(x) for x in config["feature_std"]],
            target_mean=float(config["target_mean"]),
            target_std=float(config["target_std"]),
            commands=list(config["commands"]),
            future_dt=float(config["future_dt"]),
            waypoint_gap=int(config["waypoint_gap"]),
            min_speed_mps=float(config.get("min_speed_mps", 0.0)),
            max_speed_mps=float(config.get("max_speed_mps", 8.0)),
        )
        self.model = build_mlp(len(self.config.feature_names))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.feature_mean = torch.tensor(self.config.feature_mean, dtype=torch.float32, device=device)
        self.feature_std = torch.tensor(self.config.feature_std, dtype=torch.float32, device=device).clamp_min(1e-6)
        self.feature_names = list(self.config.feature_names)

    def predict(
        self,
        command: str,
        current_speed_mps: float,
        previous_steer: float = 0.0,
        previous_throttle: float = 0.0,
        previous_brake: float = 0.0,
        baseline_speed_mps: float | None = None,
        command_elapsed_sec: float = 0.0,
        predicted_speed_waypoints: Any = None,
        predicted_route: Any = None,
        model_target_speed_mps: float | None = None,
    ) -> float:
        with self.torch.no_grad():
            if self.config.feature_version == SIMLINGO_SPEED_FEATURE_VERSION:
                features_np = build_simlingo_velocity_features(
                    command,
                    current_speed_mps,
                    previous_steer=previous_steer,
                    previous_throttle=previous_throttle,
                    previous_brake=previous_brake,
                    baseline_speed_mps=baseline_speed_mps,
                    command_elapsed_sec=command_elapsed_sec,
                    predicted_speed_waypoints=predicted_speed_waypoints,
                    predicted_route=predicted_route,
                    model_target_speed_mps=model_target_speed_mps,
                    feature_names=self.feature_names,
                )
            else:
                features_np = build_velocity_features(
                    command,
                    current_speed_mps,
                    previous_steer,
                    previous_throttle,
                    previous_brake,
                    baseline_speed_mps=baseline_speed_mps,
                    command_elapsed_sec=command_elapsed_sec,
                    feature_names=self.feature_names,
                )
            features = self.torch.tensor(features_np, dtype=self.torch.float32, device=self.device)
            features = (features - self.feature_mean) / self.feature_std
            pred_norm = self.model(features.unsqueeze(0)).squeeze().item()
            speed = pred_norm * self.config.target_std + self.config.target_mean
            return float(np.clip(speed, self.config.min_speed_mps, self.config.max_speed_mps))


def build_mlp(input_dim: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
