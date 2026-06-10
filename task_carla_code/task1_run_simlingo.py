#!/usr/bin/env python3
"""Run Task 1 CARLA trials with the released SimLingo agent."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import queue
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import cv2
import numpy as np


BASIC_COMMANDS = ("turn left", "turn right", "straight", "stop", "speed up", "slow down")
SPEED_EVAL_COMMANDS = ("stop", "speed up", "slow down")


class FallbackRoadOption(IntEnum):
    """CARLA RoadOption values used by leaderboard agents."""

    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


@dataclass
class TrialResult:
    command: str
    trial_index: int
    success: bool
    reason: str
    duration_sec: float
    frames: int
    distance_m: float
    heading_delta_deg: float
    max_speed_mps: float
    min_speed_mps: float
    final_speed_mps: float
    collision_count: int
    offroad_frames: int
    output_dir: str


class IdentityRoutePlanner:
    """RoutePlanner wrapper that treats GPS input as CARLA world coordinates."""

    def __init__(self, route_planner_cls: type, min_distance: float, max_distance: float):
        self.inner = route_planner_cls(min_distance, max_distance, 0.0, 0.0)

    def convert_gps_to_carla(self, gps: Any) -> np.ndarray:
        arr = np.asarray(gps, dtype=np.float32)
        return np.array([arr[0], arr[1], arr[2] if arr.shape[0] > 2 else 0.0], dtype=np.float32)

    def set_route(self, global_plan: list[tuple[Any, Any]], gps: bool = False, carla_map: Any = None) -> None:
        self.inner.set_route(global_plan, gps=False, carla_map=carla_map)

    def run_step(self, gps: Any) -> Any:
        return self.inner.run_step(np.asarray(gps, dtype=np.float32))

    @property
    def route(self) -> Any:
        return self.inner.route

    @property
    def is_last(self) -> bool:
        return bool(getattr(self.inner, "is_last", False))


class SimLingoTask1Policy:
    """Thin wrapper around SimLingo's official LingoAgent."""

    def __init__(
        self,
        checkpoint: pathlib.Path,
        output_dir: pathlib.Path,
        input_mode: str = "target_point_command",
        controller_inference_mode: bool = True,
        speed_control_mode: str = "route",
        route_speed_scale: float = 0.60,
        max_throttle: float = 0.45,
        brake_ratio: float = 1.35,
        clip_delta: float = 0.50,
    ):
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.input_mode = input_mode
        self.controller_inference_mode = bool(controller_inference_mode)
        self.speed_control_mode = speed_control_mode
        self.route_speed_scale = float(route_speed_scale)
        self.max_throttle = float(max_throttle)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.agent = None
        self.agent_module = None
        self.RoadOption = None

    def load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint}")

        hydra_config = self.checkpoint.parent.parent.parent / ".hydra" / "config.yaml"
        if not hydra_config.exists():
            raise FileNotFoundError(
                "Could not find SimLingo Hydra config next to checkpoint. "
                f"Expected: {hydra_config}"
            )

        try:
            from agents.navigation.local_planner import RoadOption  # type: ignore
        except Exception:
            RoadOption = FallbackRoadOption
        self.RoadOption = RoadOption

        import team_code.agent_simlingo as agent_module
        from team_code.agent_simlingo import LingoAgent
        from team_code.nav_planner import RoutePlanner

        agent_module.USE_UKF = False
        self.agent_module = agent_module
        output_root = self.output_dir / "agent_debug"
        save_path = os.environ.get("SAVE_PATH", str(output_root))
        os.environ["SAVE_PATH"] = save_path.rstrip("/") + "/"
        os.environ.setdefault("ROUTES", "data/benchmarks/task1/task1.xml")

        class Task1LingoAgent(LingoAgent):  # type: ignore[misc, valid-type]
            def __init__(inner_self) -> None:  # noqa: ANN001
                from leaderboard.envs.sensor_interface import SensorInterface

                inner_self.track = agent_module.autonomous_agent.Track.SENSORS
                inner_self._global_plan = None
                inner_self._global_plan_world_coord = None
                inner_self.sensor_interface = SensorInterface()
                inner_self.wallclock_t0 = None
                inner_self.hero_actor = None

            def _init(inner_self) -> None:  # noqa: ANN001
                inner_self.lat_ref, inner_self.lon_ref = 0.0, 0.0
                inner_self._route_planner = IdentityRoutePlanner(
                    RoutePlanner,
                    inner_self.route_planner_min_distance,
                    inner_self.route_planner_max_distance,
                )
                plan = getattr(inner_self, "_task1_global_plan_world_coord", None)
                if not plan:
                    plan = inner_self._global_plan_world_coord
                if not plan:
                    raise RuntimeError("SimLingo Task1 agent has no global plan.")
                inner_self._route_planner.set_route(plan, gps=False)
                inner_self.initialized = True
                inner_self.metric_info = {}

            def control_pid(inner_self, route_waypoints, velocity, speed_waypoints):  # noqa: ANN001
                inner_self.task1_last_pred_route = _tensor_to_list(route_waypoints)
                inner_self.task1_last_pred_speed_waypoints = _tensor_to_list(speed_waypoints)
                inner_self.task1_speed_control_mode = getattr(inner_self, "task1_speed_control_mode", "model")
                inner_self.task1_last_effective_speed_waypoints = _tensor_to_list(speed_waypoints)
                if inner_self.task1_speed_control_mode == "route":
                    speed_count = int(speed_waypoints.size(1))
                    speed_scale = float(getattr(inner_self, "task1_route_speed_scale", 1.0))
                    effective_speed_waypoints = (route_waypoints[:, :speed_count, :] * speed_scale).contiguous()
                    inner_self.task1_last_effective_speed_waypoints = _tensor_to_list(effective_speed_waypoints)
                    return super(Task1LingoAgent, inner_self).control_pid(
                        route_waypoints,
                        velocity,
                        effective_speed_waypoints,
                    )
                return super(Task1LingoAgent, inner_self).control_pid(route_waypoints, velocity, speed_waypoints)

        self.agent = Task1LingoAgent()
        self.agent.setup(str(self.checkpoint), route_index="task1")
        if hasattr(self.agent, "turn_controller") and hasattr(self.agent.turn_controller, "inference_mode"):
            self.agent.turn_controller.inference_mode = self.controller_inference_mode
        self.agent.task1_speed_control_mode = self.speed_control_mode
        self.agent.task1_route_speed_scale = self.route_speed_scale
        self.agent.config.clip_throttle = self.max_throttle
        self.agent.config.brake_ratio = self.brake_ratio
        self.agent.config.clip_delta = self.clip_delta
        if self.input_mode == "strict-command":
            self.agent.task1_strict_command_only = True
            self.agent.config.eval_route_as = "command"
        elif self.input_mode == "command":
            self.agent.task1_strict_command_only = False
            self.agent.config.eval_route_as = "command"
        else:
            self.agent.task1_strict_command_only = False
            self.agent.config.eval_route_as = "target_point_command"
        print(f"Using SimLingo route prompt mode: {self.agent.config.eval_route_as}", flush=True)
        print(f"Using Task 1 model input mode: {self.input_mode}", flush=True)
        print(f"Using SimLingo lateral controller inference mode: {self.controller_inference_mode}", flush=True)
        print(f"Using Task 1 speed control mode: {self.speed_control_mode}", flush=True)
        print(
            "Using Task 1 longitudinal caps: "
            f"route_speed_scale={self.route_speed_scale}, "
            f"max_throttle={self.max_throttle}, "
            f"brake_ratio={self.brake_ratio}, "
            f"clip_delta={self.clip_delta}",
            flush=True,
        )

    def set_route(self, route_plan: list[tuple[Any, Any]], command: str) -> None:
        if self.agent is None:
            raise RuntimeError("SimLingo agent is not loaded.")
        self.agent._task1_global_plan_world_coord = route_plan
        self.agent._global_plan_world_coord = route_plan
        self.agent._global_plan = route_plan
        self.agent.initialized = False
        self.agent.commands.clear()
        self.agent.commands.append(4)
        self.agent.commands.append(4)
        self.agent.target_point_prev = np.array([1e5, 1e5, 1e5], dtype=np.float32)
        self.agent.last_command = -1
        self.agent.last_command_tmp = -1
        self.agent.step = -1
        self.agent.custom_prompt = command_prompt(command)
        self.agent.user_flag = task1_user_flag(command) if self.input_mode == "strict-command" and self.agent.custom_prompt else None
        self.agent.task1_last_pred_route = None
        self.agent.task1_last_pred_speed_waypoints = None

    def set_hero_actor(self, vehicle: Any) -> None:
        if self.agent is None:
            raise RuntimeError("SimLingo agent is not loaded.")
        self.agent.hero_actor = vehicle

    def set_active_command(self, command: str | None) -> None:
        if self.agent is None:
            raise RuntimeError("SimLingo agent is not loaded.")
        self.agent.custom_prompt = command_prompt(command) if command else None
        self.agent.user_flag = task1_user_flag(command) if self.input_mode == "strict-command" and self.agent.custom_prompt else None

    def run_step(self, input_data: dict[str, Any], timestamp: float) -> Any:
        if self.agent is None:
            raise RuntimeError("SimLingo agent is not loaded.")
        return self.agent.run_step(input_data, timestamp)

    def diagnostics(self) -> dict[str, Any]:
        if self.agent is None:
            return {}
        return {
            "prompt": getattr(self.agent, "prompt", None),
            "prompt_task": getattr(self.agent, "prompt_tp", None),
            "model_input_mode": self.input_mode,
            "strict_command_only": bool(getattr(self.agent, "task1_strict_command_only", False)),
            "controller_inference_mode": bool(
                getattr(getattr(self.agent, "turn_controller", None), "inference_mode", False)
            ),
            "speed_control_mode": getattr(self.agent, "task1_speed_control_mode", None),
            "route_speed_scale": float(getattr(self.agent, "task1_route_speed_scale", 1.0)),
            "max_throttle": float(getattr(self.agent.config, "clip_throttle", 0.0)),
            "brake_ratio": float(getattr(self.agent.config, "brake_ratio", 0.0)),
            "clip_delta": float(getattr(self.agent.config, "clip_delta", 0.0)),
            "predicted_route": getattr(self.agent, "task1_last_pred_route", None),
            "predicted_speed_waypoints": getattr(self.agent, "task1_last_pred_speed_waypoints", None),
            "effective_speed_waypoints": getattr(self.agent, "task1_last_effective_speed_waypoints", None),
            "target_points": _to_jsonable(getattr(self.agent, "target_points", None)),
            "route_command_history": [int(x) for x in list(getattr(self.agent, "commands", []))],
        }


class Task1SimLingoRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        import carla

        self.carla = carla
        host, port = load_connection(args.connection_env, args.host, args.port)
        print(f"Connecting to CARLA at {host}:{port} ...", flush=True)
        self.client = carla.Client(host, int(port))
        self.client.set_timeout(args.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        print(f"Connected. Map: {self.map.name}", flush=True)

        self.policy = SimLingoTask1Policy(
            args.checkpoint,
            self.output_dir,
            args.simlingo_input_mode,
            args.simlingo_controller_inference_mode,
            args.simlingo_speed_control_mode,
            args.task1_route_speed_scale,
            args.task1_max_throttle,
            args.task1_brake_ratio,
            args.task1_clip_delta,
        )
        self.policy.load()
        if self.policy.agent is not None:
            self.policy.agent.route_planner_min_distance = float(args.route_planner_min_distance)
            self.policy.agent.route_planner_max_distance = float(args.route_planner_max_distance)
        if self.policy.RoadOption is None:
            self.RoadOption = FallbackRoadOption
        else:
            self.RoadOption = self.policy.RoadOption
        self.used_spawn_locations: list[tuple[str, tuple[float, float]]] = []
        self.velocity_head = None
        if args.velocity_head_checkpoint is not None:
            from task_carla_code.task1_velocity_head import VelocityHeadRuntime

            self.velocity_head = VelocityHeadRuntime(args.velocity_head_checkpoint, device=args.velocity_head_device)
            print(f"Loaded Task 1 velocity head: {args.velocity_head_checkpoint}", flush=True)

    def run(self) -> int:
        if self.args.command:
            results = [self.run_trial(normalize_command(self.args.command), 0)]
        else:
            commands = list(BASIC_COMMANDS)
            results = []
            durations = parse_duration_overrides(self.args.eval_duration_overrides)
            base_duration = self.args.duration
            for command in commands:
                for trial in range(self.args.trials_per_command):
                    original_duration = self.args.duration
                    self.args.duration = durations.get(command, base_duration)
                    results.append(self.run_trial(command, trial))
                    self.args.duration = original_duration
        self.write_eval_summary(results)
        return 0 if all(r.success for r in results) else 1

    def predict_velocity_head_speed(
        self,
        command: str,
        vehicle: Any,
        previous_control: Any | None,
        baseline_speed_mps: float | None = None,
        command_elapsed_sec: float = 0.0,
    ) -> float | None:
        if self.velocity_head is None:
            return None
        previous_steer = float(previous_control.steer) if previous_control is not None else 0.0
        previous_throttle = float(previous_control.throttle) if previous_control is not None else 0.0
        previous_brake = float(previous_control.brake) if previous_control is not None else 0.0
        return self.velocity_head.predict(
            command=command,
            current_speed_mps=speed_mps(vehicle),
            previous_steer=previous_steer,
            previous_throttle=previous_throttle,
            previous_brake=previous_brake,
            baseline_speed_mps=baseline_speed_mps,
            command_elapsed_sec=command_elapsed_sec,
        )

    def run_trial(self, command: str, trial_index: int) -> TrialResult:
        run_dir = self.output_dir / safe_name(command) / f"trial_{trial_index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ticks_path = run_dir / "ticks.jsonl"
        video_path = run_dir / f"{self.args.camera_view}.mp4"
        scenario_path = run_dir / "scenario.json"

        original_settings = self.world.get_settings()
        actors: list[Any] = []
        vehicle = None
        collision_events: list[dict[str, Any]] = []
        frames_for_video: list[np.ndarray] = []
        image_queue: queue.Queue[Any] = queue.Queue()
        video_queue: queue.Queue[Any] = queue.Queue()
        video_written = False

        try:
            self.apply_sync_settings()
            vehicle, route_plan, scenario = self.spawn_vehicle_and_route(command, trial_index)
            actors.append(vehicle)
            scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")

            collision_sensor = self.spawn_collision_sensor(vehicle, collision_events)
            actors.append(collision_sensor)

            rgb_sensor = self.spawn_camera(vehicle, self.simlingo_camera_blueprint(), self.simlingo_camera_transform())
            rgb_sensor.listen(lambda image: image_queue.put(image))
            actors.append(rgb_sensor)

            if self.args.camera_view == "front":
                record_sensor = None
            else:
                record_sensor = self.spawn_camera(vehicle, self.video_camera_blueprint(), self.video_camera_transform())
                record_sensor.listen(lambda image: video_queue.put(image))
                actors.append(record_sensor)

            self.policy.set_route(route_plan, command)
            self.policy.set_hero_actor(vehicle)
            self.drain(image_queue)
            self.drain(video_queue)

            rgb_image, record_image = self.wait_for_sensor_start(
                image_queue=image_queue,
                video_queue=video_queue if record_sensor is not None else None,
            )
            if record_sensor is not None:
                frames_for_video.append(carla_image_to_bgr(record_image))
            else:
                frames_for_video.append(resize_frame(carla_image_to_bgr(rgb_image), self.args.width, self.args.height))

            prime_policy_calls = 0
            prime_policy_latency_sec = 0.0
            if self.args.prime_policy_before_start:
                prime_active_command = None if command in SPEED_EVAL_COMMANDS and command_warmup_sec(command, self.args) > 0.0 else command
                for prime_index in range(max(1, int(self.args.prime_policy_steps))):
                    self.policy.set_active_command(prime_active_command)
                    input_data = self.build_input_data(vehicle, rgb_image)
                    t0 = time.time()
                    _ = self.policy.run_step(input_data, -float(prime_index + 1) / float(self.args.sim_fps))
                    prime_policy_latency_sec += time.time() - t0
                    prime_policy_calls += 1
                    rgb_image, record_image = self.wait_for_sensor_frame(
                        image_queue=image_queue,
                        video_queue=video_queue if record_sensor is not None else None,
                    )

            start_transform = vehicle.get_transform()
            start_location = start_transform.location
            start_yaw = start_transform.rotation.yaw
            speeds: list[float] = []
            baseline_speeds: list[float] = []
            post_command_speeds: list[float] = []
            offroad_frames = 0
            success_hold = 0
            reached_success_step: int | None = None
            termination_reason: str | None = None
            hold_sec = self.args.speed_hold_sec if command in SPEED_EVAL_COMMANDS else self.args.success_hold_sec
            min_required_hold = max(1, int(round(hold_sec * self.args.sim_fps)))
            warmup_sec = command_warmup_sec(command, self.args)
            if self.args.simlingo_input_mode == "strict-command" and command not in SPEED_EVAL_COMMANDS:
                warmup_sec = 0.0
            command_start_step = int(round(warmup_sec * self.args.sim_fps))

            total_steps = int(round(self.args.duration * self.args.sim_fps))
            simlingo_policy_hz = float(self.args.policy_hz)
            policy_interval = max(1, int(round(self.args.sim_fps / max(0.1, simlingo_policy_hz))))
            velocity_head_interval = max(1, int(round(self.args.sim_fps / max(0.1, float(self.args.velocity_head_hz)))))
            velocity_head_commands = set(parse_command_sequence(self.args.velocity_head_commands))
            control = self.carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            previous_control = None
            policy_latency_sec = 0.0
            fresh_policy = False
            policy_calls = 0
            velocity_head_calls = 0
            velocity_head_fresh = False
            map_world_route = route_plan_to_world_points(route_plan)
            cached_world_route = (
                map_world_route.copy()
                if command in SPEED_EVAL_COMMANDS and self.args.speed_command_route_source == "map"
                else np.zeros((0, 2), dtype=np.float32)
            )
            cached_route_created_step: int | None = None
            cached_model_target_speed_mps: float | None = None
            velocity_head_target_speed_mps: float | None = None
            cached_follow_diag: dict[str, Any] = {}
            cached_route_update_diag: dict[str, Any] = {}
            rate_limit_diag: dict[str, Any] = {}
            with ticks_path.open("w", encoding="utf-8") as ticks_file:
                for step in range(total_steps):
                    timestamp = step / float(self.args.sim_fps)
                    active_command = command if timestamp >= warmup_sec else None
                    execution_command = active_command if active_command is not None else "straight"
                    cached_route_ahead_points = count_cached_route_ahead_points(cached_world_route, vehicle.get_transform())
                    route_exhausted_refresh = bool(
                        self.args.refresh_policy_on_route_exhaustion
                        and self.args.simlingo_execution_mode == "cached-route-follower"
                        and active_command is not None
                        and cached_world_route.size > 0
                        and cached_route_ahead_points <= self.args.min_cached_route_ahead_points
                    )
                    command_just_started = bool(
                        self.args.force_policy_on_command_change
                        and command in SPEED_EVAL_COMMANDS
                        and warmup_sec > 0.0
                        and step == command_start_step
                    )
                    self.policy.set_active_command(active_command)
                    fresh_policy = step == 0 or command_just_started or route_exhausted_refresh or (step % policy_interval == 0)
                    velocity_head_fresh = bool(
                        self.velocity_head is not None
                        and execution_command in velocity_head_commands
                        and (step == 0 or command_just_started or step % velocity_head_interval == 0)
                    )
                    if fresh_policy:
                        policy_calls += 1
                        input_data = self.build_input_data(vehicle, rgb_image)
                        t0 = time.time()
                        if self.args.progress:
                            print(
                                f"[{command} #{trial_index}] policy_call={policy_calls} "
                                f"step={step}/{total_steps} t={timestamp:.2f}s",
                                flush=True,
                            )
                        control = self.policy.run_step(input_data, timestamp)
                        policy_latency_sec = time.time() - t0
                        diag_after_policy = self.policy.diagnostics()
                        predicted_route = extract_xy_points(diag_after_policy.get("predicted_route"))
                        cached_model_target_speed_mps = estimate_model_target_speed_mps(
                            diag_after_policy.get("predicted_speed_waypoints")
                        )
                        use_model_route = not (
                            command in SPEED_EVAL_COMMANDS
                            and self.args.speed_command_route_source == "map"
                        )
                        if predicted_route.size > 0 and use_model_route:
                            route_transform = vehicle.get_transform()
                            old_local_route = world_route_to_local(cached_world_route, route_transform)
                            predicted_route_diag = route_lateral_diagnostics(predicted_route)
                            if (
                                self.args.simlingo_execution_mode == "cached-route-follower"
                                and cached_world_route.size > 0
                                and float(self.args.cached_route_blend_new_weight) < 0.999
                                and not command_just_started
                            ):
                                cached_local_route = blend_local_routes(
                                    old_local_route,
                                    predicted_route,
                                    self.args.cached_route_blend_new_weight,
                                )
                                route_blended = True
                            else:
                                cached_local_route = predicted_route
                                route_blended = False
                            cached_world_route = local_route_to_world(cached_local_route, route_transform)
                            cached_route_created_step = step
                            cached_route_update_diag = {
                                "route_blended": route_blended,
                                "cached_route_blend_new_weight": float(self.args.cached_route_blend_new_weight),
                                "new_route_first": predicted_route[0].tolist(),
                                "new_route_last": predicted_route[-1].tolist(),
                                "new_route_lateral_diag": predicted_route_diag,
                                "model_target_speed_mps": cached_model_target_speed_mps,
                                "old_route_points_ahead": int(np.sum(old_local_route[:, 0] > 0.25))
                                if old_local_route.size
                                else 0,
                            }
                        elif predicted_route.size > 0:
                            cached_route_update_diag = {
                                "route_source": "map",
                                "model_route_ignored_for_speed_command": True,
                                "model_target_speed_mps": cached_model_target_speed_mps,
                                "new_route_lateral_diag": route_lateral_diagnostics(predicted_route),
                                "map_route_points": int(len(map_world_route)),
                            }
                        if self.args.progress:
                            print(
                                f"[{command} #{trial_index}] policy_call={policy_calls} "
                                f"latency={policy_latency_sec:.3f}s "
                                f"control=({float(control.steer):.3f},"
                                f"{float(control.throttle):.3f},{float(control.brake):.3f})",
                                flush=True,
                            )
                    if self.args.simlingo_execution_mode == "cached-route-follower":
                        if velocity_head_fresh:
                            velocity_head_calls += 1
                            baseline_count = max(
                                1,
                                int(round(float(self.args.speed_baseline_window_sec) * float(self.args.sim_fps))),
                            )
                            baseline_window = baseline_speeds[-baseline_count:] if baseline_speeds else []
                            velocity_baseline_speed_mps = (
                                float(sum(baseline_window) / len(baseline_window))
                                if baseline_window
                                else speed_mps(vehicle)
                            )
                            # The stop label is absolute zero speed. Feeding the warmup
                            # baseline after the vehicle has already slowed is out of
                            # distribution for the current velocity-head training data and
                            # can make the head drift back toward a moving target.
                            head_baseline_speed_mps = (
                                speed_mps(vehicle)
                                if execution_command == "stop"
                                else velocity_baseline_speed_mps
                            )
                            # v7 semantic velocity-head training annotations do not carry
                            # real command elapsed time, so this feature was constant zero
                            # during training. Keep runtime in-distribution until we collect
                            # elapsed-aware labels.
                            velocity_command_elapsed_sec = 0.0
                            velocity_head_target_speed_mps = self.predict_velocity_head_speed(
                                command=execution_command,
                                vehicle=vehicle,
                                previous_control=previous_control,
                                baseline_speed_mps=head_baseline_speed_mps,
                                command_elapsed_sec=velocity_command_elapsed_sec,
                            )
                        control, cached_follow_diag = follow_cached_route(
                            vehicle,
                            cached_world_route,
                            execution_command,
                            self.args,
                            self.carla,
                            cached_model_target_speed_mps,
                            velocity_head_target_speed_mps,
                        )
                        control, rate_limit_diag = rate_limit_control(control, previous_control, self.args, self.carla)
                    vehicle.apply_control(control)
                    previous_control = control

                    rgb_image, record_image = self.wait_for_sensor_frame(
                        image_queue=image_queue,
                        video_queue=video_queue if record_sensor is not None else None,
                    )
                    if record_sensor is not None:
                        frame = carla_image_to_bgr(record_image)
                    else:
                        frame = resize_frame(carla_image_to_bgr(rgb_image), self.args.width, self.args.height)
                    diag = self.policy.diagnostics()
                    if self.args.simlingo_execution_mode == "cached-route-follower":
                        diag["cached_follow_route"] = cached_follow_diag.get("cached_local_route", [])
                        diag["cached_target_point"] = cached_follow_diag.get("cached_target_point")

                    speed = speed_mps(vehicle)
                    speeds.append(speed)
                    if timestamp < warmup_sec:
                        baseline_speeds.append(speed)
                    else:
                        post_command_speeds.append(speed)
                    transform = vehicle.get_transform()
                    offroad = not is_vehicle_on_driving_lane(self.map, transform.location, self.carla)
                    if offroad:
                        offroad_frames += 1
                    target_road_status = self.target_road_status(
                        command,
                        vehicle,
                        scenario,
                        start_yaw,
                        transform.rotation.yaw,
                        start_location,
                    )
                    diag["target_road_reached"] = bool(target_road_status.get("reached"))
                    diag["target_road_reason"] = target_road_status.get("reason")
                    diag["command"] = command
                    diag["speed_mps"] = float(speed)
                    diag["ego_location"] = location_dict(transform.location)
                    diag["ego_heading_deg"] = float(transform.rotation.yaw)
                    diag["target_road_point"] = scenario.get("target_road_point")
                    diag["target_road_heading_deg"] = scenario.get("target_road_heading_deg")
                    if self.args.overlay_predictions:
                        draw_prediction_overlay(frame, diag, self.args)
                    if step % max(1, int(round(self.args.sim_fps / self.args.fps))) == 0:
                        frames_for_video.append(frame)
                    target_road_reached = target_road_status["reached"]
                    status = self.evaluate_instant_status(
                        command=command,
                        vehicle=vehicle,
                        start_yaw=start_yaw,
                        start_location=start_location,
                        speeds=speeds,
                        baseline_speeds=baseline_speeds,
                        post_command_speeds=post_command_speeds,
                        offroad_frames=offroad_frames,
                        collision_count=len(collision_events),
                    )
                    target_road_command = command in ("turn left", "turn right") or (
                        command == "straight" and self.args.straight_require_junction
                    )
                    route_safety_ok = (
                        len(collision_events) <= self.args.max_collision_events
                        and (self.args.allow_offroad_success or offroad_frames <= self.args.max_offroad_frames)
                    )
                    completion_met = (
                        bool(target_road_reached and route_safety_ok)
                        if target_road_command
                        else bool(status["condition_met"])
                    )
                    if completion_met:
                        success_hold += 1
                        if success_hold >= min_required_hold and reached_success_step is None:
                            reached_success_step = step
                            termination_reason = "target_road_held" if target_road_command else "success_condition"
                    else:
                        success_hold = 0

                    tick = {
                        "step": step,
                        "time_sec": timestamp,
                        "command": command,
                        "active_command": active_command,
                        "execution_command": execution_command,
                        "speed_test_phase": "command" if active_command is not None else "baseline",
                        "warmup_sec": warmup_sec,
                        "command_start_step": command_start_step,
                        "command_just_started": command_just_started,
                        "route_exhausted_refresh": route_exhausted_refresh,
                        "cached_route_ahead_points": cached_route_ahead_points,
                        "speed_baseline_window_sec": float(self.args.speed_baseline_window_sec),
                        "speed_hold_sec": float(hold_sec),
                        "prompt": diag.get("prompt"),
                        "prompt_task": diag.get("prompt_task"),
                        "model_input_mode": diag.get("model_input_mode"),
                        "execution_mode": self.args.simlingo_execution_mode,
                        "strict_command_only": diag.get("strict_command_only"),
                        "controller_inference_mode": diag.get("controller_inference_mode"),
                        "speed_control_mode": diag.get("speed_control_mode"),
                        "force_policy_on_command_change": bool(self.args.force_policy_on_command_change),
                        "route_speed_scale": diag.get("route_speed_scale"),
                        "max_throttle": diag.get("max_throttle"),
                        "brake_ratio": diag.get("brake_ratio"),
                        "clip_delta": diag.get("clip_delta"),
                        "predicted_route": diag.get("predicted_route"),
                        "predicted_speed_waypoints": diag.get("predicted_speed_waypoints"),
                        "effective_speed_waypoints": diag.get("effective_speed_waypoints"),
                        "cached_follow_route": diag.get("cached_follow_route"),
                        "cached_target_point": diag.get("cached_target_point"),
                        "cached_route_age_steps": None
                        if cached_route_created_step is None
                        else int(step - cached_route_created_step),
                        "cached_route_points": int(len(cached_world_route)),
                        "cached_follow": cached_follow_diag,
                        "velocity_head_target_speed_mps": velocity_head_target_speed_mps,
                        "cached_route_update": cached_route_update_diag if fresh_policy else None,
                        "control_rate_limit": rate_limit_diag,
                        "target_points": diag.get("target_points"),
                        "route_command_history": diag.get("route_command_history"),
                        "fresh_policy": bool(fresh_policy),
                        "policy_hz": float(simlingo_policy_hz),
                        "policy_interval_steps": int(policy_interval),
                        "velocity_head_fresh": bool(velocity_head_fresh),
                        "velocity_head_hz": float(self.args.velocity_head_hz),
                        "velocity_head_interval_steps": int(velocity_head_interval),
                        "velocity_head_calls": int(velocity_head_calls),
                        "velocity_head_commands": sorted(velocity_head_commands),
                        "policy_latency_sec": float(policy_latency_sec if fresh_policy else 0.0),
                        "prime_policy_calls": int(prime_policy_calls),
                        "prime_policy_latency_sec": float(prime_policy_latency_sec),
                        "steer": float(control.steer),
                        "throttle": float(control.throttle),
                        "brake": float(control.brake),
                        "speed_mps": float(speed),
                        "location": location_dict(transform.location),
                        "heading_deg": float(transform.rotation.yaw),
                        "heading_delta_deg": angle_delta_deg(start_yaw, transform.rotation.yaw),
                        "offroad": bool(offroad),
                        "target_road_id": scenario.get("target_road_id"),
                        "target_road_lane_id": scenario.get("target_lane_id"),
                        "target_road_point": scenario.get("target_road_point"),
                        "target_road_distance_m": target_road_status.get("distance_m"),
                        "target_road_heading_error_deg": target_road_status.get("heading_error_deg"),
                        "target_road_lane_center_distance_m": target_road_status.get("lane_center_distance_m"),
                        "straight_branch_progress_m": target_road_status.get("straight_branch_progress_m"),
                        "straight_branch_lateral_m": target_road_status.get("straight_branch_lateral_m"),
                        "target_road_same_road": target_road_status.get("same_road"),
                        "target_road_reached": bool(target_road_reached),
                        "target_road_reason": target_road_status.get("reason"),
                        "target_road_body_on_target_road": target_road_status.get("body_on_target_road"),
                        "target_road_body_target_samples": target_road_status.get("body_target_road_samples"),
                        "target_road_body_total_samples": target_road_status.get("body_total_samples"),
                        "target_road_body_required_samples": target_road_status.get("body_required_samples"),
                        "target_road_body_samples": target_road_status.get("body_samples"),
                        "collision_count": len(collision_events),
                        "instant_success": status,
                        "termination_reason": termination_reason,
                    }
                    ticks_file.write(json.dumps(tick) + "\n")
                    ticks_file.flush()

                    if self.args.stop_on_offroad and offroad_frames > self.args.max_offroad_frames:
                        termination_reason = "offroad"
                        break
                    if self.args.stop_on_success and reached_success_step is not None:
                        break

            write_video(video_path, frames_for_video, self.args.fps)
            video_written = True
            result = self.build_result(
                command=command,
                trial_index=trial_index,
                run_dir=run_dir,
                vehicle=vehicle,
                start_location=start_location,
                start_yaw=start_yaw,
                speeds=speeds,
                post_command_speeds=post_command_speeds,
                baseline_speeds=baseline_speeds,
                offroad_frames=offroad_frames,
                collision_events=collision_events,
                frames=len(frames_for_video),
                reached_success_step=reached_success_step,
                scenario=scenario,
            )
            (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "controller": "simlingo_lingoagent",
                        "executable_action_schema": ["steer", "throttle", "brake"],
                        "checkpoint": str(self.args.checkpoint),
                        "model_input_mode": self.args.simlingo_input_mode,
                        "execution_mode": self.args.simlingo_execution_mode,
                        "policy_hz": self.args.policy_hz,
                        "velocity_head_hz": self.args.velocity_head_hz,
                        "velocity_head_commands": self.args.velocity_head_commands,
                        "route_tensors_in_model_input": self.args.simlingo_input_mode != "strict-command",
                        "controller_inference_mode": self.args.simlingo_controller_inference_mode,
                        "speed_control_mode": self.args.simlingo_speed_control_mode,
                        "command_warmup_sec": warmup_sec,
                        "speed_baseline_window_sec": self.args.speed_baseline_window_sec,
                        "speed_hold_sec": hold_sec,
                        "speed_delta_threshold": self.args.speed_delta_threshold,
                        "stop_speed_threshold": self.args.stop_speed_threshold,
                        "stop_baseline_min_speed_mps": self.args.stop_baseline_min_speed_mps,
                        "slowdown_min_success_speed_mps": self.args.slowdown_min_success_speed_mps,
                        "force_policy_on_command_change": self.args.force_policy_on_command_change,
                        "refresh_policy_on_route_exhaustion": self.args.refresh_policy_on_route_exhaustion,
                        "min_cached_route_ahead_points": self.args.min_cached_route_ahead_points,
                        "route_speed_scale": self.args.task1_route_speed_scale,
                        "max_throttle": self.args.task1_max_throttle,
                        "brake_ratio": self.args.task1_brake_ratio,
                        "clip_delta": self.args.task1_clip_delta,
                        "map": self.map.name,
                        "command": command,
                        "trial_index": trial_index,
                        "success": result.success,
                        "reason": result.reason,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[{command} #{trial_index}] success={result.success} reason={result.reason}", flush=True)
            return result
        finally:
            if not video_written:
                try:
                    write_video(video_path, frames_for_video, self.args.fps)
                except Exception:
                    pass
            for actor in reversed(actors):
                try:
                    if hasattr(actor, "stop"):
                        actor.stop()
                    actor.destroy()
                except Exception:
                    pass
            self.world.apply_settings(original_settings)

    def build_input_data(self, vehicle: Any, rgb_image: Any) -> dict[str, Any]:
        transform = vehicle.get_transform()
        loc = transform.location
        yaw_rad = math.radians(transform.rotation.yaw)
        leaderboard_compass = yaw_rad + math.pi / 2.0
        imu = np.zeros(7, dtype=np.float32)
        imu[-1] = float(leaderboard_compass)
        return {
            "rgb_0": (None, carla_image_to_bgra(rgb_image)),
            "gps": (None, np.array([loc.x, loc.y, loc.z], dtype=np.float32)),
            "imu": (None, imu),
            "speed": (None, {"speed": float(speed_mps(vehicle))}),
        }

    def build_result(
        self,
        command: str,
        trial_index: int,
        run_dir: pathlib.Path,
        vehicle: Any,
        start_location: Any,
        start_yaw: float,
        speeds: list[float],
        post_command_speeds: list[float],
        baseline_speeds: list[float],
        offroad_frames: int,
        collision_events: list[dict[str, Any]],
        frames: int,
        reached_success_step: int | None,
        scenario: dict[str, Any],
    ) -> TrialResult:
        transform = vehicle.get_transform()
        distance = distance_2d(start_location, transform.location)
        heading_delta = angle_delta_deg(start_yaw, transform.rotation.yaw)
        max_speed = max(speeds) if speeds else 0.0
        min_speed = min(post_command_speeds or speeds) if speeds else 0.0
        final_speed = speeds[-1] if speeds else 0.0
        status = self.evaluate_instant_status(
            command=command,
            vehicle=vehicle,
            start_yaw=start_yaw,
            start_location=start_location,
            speeds=speeds,
            baseline_speeds=baseline_speeds,
            post_command_speeds=post_command_speeds,
            offroad_frames=offroad_frames,
            collision_count=len(collision_events),
        )
        target_road_command = command in ("turn left", "turn right") or (
            command == "straight" and self.args.straight_require_junction
        )
        if target_road_command:
            target_status = self.target_road_status(
                command,
                vehicle,
                scenario,
                start_yaw,
                transform.rotation.yaw,
                start_location,
            )
            success = bool(reached_success_step is not None)
            reason = "ok" if success else str(target_status["reason"])
        else:
            success = bool(reached_success_step is not None or status["condition_met"])
            reason = "ok" if success else str(status["reason"])
        return TrialResult(
            command=command,
            trial_index=trial_index,
            success=success,
            reason=reason,
            duration_sec=float(self.args.duration),
            frames=int(frames),
            distance_m=float(distance),
            heading_delta_deg=float(heading_delta),
            max_speed_mps=float(max_speed),
            min_speed_mps=float(min_speed),
            final_speed_mps=float(final_speed),
            collision_count=len(collision_events),
            offroad_frames=int(offroad_frames),
            output_dir=str(run_dir),
        )

    def evaluate_instant_status(
        self,
        command: str,
        vehicle: Any,
        start_yaw: float,
        start_location: Any,
        speeds: list[float],
        baseline_speeds: list[float],
        post_command_speeds: list[float],
        offroad_frames: int,
        collision_count: int,
    ) -> dict[str, Any]:
        transform = vehicle.get_transform()
        speed_now = speeds[-1] if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        post_min = min(post_command_speeds) if post_command_speeds else speed_now
        post_max = max(post_command_speeds) if post_command_speeds else speed_now
        initial_speed = speeds[0] if speeds else 0.0
        baseline_count = max(1, int(round(self.args.speed_baseline_window_sec * self.args.sim_fps)))
        baseline_window = baseline_speeds[-baseline_count:] if baseline_speeds else []
        baseline_speed = float(sum(baseline_window) / len(baseline_window)) if baseline_window else initial_speed
        warmup_speed = baseline_speed
        heading_delta = angle_delta_deg(start_yaw, transform.rotation.yaw)
        distance = distance_2d(start_location, transform.location)
        lane_wp = self.map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        lane_heading_error = None
        lane_center_distance = None
        if lane_wp is not None:
            lane_heading_error = abs(angle_delta_deg(lane_wp.transform.rotation.yaw, transform.rotation.yaw))
            lane_center_distance = distance_2d(lane_wp.transform.location, transform.location)

        if collision_count > self.args.max_collision_events:
            return {"condition_met": False, "reason": f"collision_count={collision_count}"}
        if not self.args.allow_offroad_success and offroad_frames > self.args.max_offroad_frames:
            return {"condition_met": False, "reason": f"offroad_frames={offroad_frames}"}

        if command == "turn right":
            ok = (
                heading_delta >= self.args.turn_success_min_heading_deg
                and lane_heading_error is not None
                and lane_heading_error <= self.args.turn_success_max_lane_heading_error_deg
                and lane_center_distance is not None
                and lane_center_distance <= self.args.turn_success_max_lane_center_distance_m
            )
            reason = f"heading_delta={heading_delta:.2f}, lane_heading_error={lane_heading_error}, lane_center_distance={lane_center_distance}"
        elif command == "turn left":
            ok = (
                heading_delta <= -self.args.turn_success_min_heading_deg
                and lane_heading_error is not None
                and lane_heading_error <= self.args.turn_success_max_lane_heading_error_deg
                and lane_center_distance is not None
                and lane_center_distance <= self.args.turn_success_max_lane_center_distance_m
            )
            reason = f"heading_delta={heading_delta:.2f}, lane_heading_error={lane_heading_error}, lane_center_distance={lane_center_distance}"
        elif command == "straight":
            ok = distance >= self.args.straight_min_distance and abs(heading_delta) <= self.args.straight_max_heading
            reason = f"distance={distance:.2f}, heading_delta={heading_delta:.2f}"
        elif command == "stop":
            ok = (
                len(post_command_speeds) > 0
                and baseline_speed >= self.args.stop_baseline_min_speed_mps
                and speed_now <= self.args.stop_speed_threshold
            )
            reason = (
                f"baseline_speed={baseline_speed:.2f}, speed_now={speed_now:.2f}, "
                f"threshold={self.args.stop_speed_threshold:.2f}"
            )
        elif command == "speed up":
            ok = len(post_command_speeds) > 0 and speed_now >= baseline_speed + self.args.speed_delta_threshold
            reason = (
                f"baseline_speed={baseline_speed:.2f}, speed_now={speed_now:.2f}, "
                f"post_max={post_max:.2f}, required_delta={self.args.speed_delta_threshold:.2f}"
            )
        elif command == "slow down":
            ok = (
                len(post_command_speeds) > 0
                and speed_now <= baseline_speed - self.args.speed_delta_threshold
                and speed_now >= self.args.slowdown_min_success_speed_mps
            )
            reason = (
                f"baseline_speed={baseline_speed:.2f}, speed_now={speed_now:.2f}, "
                f"post_min={post_min:.2f}, required_delta={self.args.speed_delta_threshold:.2f}, "
                f"min_moving_speed={self.args.slowdown_min_success_speed_mps:.2f}"
            )
        else:
            ok = False
            reason = f"unsupported command {command}"
        return {
            "condition_met": bool(ok),
            "reason": reason,
            "heading_delta_deg": float(heading_delta),
            "distance_m": float(distance),
            "lane_heading_error_deg": None if lane_heading_error is None else float(lane_heading_error),
            "lane_center_distance_m": None if lane_center_distance is None else float(lane_center_distance),
            "speed_mps": float(speed_now),
            "max_speed_mps": float(max_speed),
            "initial_speed_mps": float(initial_speed),
            "warmup_speed_mps": float(warmup_speed),
            "baseline_speed_mps": float(baseline_speed),
            "post_command_min_speed_mps": float(post_min),
            "post_command_max_speed_mps": float(post_max),
            "baseline_sample_count": int(len(baseline_speeds)),
            "post_command_sample_count": int(len(post_command_speeds)),
        }

    def target_road_status(
        self,
        command: str,
        vehicle: Any,
        scenario: dict[str, Any],
        start_yaw: float,
        current_yaw: float,
        start_location: Any,
    ) -> dict[str, Any]:
        status = {
            "reached": False,
            "same_road": False,
            "distance_m": None,
            "heading_delta_deg": None,
            "heading_error_deg": None,
            "heading_axis_error_deg": None,
            "lane_center_distance_m": None,
            "distance_from_start_m": None,
            "straight_branch_progress_m": None,
            "straight_branch_lateral_m": None,
            "body_on_target_road": False,
            "body_target_road_samples": 0,
            "body_total_samples": 0,
            "body_required_samples": 0,
            "body_samples": [],
            "reason": "not_checked",
        }
        if command not in ("turn left", "turn right", "straight"):
            status["reason"] = "not_target_road_command"
            return status
        target_road_id = scenario.get("target_road_id")
        if target_road_id is None:
            status["reason"] = "missing_target_road"
            return status
        transform = vehicle.get_transform()
        location = transform.location
        status["distance_from_start_m"] = float(distance_2d(start_location, location))
        lane_wp = self.map.get_waypoint(
            location,
            project_to_road=False,
            lane_type=self.carla.LaneType.Driving,
        )
        if lane_wp is None:
            status["reason"] = "center_not_on_driving_lane"
            return status
        heading_delta = angle_delta_deg(start_yaw, current_yaw)
        status["heading_delta_deg"] = float(heading_delta)

        same_road = int(lane_wp.road_id) == int(target_road_id)
        status["same_road"] = bool(same_road)
        directed_heading_error = float(abs(angle_delta_deg(lane_wp.transform.rotation.yaw, current_yaw)))
        status["heading_error_deg"] = directed_heading_error
        status["heading_axis_error_deg"] = float(min(directed_heading_error, abs(180.0 - directed_heading_error)))
        status["lane_center_distance_m"] = float(distance_2d(lane_wp.transform.location, location))

        target_point = scenario.get("target_road_point") or {}
        if {"x", "y"}.issubset(target_point):
            dx = float(location.x) - float(target_point["x"])
            dy = float(location.y) - float(target_point["y"])
            status["distance_m"] = float((dx * dx + dy * dy) ** 0.5)

        branch_point = scenario.get("branch_point") or {}
        if command == "straight" and {"x", "y"}.issubset(branch_point):
            road_heading = scenario.get("target_road_heading_deg")
            if road_heading is None:
                road_heading = lane_wp.transform.rotation.yaw
            heading_rad = math.radians(float(road_heading))
            dx = float(location.x) - float(branch_point["x"])
            dy = float(location.y) - float(branch_point["y"])
            forward_x = math.cos(heading_rad)
            forward_y = math.sin(heading_rad)
            right_x = -math.sin(heading_rad)
            right_y = math.cos(heading_rad)
            status["straight_branch_progress_m"] = float(dx * forward_x + dy * forward_y)
            status["straight_branch_lateral_m"] = float(abs(dx * right_x + dy * right_y))

        turn_heading_ok = True
        if command == "turn right":
            turn_heading_ok = heading_delta >= self.args.target_road_min_heading_deg
        elif command == "turn left":
            turn_heading_ok = heading_delta <= -self.args.target_road_min_heading_deg
        near_selected_target = (
            status["distance_m"] is not None
            and status["distance_m"] <= self.args.target_road_reach_distance_m
        )
        if command in ("turn left", "turn right") and not turn_heading_ok and not near_selected_target:
            direction = "right" if command == "turn right" else "left"
            status["reason"] = f"{direction}_heading_delta_too_small={heading_delta:.2f}"
            return status

        if (
            status["heading_axis_error_deg"] is None
            or status["heading_axis_error_deg"] > self.args.target_road_max_heading_error_deg
        ):
            status["reason"] = (
                f"heading_axis_error={status['heading_axis_error_deg']} "
                f"directed_heading_error={status['heading_error_deg']}"
            )
            return status
        if (
            status["lane_center_distance_m"] is None
            or status["lane_center_distance_m"] > self.args.target_road_max_lane_center_distance_m
        ):
            status["reason"] = f"lane_center_distance={status['lane_center_distance_m']}"
            return status
        if command == "straight":
            body_status = self.vehicle_body_on_driving_lane(vehicle)
        else:
            body_status = (
                self.vehicle_body_on_target_road(vehicle, int(target_road_id))
                if same_road
                else self.vehicle_body_on_driving_lane(vehicle)
            )
        status.update(body_status)
        if not status["body_on_target_road"]:
            body_reason = "body_not_on_target_road" if same_road else "body_not_on_driving_lane"
            status["reason"] = f"{body_reason} {status['body_target_road_samples']}/{status['body_total_samples']}"
            return status

        if command == "straight" and status["distance_from_start_m"] < self.args.straight_target_min_distance_m:
            status["reason"] = (
                f"straight_distance_from_start={status['distance_from_start_m']:.2f} "
                f"< {self.args.straight_target_min_distance_m:.2f}"
            )
            return status
        if (
            command == "straight"
            and status["straight_branch_progress_m"] is not None
            and status["straight_branch_progress_m"] < self.args.straight_post_junction_min_distance_m
        ):
            status["reason"] = (
                f"straight_branch_progress={status['straight_branch_progress_m']:.2f} "
                f"< {self.args.straight_post_junction_min_distance_m:.2f}"
            )
            return status

        require_target_distance = self.args.target_road_require_distance
        if require_target_distance and (
            status["distance_m"] is None or status["distance_m"] > self.args.target_road_reach_distance_m
        ):
            status["reason"] = f"target_distance={status['distance_m']}"
            return status

        status["reached"] = True
        if command == "straight":
            status["reason"] = (
                f"ok_straight_aligned_on_driving_lane "
                f"center_road={lane_wp.road_id} target_road={target_road_id} "
                f"distance_from_start={status['distance_from_start_m']:.2f}"
            )
        else:
            status["reason"] = "ok" if same_road else f"ok_aligned_on_driving_lane center_road={lane_wp.road_id} target_road={target_road_id}"
        return status

    def vehicle_body_on_target_road(self, vehicle: Any, target_road_id: int) -> dict[str, Any]:
        transform = vehicle.get_transform()
        extent = vehicle.bounding_box.extent
        yaw = math.radians(float(transform.rotation.yaw))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        origin = np.asarray([float(transform.location.x), float(transform.location.y)], dtype=np.float32)
        front = float(extent.x) + float(self.args.target_road_body_margin_m)
        side = float(extent.y) + float(self.args.target_road_body_margin_m)
        offsets = [
            (0.0, 0.0),
            (front, 0.0),
            (-front, 0.0),
            (front, side),
            (front, -side),
            (-front, side),
            (-front, -side),
        ]
        matched = 0
        samples = []
        for x_forward, y_right in offsets:
            point = origin + float(x_forward) * forward + float(y_right) * right
            loc = self.carla.Location(float(point[0]), float(point[1]), float(transform.location.z))
            wp = self.map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
            ok = wp is not None and int(wp.road_id) == int(target_road_id)
            matched += int(ok)
            samples.append(
                {
                    "x": float(loc.x),
                    "y": float(loc.y),
                    "on_target_road": bool(ok),
                    "road_id": None if wp is None else int(wp.road_id),
                    "lane_id": None if wp is None else int(wp.lane_id),
                }
            )
        required = max(1, int(math.ceil(len(offsets) * float(self.args.target_road_body_sample_fraction))))
        return {
            "body_on_target_road": matched >= required,
            "body_target_road_samples": matched,
            "body_total_samples": len(offsets),
            "body_required_samples": required,
            "body_samples": samples,
        }

    def vehicle_body_on_driving_lane(self, vehicle: Any) -> dict[str, Any]:
        transform = vehicle.get_transform()
        extent = vehicle.bounding_box.extent
        yaw = math.radians(float(transform.rotation.yaw))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        origin = np.asarray([float(transform.location.x), float(transform.location.y)], dtype=np.float32)
        front = float(extent.x) + float(self.args.target_road_body_margin_m)
        side = float(extent.y) + float(self.args.target_road_body_margin_m)
        offsets = [
            (0.0, 0.0),
            (front, 0.0),
            (-front, 0.0),
            (front, side),
            (front, -side),
            (-front, side),
            (-front, -side),
        ]
        matched = 0
        samples = []
        for x_forward, y_right in offsets:
            point = origin + float(x_forward) * forward + float(y_right) * right
            loc = self.carla.Location(float(point[0]), float(point[1]), float(transform.location.z))
            wp = self.map.get_waypoint(
                loc,
                project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
            ok = wp is not None
            matched += int(ok)
            samples.append(
                {
                    "x": float(loc.x),
                    "y": float(loc.y),
                    "on_target_road": bool(ok),
                    "road_id": None if wp is None else int(wp.road_id),
                    "lane_id": None if wp is None else int(wp.lane_id),
                }
            )
        required = max(1, int(math.ceil(len(offsets) * float(self.args.target_road_body_sample_fraction))))
        return {
            "body_on_target_road": matched >= required,
            "body_target_road_samples": matched,
            "body_total_samples": len(offsets),
            "body_required_samples": required,
            "body_samples": samples,
        }

    def spawn_vehicle_and_route(self, command: str, trial_index: int) -> tuple[Any, list[tuple[Any, Any]], dict[str, Any]]:
        blueprint = random.choice(self.world.get_blueprint_library().filter(self.args.vehicle_filter))
        spawn, route_plan, scenario = self.select_scenario(command, trial_index)
        vehicle = self.world.try_spawn_actor(blueprint, spawn)
        if vehicle is None:
            raise RuntimeError(f"Could not spawn ego vehicle at {spawn}")
        vehicle.set_autopilot(False)
        print(
            f"Spawned SimLingo ego: {blueprint.id} at spawn_index={scenario['spawn_index']} ({scenario['reason']})",
            flush=True,
        )
        return vehicle, route_plan, scenario

    def select_scenario(self, command: str, trial_index: int) -> tuple[Any, list[tuple[Any, Any]], dict[str, Any]]:
        spawn_points = list(self.map.get_spawn_points())
        rng = random.Random(self.args.scenario_seed + trial_index * 1009 + BASIC_COMMANDS.index(command) * 9176)
        indices = list(range(len(spawn_points)))
        rng.shuffle(indices)
        if self.args.spawn_index is not None and self.args.spawn_policy == "index":
            indices = [self.args.spawn_index]

        last_error = "no candidate checked"
        requested_min_distance = 0.0 if self.args.spawn_index is not None else float(self.args.min_spawn_distance_m)
        min_distance_attempts = [requested_min_distance]
        if requested_min_distance > 0.0 and self.args.relax_spawn_distance:
            min_distance_attempts.extend([requested_min_distance * 0.75, requested_min_distance * 0.5, 0.0])
        for min_distance in min_distance_attempts:
            for idx in indices[: self.args.max_spawn_candidates]:
                spawn = spawn_points[idx]
                nearest_used_distance = self.nearest_used_spawn_distance(command, spawn.location)
                if nearest_used_distance is not None and nearest_used_distance < min_distance:
                    last_error = f"spawn too close nearest={nearest_used_distance:.2f} < {min_distance:.2f}"
                    continue
                wp = self.map.get_waypoint(
                    spawn.location,
                    project_to_road=True,
                    lane_type=self.carla.LaneType.Driving,
                )
                if wp is None:
                    last_error = "spawn not on driving lane"
                    continue
                route_plan, meta = self.build_route(wp, command)
                if not route_plan:
                    last_error = meta.get("reason", "empty route")
                    continue
                if command in ("turn left", "turn right", "straight") and wp.is_junction:
                    last_error = "spawn starts inside junction"
                    continue
                if command in ("turn left", "turn right", "straight"):
                    branch_index = meta.get("branch_index")
                    if branch_index is None:
                        last_error = "missing junction branch index"
                        continue
                    if int(branch_index) < self.args.min_pre_junction_route_points:
                        last_error = (
                            f"junction branch too close branch_index={branch_index} "
                            f"< {self.args.min_pre_junction_route_points}"
                        )
                        continue
                if command == "straight" and self.args.straight_require_junction and not meta.get("branch_committed", False):
                    last_error = "straight route is not an intersection straight branch"
                    continue
                if command == "straight" and abs(meta.get("route_heading_delta_deg", 0.0)) > self.args.straight_route_max_heading:
                    last_error = f"straight route drifts {meta.get('route_heading_delta_deg')}"
                    continue
                if command in SPEED_EVAL_COMMANDS and abs(meta.get("route_heading_delta_deg", 0.0)) > self.args.speed_route_max_heading:
                    last_error = f"speed route drifts {meta.get('route_heading_delta_deg')}"
                    continue
                scenario = {
                    "map": self.map.name,
                    "spawn_index": idx,
                    "command": command,
                    "reason": meta.get("reason", "ok"),
                    "spawn_min_distance_requested_m": float(requested_min_distance),
                    "spawn_min_distance_applied_m": float(min_distance),
                    "spawn_nearest_used_distance_m": None
                    if nearest_used_distance is None
                    else float(nearest_used_distance),
                    "spawn_diversity_scope": self.args.spawn_diversity_scope,
                    **meta,
                }
                self.used_spawn_locations.append((command, (float(spawn.location.x), float(spawn.location.y))))
                return spawn, route_plan, scenario
        raise RuntimeError(f"Could not select scenario for {command}: {last_error}")

    def nearest_used_spawn_distance(self, command: str, location: Any) -> float | None:
        distances = []
        for used_command, (x, y) in self.used_spawn_locations:
            if self.args.spawn_diversity_scope == "command" and used_command != command:
                continue
            distances.append(math.hypot(float(location.x) - x, float(location.y) - y))
        return min(distances) if distances else None

    def build_route(self, start_wp: Any, command: str) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        if command in ("turn left", "turn right", "straight"):
            return self.build_intersection_route(start_wp, command)
        return self.build_lane_follow_route(start_wp, command, self.args.speed_route_length_m)

    def build_lane_follow_route(self, start_wp: Any, command: str, route_length_m: float) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        route: list[tuple[Any, Any]] = []
        current = start_wp
        steps = max(5, int(route_length_m / self.args.route_step_m))
        for _ in range(steps):
            route.append((current.transform, self.RoadOption.LANEFOLLOW))
            nxt = current.next(self.args.route_step_m)
            if not nxt:
                break
            current = choose_straight_successor(current, nxt)
        heading_delta = angle_delta_deg(start_wp.transform.rotation.yaw, current.transform.rotation.yaw)
        return route, {
            "reason": "lane-follow route",
            "route_heading_delta_deg": float(heading_delta),
            "route_points": len(route),
        }

    def build_intersection_route(self, start_wp: Any, command: str) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        route: list[tuple[Any, Any]] = []
        current = start_wp
        branch_committed = False
        branch_index: int | None = None
        entry_yaw = start_wp.transform.rotation.yaw
        chosen_delta = 0.0
        chosen_reason = "no junction branch"

        max_steps = int(self.args.turn_route_length_m / self.args.route_step_m)
        for _ in range(max_steps):
            route.append((current.transform, self.RoadOption.LANEFOLLOW))
            successors = current.next(self.args.route_step_m)
            if not successors:
                break

            if not branch_committed and (len(successors) > 1 or current.is_junction):
                choice, delta, reason = self.choose_branch(current, successors, command)
                if choice is not None:
                    chosen_delta = delta
                    chosen_reason = reason
                    branch_committed = True
                    branch_index = len(route) - 1
                    current = choice
                    route[-1] = (route[-1][0], road_option_for_command(self.RoadOption, command))
                    continue

            current = choose_straight_successor(current, successors)

        if command in ("turn left", "turn right") and not branch_committed:
            return [], {"reason": chosen_reason}
        heading_delta = angle_delta_deg(start_wp.transform.rotation.yaw, current.transform.rotation.yaw)
        if command == "turn left" and chosen_delta > -self.args.junction_min_turn_deg:
            return [], {"reason": f"left branch too weak delta={chosen_delta:.2f}"}
        if command == "turn right" and chosen_delta < self.args.junction_min_turn_deg:
            return [], {"reason": f"right branch too weak delta={chosen_delta:.2f}"}
        if command == "straight" and abs(heading_delta) > self.args.straight_route_max_heading:
            return [], {"reason": f"straight heading drift {heading_delta:.2f}"}
        if branch_index is not None:
            option = road_option_for_command(self.RoadOption, command)
            start = 0 if command in ("turn left", "turn right") else max(0, branch_index - 5)
            end = min(len(route), branch_index + 35)
            for idx in range(start, end):
                route[idx] = (route[idx][0], option)
        target_road_id = None
        target_lane_id = None
        target_road_point = None
        target_road_heading_deg = None
        target_road_route_index = None
        branch_point = None
        if branch_index is not None and route:
            branch_point = location_dict(route[branch_index][0].location)
            target_depth_m = (
                float(self.args.straight_target_depth_m)
                if command == "straight"
                else float(self.args.target_road_depth_m)
            )
            target_idx = min(
                len(route) - 1,
                branch_index + max(8, int(round(target_depth_m / self.args.route_step_m))),
            )
            target_road_route_index = int(target_idx)
            target_loc = route[target_idx][0].location
            target_wp = self.map.get_waypoint(
                target_loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
            if target_wp is not None:
                target_road_id = int(target_wp.road_id)
                target_lane_id = int(target_wp.lane_id)
                target_road_heading_deg = float(target_wp.transform.rotation.yaw)
                target_road_point = location_dict(target_wp.transform.location)
        return route, {
            "reason": "junction route" if branch_committed else "straight continuation route",
            "branch_committed": branch_committed,
            "branch_index": branch_index,
            "branch_point": branch_point,
            "branch_delta_deg": float(chosen_delta),
            "route_heading_delta_deg": float(heading_delta),
            "route_points": len(route),
            "target_road_depth_m": target_depth_m if branch_index is not None else None,
            "target_road_route_index": target_road_route_index,
            "target_road_id": target_road_id,
            "target_lane_id": target_lane_id,
            "target_road_heading_deg": target_road_heading_deg,
            "target_road_point": target_road_point,
        }

    def choose_branch(self, current: Any, successors: list[Any], command: str) -> tuple[Any | None, float, str]:
        candidates = []
        entry_yaw = current.transform.rotation.yaw
        for succ in successors:
            probe = succ
            for _ in range(10):
                nxt = probe.next(self.args.route_step_m)
                if not nxt:
                    break
                probe = choose_straight_successor(probe, nxt)
                if not probe.is_junction:
                    break
            delta = angle_delta_deg(entry_yaw, probe.transform.rotation.yaw)
            candidates.append((succ, delta))
        if not candidates:
            return None, 0.0, "no branch candidates"
        if command == "turn right":
            valid = [(wp, d) for wp, d in candidates if d >= self.args.junction_min_turn_deg]
            if not valid:
                return None, max(d for _, d in candidates), "no right branch"
            return max(valid, key=lambda x: x[1])[0], max(valid, key=lambda x: x[1])[1], "right branch"
        if command == "turn left":
            valid = [(wp, d) for wp, d in candidates if d <= -self.args.junction_min_turn_deg]
            if not valid:
                return None, min(d for _, d in candidates), "no left branch"
            return min(valid, key=lambda x: x[1])[0], min(valid, key=lambda x: x[1])[1], "left branch"
        valid = [(wp, d) for wp, d in candidates if abs(d) <= self.args.straight_route_max_heading]
        if not valid:
            return None, min(candidates, key=lambda x: abs(x[1]))[1], "no straight branch"
        return min(valid, key=lambda x: abs(x[1]))[0], min(valid, key=lambda x: abs(x[1]))[1], "straight branch"

    def spawn_camera(self, vehicle: Any, blueprint: Any, transform: Any) -> Any:
        return self.world.spawn_actor(blueprint, transform, attach_to=vehicle)

    def simlingo_camera_blueprint(self) -> Any:
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "1024")
        bp.set_attribute("image_size_y", "512")
        bp.set_attribute("fov", "110")
        bp.set_attribute("sensor_tick", str(1.0 / float(self.args.sim_fps)))
        return bp

    def video_camera_blueprint(self) -> Any:
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.args.width))
        bp.set_attribute("image_size_y", str(self.args.height))
        bp.set_attribute("fov", str(self.args.fov))
        bp.set_attribute("sensor_tick", str(1.0 / float(self.args.sim_fps)))
        return bp

    def simlingo_camera_transform(self) -> Any:
        return self.carla.Transform(
            self.carla.Location(x=-1.5, y=0.0, z=2.0),
            self.carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
        )

    def video_camera_transform(self) -> Any:
        if self.args.camera_view == "topdown":
            return self.carla.Transform(
                self.carla.Location(x=0.0, y=0.0, z=self.args.topdown_z),
                self.carla.Rotation(pitch=-90.0, yaw=0.0),
            )
        if self.args.camera_view == "chase":
            return self.carla.Transform(
                self.carla.Location(x=-8.0, y=0.0, z=4.0),
                self.carla.Rotation(pitch=-18.0, yaw=0.0),
            )
        return self.simlingo_camera_transform()

    def spawn_collision_sensor(self, vehicle: Any, collision_events: list[dict[str, Any]]) -> Any:
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, self.carla.Transform(), attach_to=vehicle)

        def _on_collision(event: Any) -> None:
            collision_events.append(
                {
                    "frame": int(event.frame),
                    "other_actor": getattr(event.other_actor, "type_id", "unknown"),
                    "normal_impulse": {
                        "x": float(event.normal_impulse.x),
                        "y": float(event.normal_impulse.y),
                        "z": float(event.normal_impulse.z),
                    },
                }
            )

        sensor.listen(_on_collision)
        return sensor

    def apply_sync_settings(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.args.sim_fps)
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

    def wait_for_sensor_start(
        self,
        image_queue: queue.Queue[Any],
        video_queue: queue.Queue[Any] | None = None,
    ) -> tuple[Any, Any | None]:
        return self.wait_for_sensor_frame(
            image_queue=image_queue,
            video_queue=video_queue,
            max_ticks=max(1, int(self.args.sensor_warmup_ticks)),
            startup=True,
        )

    def wait_for_sensor_frame(
        self,
        image_queue: queue.Queue[Any],
        video_queue: queue.Queue[Any] | None = None,
        max_ticks: int | None = None,
        startup: bool = False,
    ) -> tuple[Any, Any | None]:
        rgb_image = None
        video_image = None
        ticks = max(1, int(max_ticks if max_ticks is not None else self.args.sensor_retry_ticks))
        for _ in range(ticks):
            self.world.tick()
            if rgb_image is None:
                try:
                    rgb_image = image_queue.get(timeout=self.args.sensor_timeout)
                except queue.Empty:
                    pass
            if video_queue is not None and video_image is None:
                try:
                    video_image = video_queue.get(timeout=self.args.sensor_timeout)
                except queue.Empty:
                    pass
            if rgb_image is not None and (video_queue is None or video_image is not None):
                return rgb_image, video_image
        missing = ["rgb_0"] if rgb_image is None else []
        if video_queue is not None and video_image is None:
            missing.append(self.args.camera_view)
        raise RuntimeError(
            f"Timed out waiting for CARLA camera image during {'sensor startup' if startup else 'simulation tick'}. "
            f"Missing streams: {', '.join(missing)}. "
            "If this persists, verify the CARLA runtime was not started with no-rendering mode."
        )

    @staticmethod
    def drain(q: queue.Queue[Any]) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def write_eval_summary(self, results: list[TrialResult]) -> None:
        rows = [asdict(result) for result in results]
        successes = sum(1 for result in results if result.success)
        total = len(results)
        success_rate = successes / total if total else 0.0
        summary = {
            "controller": "simlingo_lingoagent",
            "executable_action_schema": ["steer", "throttle", "brake"],
            "checkpoint": str(self.args.checkpoint),
            "map": self.map.name,
            "total_trials": total,
            "successes": successes,
            "success_rate": success_rate,
            "target_success_rate": self.args.target_success_rate,
            "passed_target": success_rate >= self.args.target_success_rate if total else False,
            "results": rows,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (self.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else list(TrialResult.__annotations__)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        failures = [result for result in results if not result.success]
        lines = [
            "# SimLingo Task 1 Failure Analysis",
            "",
            "Executable action: `carla.VehicleControl(steer, throttle, brake)`",
            f"Success rate: {success_rate:.3f}",
            "",
        ]
        if failures:
            for failure in failures:
                lines.append(f"- `{failure.command}` trial {failure.trial_index}: {failure.reason}")
        else:
            lines.append("No failures.")
        (self.output_dir / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tensor_to_list(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return _to_jsonable(value)
    except Exception as exc:
        return f"unserializable: {exc}"


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def load_connection(path: pathlib.Path | None, host: str | None, port: int | None) -> tuple[str, int]:
    env = {}
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"connection.env not found: {path}")
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    final_host = host or env.get("CARLA_HOST") or env.get("HOST") or "127.0.0.1"
    final_port = int(port or env.get("CARLA_PORT") or env.get("PORT") or 2000)
    return final_host, final_port


def normalize_command(command: str) -> str:
    normalized = " ".join(command.strip().lower().replace("_", " ").split())
    aliases = {
        "left": "turn left",
        "right": "turn right",
        "go left": "turn left",
        "go right": "turn right",
        "forward": "straight",
        "go straight": "straight",
        "accelerate": "speed up",
        "faster": "speed up",
        "decelerate": "slow down",
        "slower": "slow down",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in BASIC_COMMANDS:
        raise ValueError(f"Unsupported command {command!r}. Expected one of: {', '.join(BASIC_COMMANDS)}")
    return normalized


def parse_command_sequence(raw: str) -> list[str]:
    commands = [normalize_command(part) for part in raw.split(",") if part.strip()]
    if not commands:
        raise ValueError("Expected at least one command.")
    return commands


def command_prompt(command: str | None) -> str | None:
    if command == "turn left":
        return "Command: turn left at the next intersection."
    if command == "turn right":
        return "Command: turn right at the next intersection."
    if command == "straight":
        return "Command: go straight at the next intersection."
    if command == "stop":
        return "Command: continue following your current lane and brake to a complete stop."
    if command == "speed up":
        return "Command: continue following your current lane and speed up."
    if command == "slow down":
        return "Command: continue following your current lane and slow down."
    return None


def task1_user_flag(command: str | None) -> int | None:
    if command == "stop":
        return 0  # SimLingo wraps this as <SAFETY>.
    if command is not None:
        return 1  # SimLingo wraps this as <INSTRUCTION_FOLLOWING>.
    return None


def command_warmup_sec(command: str, args: argparse.Namespace) -> float:
    if command == "stop":
        return float(args.stop_warmup_sec)
    if command in ("speed up", "slow down"):
        return float(args.speed_command_warmup_sec)
    return 0.0


def road_option_for_command(RoadOption: Any, command: str) -> Any:
    if command == "turn left":
        return RoadOption.LEFT
    if command == "turn right":
        return RoadOption.RIGHT
    if command == "straight":
        return RoadOption.STRAIGHT
    return RoadOption.LANEFOLLOW


def choose_straight_successor(current: Any, successors: list[Any]) -> Any:
    if len(successors) == 1:
        return successors[0]
    yaw = current.transform.rotation.yaw
    return min(successors, key=lambda wp: abs(angle_delta_deg(yaw, wp.transform.rotation.yaw)))


def angle_delta_deg(source: float, target: float) -> float:
    return (target - source + 180.0) % 360.0 - 180.0


def distance_2d(a: Any, b: Any) -> float:
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))


def speed_mps(vehicle: Any) -> float:
    vel = vehicle.get_velocity()
    return float(math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))


def extract_xy_points(raw: Any) -> np.ndarray:
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
    return arr[:, :2].copy()


def local_route_to_world(points: np.ndarray, transform: Any) -> np.ndarray:
    """Convert SimLingo local points [x_forward, y_right] to CARLA world x/y."""
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    yaw = math.radians(float(transform.rotation.yaw))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    origin = np.asarray([float(transform.location.x), float(transform.location.y)], dtype=np.float32)
    world = origin[None, :] + points[:, 0:1] * forward[None, :] + points[:, 1:2] * right[None, :]
    return world.astype(np.float32)


def blend_local_routes(old_local: np.ndarray, new_local: np.ndarray, new_weight: float) -> np.ndarray:
    old_ahead = old_local[old_local[:, 0] > 0.25] if old_local.size else np.zeros((0, 2), dtype=np.float32)
    if old_ahead.size == 0 or new_local.size == 0:
        return new_local
    count = min(len(old_ahead), len(new_local))
    blended = new_local.copy()
    weight = float(np.clip(new_weight, 0.0, 1.0))
    blended[:count] = (1.0 - weight) * old_ahead[:count] + weight * new_local[:count]
    return blended.astype(np.float32)


def route_plan_to_world_points(route_plan: list[tuple[Any, Any]]) -> np.ndarray:
    points = []
    for transform, _option in route_plan:
        location = transform.location
        points.append([float(location.x), float(location.y)])
    return np.asarray(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)


def world_route_to_local(points: np.ndarray, transform: Any) -> np.ndarray:
    """Convert CARLA world x/y points to SimLingo local [x_forward, y_right]."""
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    yaw = math.radians(float(transform.rotation.yaw))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    origin = np.asarray([float(transform.location.x), float(transform.location.y)], dtype=np.float32)
    delta = points[:, :2] - origin[None, :]
    return np.stack([delta @ forward, delta @ right], axis=1).astype(np.float32)


def count_cached_route_ahead_points(world_points: np.ndarray, transform: Any) -> int:
    if world_points.size == 0:
        return 0
    local = world_route_to_local(world_points, transform)
    return int(np.sum(local[:, 0] > 0.25))


def route_lateral_diagnostics(local_points: np.ndarray) -> dict[str, float | int]:
    if local_points.size == 0:
        return {
            "points": 0,
            "last_y_right_m": 0.0,
            "min_y_right_m": 0.0,
            "max_y_right_m": 0.0,
            "mean_abs_y_right_m": 0.0,
        }
    y = local_points[:, 1]
    return {
        "points": int(len(local_points)),
        "last_y_right_m": float(y[-1]),
        "min_y_right_m": float(np.min(y)),
        "max_y_right_m": float(np.max(y)),
        "mean_abs_y_right_m": float(np.mean(np.abs(y))),
    }


def rate_limit_control(control: Any, previous: Any, args: argparse.Namespace, carla_module: Any) -> tuple[Any, dict[str, float]]:
    if previous is None:
        return control, {"steer_delta_limited": 0.0, "raw_steer": float(control.steer)}
    max_delta = float(args.cached_route_max_steer_rate_per_sec) / max(1.0, float(args.sim_fps))
    raw_steer = float(control.steer)
    prev_steer = float(previous.steer)
    limited_steer = float(np.clip(raw_steer, prev_steer - max_delta, prev_steer + max_delta))
    limited = carla_module.VehicleControl(
        steer=limited_steer,
        throttle=float(control.throttle),
        brake=float(control.brake),
    )
    return limited, {"steer_delta_limited": abs(raw_steer - limited_steer), "raw_steer": raw_steer}


def estimate_model_target_speed_mps(raw_speed_waypoints: Any) -> float | None:
    points = extract_xy_points(raw_speed_waypoints)
    if len(points) < 3:
        return None
    # Matches SimLingo's native controller for carla_fps=20, wp_dilation=1, data_save_freq=5:
    # one_second=4, half_second=2, desired_speed=norm(speed_wp[0] - speed_wp[2]) * 2.
    speed = float(np.linalg.norm(points[0] - points[2]) * 2.0)
    if not math.isfinite(speed):
        return None
    return speed


def follow_cached_route(
    vehicle: Any,
    world_points: np.ndarray,
    command: str,
    args: argparse.Namespace,
    carla_module: Any,
    model_target_speed_mps: float | None = None,
    velocity_head_target_speed_mps: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    transform = vehicle.get_transform()
    local = world_route_to_local(world_points, transform)
    local = local[local[:, 0] > 0.25]
    speed = speed_mps(vehicle)
    if local.size == 0:
        return carla_module.VehicleControl(steer=0.0, throttle=0.0, brake=1.0), {
            "cached_local_route": [],
            "cached_target_point": None,
            "cached_lookahead_m": None,
            "cached_target_speed_mps": 0.0,
            "cached_reason": "empty_or_behind",
        }

    lookahead = float(args.cached_route_base_lookahead_m) + float(args.cached_route_speed_lookahead_gain) * speed
    lookahead = float(np.clip(lookahead, args.cached_route_min_lookahead_m, args.cached_route_max_lookahead_m))
    distances = np.linalg.norm(local, axis=1)
    ahead = np.where(distances >= lookahead)[0]
    target_index = int(ahead[0]) if len(ahead) else int(len(local) - 1)
    target = local[target_index]

    angle = math.atan2(float(target[1]), max(0.1, float(target[0])))
    steer = float(np.clip(float(args.cached_route_steer_gain) * angle, -args.cached_route_max_steer, args.cached_route_max_steer))

    speed_source = "command"
    if velocity_head_target_speed_mps is not None:
        target_speed = float(np.clip(velocity_head_target_speed_mps, args.velocity_head_min_speed_mps, args.velocity_head_max_speed_mps))
        speed_source = "velocity_head"
    elif (
        args.simlingo_speed_control_mode == "model"
        or (args.simlingo_speed_control_mode == "model-speed-commands" and command in SPEED_EVAL_COMMANDS)
    ) and model_target_speed_mps is not None:
        target_speed = float(np.clip(model_target_speed_mps, 0.0, args.cached_route_fast_speed_mps))
        speed_source = "model"
    elif command == "stop":
        target_speed = 0.0
    elif command in ("turn left", "turn right"):
        target_speed = float(args.cached_route_turn_speed_mps)
    elif command == "slow down":
        target_speed = float(args.cached_route_slow_speed_mps)
    elif command == "speed up":
        target_speed = float(args.cached_route_fast_speed_mps)
    else:
        target_speed = float(args.cached_route_cruise_speed_mps)

    error = target_speed - speed
    if target_speed <= 0.05 or error < -float(args.cached_route_brake_margin_mps):
        throttle = 0.0
        brake = float(np.clip((speed - target_speed) * args.cached_route_brake_gain, 0.0, args.cached_route_max_brake))
    else:
        if error <= float(args.cached_route_throttle_deadband_mps):
            throttle = 0.0
        else:
            throttle_feedforward = float(args.cached_route_throttle_feedforward) + (
                float(args.cached_route_speed_throttle_feedforward_gain) * max(0.0, target_speed)
            )
            throttle = float(
                np.clip(
                    throttle_feedforward + error * args.cached_route_throttle_gain,
                    0.0,
                    args.cached_route_max_throttle,
                )
            )
        brake = 0.0

    return carla_module.VehicleControl(steer=steer, throttle=throttle, brake=brake), {
        "cached_local_route": local.tolist(),
        "cached_target_point": target.tolist(),
        "cached_target_index": target_index,
        "cached_lookahead_m": lookahead,
        "cached_target_speed_mps": target_speed,
        "cached_model_target_speed_mps": model_target_speed_mps,
        "cached_velocity_head_target_speed_mps": velocity_head_target_speed_mps,
        "cached_speed_source": speed_source,
        "cached_speed_error_mps": error,
        "cached_throttle_feedforward": throttle_feedforward if "throttle_feedforward" in locals() else 0.0,
        "cached_reason": "ok",
    }


def is_vehicle_on_driving_lane(carla_map: Any, location: Any, carla_module: Any) -> bool:
    wp = carla_map.get_waypoint(location, project_to_road=False, lane_type=carla_module.LaneType.Driving)
    return wp is not None


def carla_image_to_bgra(image: Any) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr.copy()


def carla_image_to_bgr(image: Any) -> np.ndarray:
    return np.ascontiguousarray(carla_image_to_bgra(image)[:, :, :3])


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))


def draw_prediction_overlay(frame: np.ndarray, diag: dict[str, Any], args: argparse.Namespace) -> None:
    height, width = frame.shape[:2]
    ppm = float(args.prediction_overlay_pixels_per_meter)
    if ppm <= 0.0:
        ground_width_m = 2.0 * float(args.topdown_z) * math.tan(math.radians(float(args.fov)) / 2.0)
        ppm = width / max(1.0, ground_width_m)

    def to_pixels(points: np.ndarray) -> list[tuple[int, int]]:
        pixels = []
        cx = width * 0.5
        cy = height * 0.5
        for x_forward, y_right in points:
            col = int(round(cx + float(y_right) * ppm))
            row = int(round(cy - float(x_forward) * ppm))
            if -50 <= col <= width + 50 and -50 <= row <= height + 50:
                pixels.append((col, row))
        return pixels

    def extract(raw: Any) -> np.ndarray:
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
        return arr[:, :2]

    route = extract(diag.get("predicted_route"))
    speed_route = extract(diag.get("predicted_speed_waypoints"))
    cached_route = extract(diag.get("cached_follow_route"))
    command = str(diag.get("command") or "")

    cv2.circle(frame, (width // 2, height // 2), 5, (255, 255, 255), -1)
    cv2.putText(frame, "ego", (width // 2 + 8, height // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    for points, color, label, radius in (
        (route, (255, 0, 255), "pred path", 4),
        (speed_route, (0, 255, 0), "pred speed", 3),
        (cached_route, (255, 255, 0), "cached follow", 3),
    ):
        pixels = to_pixels(points)
        if len(pixels) >= 2:
            cv2.polylines(frame, [np.asarray(pixels, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for idx, point in enumerate(pixels):
            cv2.circle(frame, point, radius, color, -1)
            if idx in (0, len(pixels) - 1):
                cv2.circle(frame, point, radius + 2, color, 1)

    cv2.rectangle(frame, (8, 8), (270, 80), (0, 0, 0), -1)
    cv2.putText(frame, "purple: predicted path", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(frame, "green: predicted speed wps", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "cyan: cached follow path", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if command in ("stop", "speed up", "slow down"):
        speed = float(diag.get("speed_mps") or 0.0)
        target_speed = None
        cached_follow = diag.get("cached_follow")
        if isinstance(cached_follow, dict):
            raw_target = cached_follow.get("cached_target_speed_mps")
            if raw_target is not None:
                try:
                    target_speed = float(raw_target)
                except (TypeError, ValueError):
                    target_speed = None
        box_w = 230
        box_h = 72 if target_speed is not None else 50
        x0 = max(8, width - box_w - 10)
        y0 = 10
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"speed {speed:.2f} m/s",
            (x0 + 12, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if target_speed is not None:
            cv2.putText(
                frame,
                f"target {target_speed:.2f} m/s",
                (x0 + 12, y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )


def target_road_polygon_pixels(
    diag: dict[str, Any],
    ppm: float,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> np.ndarray | None:
    target_point = diag.get("target_road_point")
    ego_location = diag.get("ego_location")
    target_heading = diag.get("target_road_heading_deg")
    ego_heading = diag.get("ego_heading_deg")
    if not isinstance(target_point, dict) or not isinstance(ego_location, dict):
        return None
    if target_heading is None or ego_heading is None:
        return None
    try:
        target_xy = np.asarray([float(target_point["x"]), float(target_point["y"])], dtype=np.float32)
        ego_xy = np.asarray([float(ego_location["x"]), float(ego_location["y"])], dtype=np.float32)
        ego_yaw = math.radians(float(ego_heading))
        road_yaw = math.radians(float(target_heading))
    except (KeyError, TypeError, ValueError):
        return None

    ego_forward = np.asarray([math.cos(ego_yaw), math.sin(ego_yaw)], dtype=np.float32)
    ego_right = np.asarray([-math.sin(ego_yaw), math.cos(ego_yaw)], dtype=np.float32)
    road_forward_world = np.asarray([math.cos(road_yaw), math.sin(road_yaw)], dtype=np.float32)
    road_right_world = np.asarray([-math.sin(road_yaw), math.cos(road_yaw)], dtype=np.float32)

    center_delta = target_xy - ego_xy
    center_local = np.asarray([center_delta @ ego_forward, center_delta @ ego_right], dtype=np.float32)
    road_forward_local = np.asarray([road_forward_world @ ego_forward, road_forward_world @ ego_right], dtype=np.float32)
    road_right_local = np.asarray([road_right_world @ ego_forward, road_right_world @ ego_right], dtype=np.float32)

    half_length = float(args.target_road_overlay_length_m) * 0.5
    half_width = float(args.target_road_overlay_width_m) * 0.5
    corners = []
    for forward_sign, right_sign in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        local = (
            center_local
            + forward_sign * half_length * road_forward_local
            + right_sign * half_width * road_right_local
        )
        col = int(round(width * 0.5 + float(local[1]) * ppm))
        row = int(round(height * 0.5 - float(local[0]) * ppm))
        corners.append((col, row))
    return np.asarray(corners, dtype=np.int32)


def wait_for_image(q: queue.Queue[Any], timeout: float) -> Any:
    try:
        return q.get(timeout=timeout)
    except queue.Empty as exc:
        raise RuntimeError("Timed out waiting for CARLA camera image.") from exc


def write_video(path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    height, width = first.shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        normalized_frames.append(frame[:, :, :3])

    cleanup_video_sidecars(path)
    ffmpeg = find_ffmpeg()
    if ffmpeg is not None:
        codec_args, codec_name = pick_ffmpeg_encoder(ffmpeg)
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            *codec_args,
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception as exc:
            print(f"WARNING: could not start ffmpeg ({exc}); falling back to OpenCV mp4v.", flush=True)
        else:
            try:
                assert proc.stdin is not None
                for frame in normalized_frames:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                proc.stdin.close()
                return_code = proc.wait()
                if return_code == 0:
                    print(f"Wrote MP4 with ffmpeg encoder: {codec_name} -> {path}", flush=True)
                    return
            finally:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
                if proc.poll() is None:
                    proc.kill()
            print("WARNING: ffmpeg video encoding failed; falling back to OpenCV mp4v.", flush=True)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    try:
        for frame in normalized_frames:
            writer.write(frame)
    finally:
        writer.release()
    print(
        "WARNING: wrote MP4 with OpenCV mp4v fallback. "
        "If your viewer rejects it, install/use ffmpeg with an H.264 encoder.",
        flush=True,
    )


def cleanup_video_sidecars(path: pathlib.Path) -> None:
    sidecars = [
        path.with_suffix(".avi"),
        path.with_name(path.stem + "_mjpg.avi"),
        path.with_name(path.stem + "_first.jpg"),
        path.with_name(path.stem + "_middle.jpg"),
        path.with_name(path.stem + "_last.jpg"),
        path.with_name(path.stem + "_contact.jpg"),
    ]
    for sidecar in sidecars:
        try:
            if sidecar.exists():
                sidecar.unlink()
        except OSError:
            pass


def ffmpeg_encoders(ffmpeg: str) -> str:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout
    except Exception:
        return ""


def pick_ffmpeg_encoder(ffmpeg: str) -> tuple[list[str], str]:
    encoders = ffmpeg_encoders(ffmpeg)
    if " libx264 " in encoders:
        return ["-vcodec", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"], "libx264"
    if " libopenh264 " in encoders:
        return ["-vcodec", "libopenh264", "-b:v", "2500k", "-pix_fmt", "yuv420p", "-tag:v", "avc1"], "libopenh264"
    if " h264_nvenc " in encoders:
        return ["-vcodec", "h264_nvenc", "-preset", "p4", "-b:v", "2500k", "-pix_fmt", "yuv420p", "-tag:v", "avc1"], "h264_nvenc"
    if " mpeg4 " in encoders:
        return ["-vcodec", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p"], "mpeg4"
    return ["-vcodec", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p"], "mpeg4"


def find_ffmpeg() -> str | None:
    candidates = [
        os.environ.get("TASK1_FFMPEG"),
        os.environ.get("FFMPEG_BIN"),
        shutil.which("ffmpeg"),
        "/lab/haoq_lab/cse12312032/miniconda3/envs/simlingo/bin/ffmpeg",
        "/lab/haoq_lab/cse12312032/miniconda3/envs/drivevla/bin/ffmpeg",
        "/lab/haoq_lab/cse12312032/miniconda3/envs/openvla/bin/ffmpeg",
    ]
    existing = [candidate for candidate in candidates if candidate and pathlib.Path(candidate).exists()]
    if not existing:
        return None
    for candidate in existing:
        encoders = ffmpeg_encoders(candidate)
        if any(name in encoders for name in (" libx264 ", " libopenh264 ", " h264_nvenc ")):
            return candidate
    return existing[0]


def location_dict(location: Any) -> dict[str, float]:
    return {"x": float(location.x), "y": float(location.y), "z": float(location.z)}


def safe_name(command: str) -> str:
    return command.replace(" ", "_")


def parse_duration_overrides(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not raw:
        return out
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        out[normalize_command(key)] = float(value)
    return out


def smoke_model(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for smoke-model")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = SimLingoTask1Policy(args.checkpoint, output_dir)
    start = time.time()
    policy.load()
    elapsed = time.time() - start
    print(json.dumps({"loaded": True, "checkpoint": str(args.checkpoint), "latency_sec": elapsed}, indent=2))
    return 0


def camera_smoke(args: argparse.Namespace) -> int:
    import carla

    host, port = load_connection(args.connection_env, args.host, args.port)
    client = carla.Client(host, int(port))
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    original_settings = world.get_settings()
    actors: list[Any] = []
    image_queue: queue.Queue[Any] = queue.Queue()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.sim_fps)
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points.")
        spawn_index = args.spawn_index if args.spawn_index is not None else 0
        vehicle_bp = world.get_blueprint_library().filter(args.vehicle_filter)[0]
        vehicle_bp.set_attribute("role_name", "hero")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[spawn_index % len(spawn_points)])
        if vehicle is None:
            raise RuntimeError(f"Could not spawn vehicle at spawn_index={spawn_index}.")
        actors.append(vehicle)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "512")
        camera_bp.set_attribute("fov", "110")
        camera_bp.set_attribute("sensor_tick", str(1.0 / float(args.sim_fps)))
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-1.5, y=0.0, z=2.0), carla.Rotation()),
            attach_to=vehicle,
        )
        camera.listen(lambda image: image_queue.put(image))
        actors.append(camera)

        image = None
        for _ in range(max(1, int(args.sensor_warmup_ticks))):
            world.tick()
            try:
                image = image_queue.get(timeout=args.sensor_timeout)
                break
            except queue.Empty:
                pass
        if image is None:
            raise RuntimeError(
                "Camera smoke failed: no RGB image received. "
                "Check that CARLA no_rendering_mode is false and the runtime supports offscreen rendering."
            )

        print(
            json.dumps(
                {
                    "camera_smoke": True,
                    "map": carla_map.name,
                    "spawn_index": spawn_index,
                    "frame": int(image.frame),
                    "width": int(image.width),
                    "height": int(image.height),
                    "synchronous_mode": bool(world.get_settings().synchronous_mode),
                    "no_rendering_mode": bool(world.get_settings().no_rendering_mode),
                },
                indent=2,
            )
        )
        return 0
    finally:
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)


def debug_imports(args: argparse.Namespace) -> int:
    import importlib.util

    import torch

    carla_spec = importlib.util.find_spec("carla")
    try:
        libcarla_spec_origin = (
            getattr(importlib.util.find_spec("carla.libcarla"), "origin", None) if carla_spec else None
        )
    except Exception as exc:
        libcarla_spec_origin = f"ERROR: {exc!r}"
    result: dict[str, Any] = {
        "python": sys.version,
        "sys_path_head": sys.path[:12],
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "carla_root": os.environ.get("CARLA_ROOT"),
        "carla_import_compat_root": os.environ.get("CARLA_IMPORT_COMPAT_ROOT"),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint_exists": args.checkpoint.exists() if args.checkpoint else None,
        "carla_spec_origin": getattr(carla_spec, "origin", None),
        "libcarla_spec_origin": libcarla_spec_origin,
    }
    try:
        import carla

        result["carla"] = getattr(carla, "__file__", "ok")
    except Exception as exc:
        result["carla_error"] = repr(exc)
    try:
        import team_code.agent_simlingo as agent_simlingo

        result["agent_simlingo"] = getattr(agent_simlingo, "__file__", "ok")
    except Exception as exc:
        result["agent_simlingo_error"] = repr(exc)
    if args.checkpoint:
        result["hydra_config"] = str(args.checkpoint.parent.parent.parent / ".hydra" / "config.yaml")
        result["hydra_config_exists"] = (args.checkpoint.parent.parent.parent / ".hydra" / "config.yaml").exists()
    print(json.dumps(result, indent=2))
    return 0 if "carla_error" not in result and "agent_simlingo_error" not in result else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-env", type=pathlib.Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("logs/task1/simlingo"))
    parser.add_argument("--mode", choices=("run", "camera-smoke", "smoke-model", "debug-imports"), default="run")
    parser.add_argument("--velocity-head-checkpoint", type=pathlib.Path)
    parser.add_argument("--velocity-head-device", default="cpu")
    parser.add_argument("--velocity-head-min-speed-mps", type=float, default=0.0)
    parser.add_argument("--velocity-head-max-speed-mps", type=float, default=8.0)
    parser.add_argument(
        "--velocity-head-hz",
        type=float,
        default=5.0,
        help="Velocity-head inference frequency. This affects only scalar speed control, not SimLingo route generation.",
    )
    parser.add_argument(
        "--velocity-head-commands",
        default="stop,speed up,slow down",
        help="Comma-separated commands whose scalar speed is controlled by the velocity head.",
    )
    parser.add_argument(
        "--simlingo-input-mode",
        choices=("target_point_command", "command", "strict-command"),
        default="strict-command",
        help=(
            "Navigation input given to the model. "
            "'target_point_command' is the checkpoint's native route-conditioned mode. "
            "'command' uses route-derived text commands. "
            "'strict-command' uses only the explicit Task 1 command text and disables target-point route tensors."
        ),
    )
    parser.add_argument(
        "--simlingo-execution-mode",
        choices=("simlingo-control", "cached-route-follower"),
        default="cached-route-follower",
        help=(
            "'simlingo-control' applies SimLingo's own VehicleControl output. "
            "'cached-route-follower' calls SimLingo for a trajectory, caches it in world coordinates, "
            "and follows that cached trajectory between policy calls."
        ),
    )
    parser.add_argument(
        "--simlingo-controller-inference-mode",
        action="store_true",
        default=True,
        help="Use SimLingo's sparse predicted-waypoint lateral controller lookahead. This is the correct mode for model inference.",
    )
    parser.add_argument(
        "--no-simlingo-controller-inference-mode",
        dest="simlingo_controller_inference_mode",
        action="store_false",
        help="Use the expert-route lateral controller lookahead. Mostly useful for A/B debugging.",
    )
    parser.add_argument(
        "--simlingo-speed-control-mode",
        choices=("route", "model", "model-speed-commands"),
        default="model-speed-commands",
        help=(
            "Longitudinal control source. 'model' converts SimLingo's predicted speed waypoints into a scalar target speed; "
            "'route' derives speed from command/route fallback settings; "
            "'model-speed-commands' uses model speed only for stop/speed-up/slow-down."
        ),
    )
    parser.add_argument(
        "--speed-command-route-source",
        choices=("map", "model"),
        default="map",
        help=(
            "Steering route source for stop/speed-up/slow-down tests. "
            "'map' keeps lane-follow steering stable so the test measures scalar speed behavior; "
            "'model' lets SimLingo route predictions control steering."
        ),
    )
    parser.add_argument(
        "--task1-route-speed-scale",
        type=float,
        default=0.60,
        help="Scale applied only to route-derived speed waypoints. Lower values reduce turn entry speed.",
    )
    parser.add_argument(
        "--task1-max-throttle",
        type=float,
        default=0.45,
        help="Task 1 throttle cap for smoother closed-loop tracking.",
    )
    parser.add_argument(
        "--task1-brake-ratio",
        type=float,
        default=1.35,
        help="Task 1 brake trigger ratio. Higher values reduce full-brake oscillation.",
    )
    parser.add_argument(
        "--task1-clip-delta",
        type=float,
        default=0.50,
        help="Task 1 longitudinal PID speed-error cap.",
    )
    parser.add_argument("--cached-route-min-lookahead-m", type=float, default=2.0)
    parser.add_argument("--cached-route-base-lookahead-m", type=float, default=3.0)
    parser.add_argument("--cached-route-speed-lookahead-gain", type=float, default=0.35)
    parser.add_argument("--cached-route-max-lookahead-m", type=float, default=6.0)
    parser.add_argument("--cached-route-steer-gain", type=float, default=1.35)
    parser.add_argument("--cached-route-max-steer", type=float, default=0.75)
    parser.add_argument(
        "--cached-route-blend-new-weight",
        type=float,
        default=0.30,
        help="At each model refresh, blend this fraction of the new route into the previous cached route.",
    )
    parser.add_argument(
        "--cached-route-max-steer-rate-per-sec",
        type=float,
        default=1.1,
        help="Maximum steering change per second in cached-route-follower mode.",
    )
    parser.add_argument("--cached-route-turn-speed-mps", type=float, default=3.5)
    parser.add_argument("--cached-route-cruise-speed-mps", type=float, default=5.0)
    parser.add_argument("--cached-route-fast-speed-mps", type=float, default=6.0)
    parser.add_argument("--cached-route-slow-speed-mps", type=float, default=2.2)
    parser.add_argument("--cached-route-throttle-gain", type=float, default=0.25)
    parser.add_argument(
        "--cached-route-throttle-feedforward",
        type=float,
        default=0.0,
        help="Constant throttle term used when tracking a positive speed error.",
    )
    parser.add_argument(
        "--cached-route-speed-throttle-feedforward-gain",
        type=float,
        default=0.055,
        help="Target-speed-proportional throttle term for scalar speed tracking.",
    )
    parser.add_argument(
        "--cached-route-throttle-deadband-mps",
        type=float,
        default=0.05,
        help="No-throttle band around the target speed to avoid creeping/oscillation.",
    )
    parser.add_argument("--cached-route-max-throttle", type=float, default=0.65)
    parser.add_argument("--cached-route-brake-gain", type=float, default=0.35)
    parser.add_argument("--cached-route-max-brake", type=float, default=0.6)
    parser.add_argument("--cached-route-brake-margin-mps", type=float, default=0.7)
    parser.add_argument(
        "--refresh-policy-on-route-exhaustion",
        action="store_true",
        default=True,
        help="Refresh SimLingo immediately when the cached trajectory has too few points ahead.",
    )
    parser.add_argument(
        "--no-refresh-policy-on-route-exhaustion",
        dest="refresh_policy_on_route_exhaustion",
        action="store_false",
    )
    parser.add_argument("--min-cached-route-ahead-points", type=int, default=3)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--command", choices=BASIC_COMMANDS)
    group.add_argument("--eval-suite", choices=("basic",))
    parser.add_argument("--trials-per-command", type=int, default=1)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--eval-duration-overrides",
        default="turn left:35,turn right:35,straight:12,stop:8,speed up:12,slow down:12",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--sim-fps", type=int, default=20)
    parser.add_argument("--policy-hz", type=float, default=0.20)
    parser.add_argument(
        "--prime-policy-before-start",
        action="store_true",
        default=True,
        help="Call SimLingo once before the measured driving loop so the first loop step has a real model prediction.",
    )
    parser.add_argument("--no-prime-policy-before-start", dest="prime_policy_before_start", action="store_false")
    parser.add_argument("--prime-policy-steps", type=int, default=1)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=110.0)
    parser.add_argument("--topdown-z", type=float, default=55.0)
    parser.add_argument("--camera-view", choices=("front", "topdown", "chase"), default="topdown")
    parser.add_argument("--overlay-predictions", action="store_true", default=True)
    parser.add_argument("--no-overlay-predictions", dest="overlay_predictions", action="store_false")
    parser.add_argument(
        "--prediction-overlay-pixels-per-meter",
        type=float,
        default=0.0,
        help="Topdown overlay scale. <=0 derives scale from topdown camera height/FOV.",
    )
    parser.add_argument("--target-road-overlay-length-m", type=float, default=34.0)
    parser.add_argument("--target-road-overlay-width-m", type=float, default=8.0)
    parser.add_argument("--sensor-timeout", type=float, default=5.0)
    parser.add_argument("--sensor-warmup-ticks", type=int, default=40)
    parser.add_argument("--sensor-retry-ticks", type=int, default=10)

    parser.add_argument("--spawn-policy", choices=("index", "junction"), default="junction")
    parser.add_argument("--spawn-index", type=int)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--max-spawn-candidates", type=int, default=600)
    parser.add_argument(
        "--min-spawn-distance-m",
        type=float,
        default=35.0,
        help="Prefer accepted trial spawns at least this far from previous accepted spawns.",
    )
    parser.add_argument(
        "--spawn-diversity-scope",
        choices=("command", "global"),
        default="command",
        help="'command' spaces trials within each command; 'global' spaces all commands from each other.",
    )
    parser.add_argument("--relax-spawn-distance", action="store_true", default=True)
    parser.add_argument("--no-relax-spawn-distance", dest="relax_spawn_distance", action="store_false")
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--route-step-m", type=float, default=2.0)
    parser.add_argument("--turn-route-length-m", type=float, default=140.0)
    parser.add_argument("--speed-route-length-m", type=float, default=100.0)
    parser.add_argument("--route-planner-min-distance", type=float, default=20.0)
    parser.add_argument("--route-planner-max-distance", type=float, default=80.0)
    parser.add_argument("--junction-min-turn-deg", type=float, default=35.0)
    parser.add_argument(
        "--min-pre-junction-route-points",
        type=int,
        default=4,
        help="Reject left/right/straight test spawns whose requested junction branch is too close to the spawn.",
    )
    parser.add_argument(
        "--straight-require-junction",
        action="store_true",
        default=True,
        help="Require straight trials to use an actual intersection straight branch.",
    )
    parser.add_argument(
        "--no-straight-require-junction",
        dest="straight_require_junction",
        action="store_false",
    )
    parser.add_argument("--straight-route-max-heading", type=float, default=12.0)
    parser.add_argument("--speed-route-max-heading", type=float, default=8.0)

    parser.add_argument("--target-success-rate", type=float, default=0.80)
    parser.add_argument("--success-hold-sec", type=float, default=1.0)
    parser.add_argument("--stop-on-success", action="store_true", default=True)
    parser.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
    parser.add_argument("--stop-on-offroad", action="store_true", default=True)
    parser.add_argument("--no-stop-on-offroad", dest="stop_on_offroad", action="store_false")
    parser.add_argument("--target-road-depth-m", type=float, default=65.0)
    parser.add_argument(
        "--straight-target-depth-m",
        type=float,
        default=25.0,
        help="Distance after the straight junction branch used for straight-command success.",
    )
    parser.add_argument(
        "--straight-target-min-distance-m",
        type=float,
        default=40.0,
        help="Minimum ego travel distance before a straight-through-intersection trial can succeed.",
    )
    parser.add_argument(
        "--straight-post-junction-min-distance-m",
        type=float,
        default=10.0,
        help="Minimum distance past the selected straight junction branch before straight can succeed.",
    )
    parser.add_argument("--target-road-min-heading-deg", type=float, default=75.0)
    parser.add_argument("--target-road-reach-distance-m", type=float, default=8.0)
    parser.add_argument(
        "--target-road-require-distance",
        action="store_true",
        help="Also require ego center to be near the generated target point. Off by default because same-road/alignment/hold is the Task 1 success criterion.",
    )
    parser.add_argument("--target-road-max-heading-error-deg", type=float, default=10.0)
    parser.add_argument("--target-road-max-lane-center-distance-m", type=float, default=1.8)
    parser.add_argument("--target-road-body-sample-fraction", type=float, default=0.7)
    parser.add_argument("--target-road-body-margin-m", type=float, default=0.0)
    parser.add_argument("--max-collision-events", type=int, default=0)
    parser.add_argument("--max-offroad-frames", type=int, default=0)
    parser.add_argument("--allow-offroad-success", action="store_true")
    parser.add_argument("--turn-success-min-heading-deg", type=float, default=55.0)
    parser.add_argument("--turn-success-max-lane-heading-error-deg", type=float, default=18.0)
    parser.add_argument("--turn-success-max-lane-center-distance-m", type=float, default=3.0)
    parser.add_argument("--straight-min-distance", type=float, default=8.0)
    parser.add_argument("--straight-max-heading", type=float, default=8.0)
    parser.add_argument("--stop-speed-threshold", type=float, default=0.35)
    parser.add_argument("--stop-warmup-sec", type=float, default=3.0)
    parser.add_argument("--speed-command-warmup-sec", type=float, default=3.0)
    parser.add_argument(
        "--speed-baseline-window-sec",
        type=float,
        default=1.5,
        help="For stop/speed-up/slow-down, compare command response against this final slice of neutral driving.",
    )
    parser.add_argument(
        "--speed-hold-sec",
        type=float,
        default=2.0,
        help="For stop/speed-up/slow-down, require the speed condition to remain true for this long.",
    )
    parser.add_argument(
        "--stop-baseline-min-speed-mps",
        type=float,
        default=1.0,
        help="Stop command only counts after the neutral baseline phase proves the car was moving first.",
    )
    parser.add_argument("--speed-delta-threshold", type=float, default=0.8)
    parser.add_argument(
        "--slowdown-min-success-speed-mps",
        type=float,
        default=1.0,
        help="Slow-down success requires the car to remain moving at least this fast.",
    )
    parser.add_argument(
        "--force-policy-on-command-change",
        action="store_true",
        default=True,
        help="Force a fresh SimLingo call on the tick where stop/speed-up/slow-down becomes active.",
    )
    parser.add_argument(
        "--no-force-policy-on-command-change",
        dest="force_policy_on_command_change",
        action="store_false",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "debug-imports":
        return debug_imports(args)
    if args.mode == "smoke-model":
        return smoke_model(args)
    if args.mode == "camera-smoke":
        return camera_smoke(args)
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --mode run")
    if not args.command and not args.eval_suite:
        raise SystemExit("Either --command or --eval-suite is required for --mode run.")
    return Task1SimLingoRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
