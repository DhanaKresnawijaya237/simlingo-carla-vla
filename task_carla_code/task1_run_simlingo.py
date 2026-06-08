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
import sys
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import cv2
import numpy as np


BASIC_COMMANDS = ("turn left", "turn right", "straight", "stop", "speed up", "slow down")


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

    def __init__(self, checkpoint: pathlib.Path, output_dir: pathlib.Path):
        self.checkpoint = checkpoint
        self.output_dir = output_dir
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
                return super(Task1LingoAgent, inner_self).control_pid(route_waypoints, velocity, speed_waypoints)

        self.agent = Task1LingoAgent()
        self.agent.setup(str(self.checkpoint), route_index="task1")
        self.agent.config.eval_route_as = getattr(self.agent.model, "route_as", self.agent.config.eval_route_as)
        print(f"Using SimLingo route prompt mode: {self.agent.config.eval_route_as}", flush=True)

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
        self.agent.user_flag = 1 if self.agent.custom_prompt else None
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
        self.agent.user_flag = 1 if self.agent.custom_prompt else None

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
            "predicted_route": getattr(self.agent, "task1_last_pred_route", None),
            "predicted_speed_waypoints": getattr(self.agent, "task1_last_pred_speed_waypoints", None),
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

        self.policy = SimLingoTask1Policy(args.checkpoint, self.output_dir)
        self.policy.load()
        if self.policy.agent is not None:
            self.policy.agent.route_planner_min_distance = float(args.route_planner_min_distance)
            self.policy.agent.route_planner_max_distance = float(args.route_planner_max_distance)
        if self.policy.RoadOption is None:
            self.RoadOption = FallbackRoadOption
        else:
            self.RoadOption = self.policy.RoadOption

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

            start_transform = vehicle.get_transform()
            start_location = start_transform.location
            start_yaw = start_transform.rotation.yaw
            speeds: list[float] = []
            post_command_speeds: list[float] = []
            offroad_frames = 0
            success_hold = 0
            reached_success_step: int | None = None
            termination_reason: str | None = None
            min_required_hold = max(1, int(round(self.args.success_hold_sec * self.args.sim_fps)))
            warmup_sec = command_warmup_sec(command, self.args)

            total_steps = int(round(self.args.duration * self.args.sim_fps))
            policy_interval = max(1, int(round(self.args.sim_fps / max(0.1, float(self.args.policy_hz)))))
            control = self.carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            policy_latency_sec = 0.0
            fresh_policy = False
            policy_calls = 0
            with ticks_path.open("w", encoding="utf-8") as ticks_file:
                for step in range(total_steps):
                    timestamp = step / float(self.args.sim_fps)
                    active_command = command if timestamp >= warmup_sec else None
                    self.policy.set_active_command(active_command)
                    fresh_policy = step == 0 or (step % policy_interval == 0)
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
                        if self.args.progress:
                            print(
                                f"[{command} #{trial_index}] policy_call={policy_calls} "
                                f"latency={policy_latency_sec:.3f}s "
                                f"control=({float(control.steer):.3f},"
                                f"{float(control.throttle):.3f},{float(control.brake):.3f})",
                                flush=True,
                            )
                    vehicle.apply_control(control)

                    rgb_image, record_image = self.wait_for_sensor_frame(
                        image_queue=image_queue,
                        video_queue=video_queue if record_sensor is not None else None,
                    )
                    if record_sensor is not None:
                        frame = carla_image_to_bgr(record_image)
                    else:
                        frame = resize_frame(carla_image_to_bgr(rgb_image), self.args.width, self.args.height)
                    if step % max(1, int(round(self.args.sim_fps / self.args.fps))) == 0:
                        frames_for_video.append(frame)

                    speed = speed_mps(vehicle)
                    speeds.append(speed)
                    if timestamp >= warmup_sec:
                        post_command_speeds.append(speed)
                    transform = vehicle.get_transform()
                    offroad = not is_vehicle_on_driving_lane(self.map, transform.location, self.carla)
                    if offroad:
                        offroad_frames += 1
                    target_road_status = self.target_road_status(
                        command,
                        transform.location,
                        scenario,
                        start_yaw,
                        transform.rotation.yaw,
                    )
                    target_road_reached = target_road_status["reached"]
                    status = self.evaluate_instant_status(
                        command=command,
                        vehicle=vehicle,
                        start_yaw=start_yaw,
                        start_location=start_location,
                        speeds=speeds,
                        post_command_speeds=post_command_speeds,
                        offroad_frames=offroad_frames,
                        collision_count=len(collision_events),
                    )
                    if status["condition_met"]:
                        success_hold += 1
                        if success_hold >= min_required_hold and reached_success_step is None:
                            reached_success_step = step
                            termination_reason = "success_condition"
                    else:
                        success_hold = 0
                    if target_road_reached and reached_success_step is None:
                        reached_success_step = step
                        termination_reason = "target_road_reached"

                    diag = self.policy.diagnostics()
                    tick = {
                        "step": step,
                        "time_sec": timestamp,
                        "command": command,
                        "active_command": active_command,
                        "warmup_sec": warmup_sec,
                        "prompt": diag.get("prompt"),
                        "prompt_task": diag.get("prompt_task"),
                        "predicted_route": diag.get("predicted_route"),
                        "predicted_speed_waypoints": diag.get("predicted_speed_waypoints"),
                        "target_points": diag.get("target_points"),
                        "route_command_history": diag.get("route_command_history"),
                        "fresh_policy": bool(fresh_policy),
                        "policy_interval_steps": int(policy_interval),
                        "policy_latency_sec": float(policy_latency_sec if fresh_policy else 0.0),
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
                        "target_road_same_road": target_road_status.get("same_road"),
                        "target_road_reached": bool(target_road_reached),
                        "collision_count": len(collision_events),
                        "instant_success": status,
                        "termination_reason": termination_reason,
                    }
                    ticks_file.write(json.dumps(tick) + "\n")
                    ticks_file.flush()

                    if self.args.stop_on_offroad and offroad:
                        termination_reason = "offroad"
                        break
                    if self.args.stop_on_success and reached_success_step is not None:
                        break

            write_video(video_path, frames_for_video, self.args.fps)
            result = self.build_result(
                command=command,
                trial_index=trial_index,
                run_dir=run_dir,
                vehicle=vehicle,
                start_location=start_location,
                start_yaw=start_yaw,
                speeds=speeds,
                post_command_speeds=post_command_speeds,
                offroad_frames=offroad_frames,
                collision_events=collision_events,
                frames=len(frames_for_video),
                reached_success_step=reached_success_step,
            )
            (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "controller": "simlingo_lingoagent",
                        "executable_action_schema": ["steer", "throttle", "brake"],
                        "checkpoint": str(self.args.checkpoint),
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
        offroad_frames: int,
        collision_events: list[dict[str, Any]],
        frames: int,
        reached_success_step: int | None,
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
            post_command_speeds=post_command_speeds,
            offroad_frames=offroad_frames,
            collision_count=len(collision_events),
        )
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
        post_command_speeds: list[float],
        offroad_frames: int,
        collision_count: int,
    ) -> dict[str, Any]:
        transform = vehicle.get_transform()
        speed_now = speeds[-1] if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        post_min = min(post_command_speeds) if post_command_speeds else speed_now
        post_max = max(post_command_speeds) if post_command_speeds else speed_now
        warmup_speed = speeds[int(min(len(speeds) - 1, max(0, round(command_warmup_sec(command, self.args) * self.args.sim_fps))))] if speeds else 0.0
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
            ok = len(post_command_speeds) > 0 and speed_now <= self.args.stop_speed_threshold
            reason = f"final_speed={speed_now:.2f}"
        elif command == "speed up":
            ok = len(post_command_speeds) > 0 and post_max - warmup_speed >= self.args.speed_delta_threshold
            reason = f"warmup_speed={warmup_speed:.2f}, post_max={post_max:.2f}"
        elif command == "slow down":
            ok = len(post_command_speeds) > 0 and warmup_speed - post_min >= self.args.speed_delta_threshold
            reason = f"warmup_speed={warmup_speed:.2f}, post_min={post_min:.2f}"
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
            "warmup_speed_mps": float(warmup_speed),
        }

    def target_road_status(
        self,
        command: str,
        location: Any,
        scenario: dict[str, Any],
        start_yaw: float,
        current_yaw: float,
    ) -> dict[str, Any]:
        status = {
            "reached": False,
            "same_road": False,
            "distance_m": None,
            "heading_error_deg": None,
        }
        if command not in ("turn left", "turn right"):
            return status
        target_road_id = scenario.get("target_road_id")
        if target_road_id is None:
            return status
        lane_wp = self.map.get_waypoint(
            location,
            project_to_road=False,
            lane_type=self.carla.LaneType.Driving,
        )
        if lane_wp is None:
            return status
        heading_delta = angle_delta_deg(start_yaw, current_yaw)
        if abs(heading_delta) < self.args.target_road_min_heading_deg:
            return status

        same_road = int(lane_wp.road_id) == int(target_road_id)
        status["same_road"] = bool(same_road)
        status["heading_error_deg"] = float(abs(angle_delta_deg(lane_wp.transform.rotation.yaw, current_yaw)))

        target_point = scenario.get("target_road_point") or {}
        if {"x", "y"}.issubset(target_point):
            dx = float(location.x) - float(target_point["x"])
            dy = float(location.y) - float(target_point["y"])
            status["distance_m"] = float((dx * dx + dy * dy) ** 0.5)

        if not same_road:
            return status
        if status["heading_error_deg"] is None or status["heading_error_deg"] > self.args.target_road_max_heading_error_deg:
            return status
        if status["distance_m"] is None or status["distance_m"] > self.args.target_road_reach_distance_m:
            return status

        status["reached"] = True
        return status

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
        for idx in indices[: self.args.max_spawn_candidates]:
            spawn = spawn_points[idx]
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
            if command == "straight" and abs(meta.get("route_heading_delta_deg", 0.0)) > self.args.straight_route_max_heading:
                last_error = f"straight route drifts {meta.get('route_heading_delta_deg')}"
                continue
            scenario = {
                "map": self.map.name,
                "spawn_index": idx,
                "command": command,
                "reason": meta.get("reason", "ok"),
                **meta,
            }
            return spawn, route_plan, scenario
        raise RuntimeError(f"Could not select scenario for {command}: {last_error}")

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
        if branch_index is not None and route:
            target_idx = min(
                len(route) - 1,
                branch_index + max(8, int(round(self.args.target_road_depth_m / self.args.route_step_m))),
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
            "branch_delta_deg": float(chosen_delta),
            "route_heading_delta_deg": float(heading_delta),
            "route_points": len(route),
            "target_road_depth_m": float(self.args.target_road_depth_m),
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


def command_prompt(command: str | None) -> str | None:
    if command == "stop":
        return "Stop the vehicle safely."
    if command == "speed up":
        return "Drive faster while staying in the lane."
    if command == "slow down":
        return "Slow down while staying in the lane."
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


def is_vehicle_on_driving_lane(carla_map: Any, location: Any, carla_module: Any) -> bool:
    wp = carla_map.get_waypoint(location, project_to_road=False, lane_type=carla_module.LaneType.Driving)
    return wp is not None


def carla_image_to_bgra(image: Any) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr.copy()


def carla_image_to_bgr(image: Any) -> np.ndarray:
    return carla_image_to_bgra(image)[:, :, :3]


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


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

    for out_path, fourcc in (
        (path, "mp4v"),
        (path.with_suffix(".avi"), "XVID") if path.suffix.lower() == ".mp4" else (None, ""),
        (path.with_name(path.stem + "_mjpg.avi"), "MJPG") if path.suffix.lower() == ".mp4" else (None, ""),
    ):
        if out_path is None:
            continue
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height))
        if not writer.isOpened():
            continue
        try:
            for frame in normalized_frames:
                writer.write(frame)
        finally:
            writer.release()

    preview_pairs = (
        ("first", 0),
        ("middle", len(normalized_frames) // 2),
        ("last", len(normalized_frames) - 1),
    )
    for label, idx in preview_pairs:
        cv2.imwrite(str(path.with_name(f"{path.stem}_{label}.jpg")), normalized_frames[idx])
    thumbs = []
    for idx in np.linspace(0, len(normalized_frames) - 1, num=min(12, len(normalized_frames)), dtype=int):
        thumbs.append(cv2.resize(normalized_frames[int(idx)], (240, 135), interpolation=cv2.INTER_AREA))
    rows = []
    for row_start in range(0, len(thumbs), 4):
        row = thumbs[row_start: row_start + 4]
        if len(row) < 4:
            row.extend([np.zeros_like(thumbs[0]) for _ in range(4 - len(row))])
        rows.append(np.concatenate(row, axis=1))
    if rows:
        cv2.imwrite(str(path.with_name(f"{path.stem}_contact.jpg")), np.concatenate(rows, axis=0))


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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--command", choices=BASIC_COMMANDS)
    group.add_argument("--eval-suite", choices=("basic",))
    parser.add_argument("--trials-per-command", type=int, default=1)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--eval-duration-overrides",
        default="turn left:35,turn right:35,straight:12,stop:8,speed up:12,slow down:12",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sim-fps", type=int, default=20)
    parser.add_argument("--policy-hz", type=float, default=2.0)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=110.0)
    parser.add_argument("--topdown-z", type=float, default=55.0)
    parser.add_argument("--camera-view", choices=("front", "topdown", "chase"), default="topdown")
    parser.add_argument("--sensor-timeout", type=float, default=5.0)
    parser.add_argument("--sensor-warmup-ticks", type=int, default=20)
    parser.add_argument("--sensor-retry-ticks", type=int, default=5)

    parser.add_argument("--spawn-policy", choices=("index", "junction"), default="junction")
    parser.add_argument("--spawn-index", type=int)
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--max-spawn-candidates", type=int, default=600)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--route-step-m", type=float, default=2.0)
    parser.add_argument("--turn-route-length-m", type=float, default=140.0)
    parser.add_argument("--speed-route-length-m", type=float, default=100.0)
    parser.add_argument("--route-planner-min-distance", type=float, default=20.0)
    parser.add_argument("--route-planner-max-distance", type=float, default=80.0)
    parser.add_argument("--junction-min-turn-deg", type=float, default=35.0)
    parser.add_argument("--straight-route-max-heading", type=float, default=12.0)

    parser.add_argument("--target-success-rate", type=float, default=0.80)
    parser.add_argument("--success-hold-sec", type=float, default=1.0)
    parser.add_argument("--stop-on-success", action="store_true", default=True)
    parser.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
    parser.add_argument("--stop-on-offroad", action="store_true", default=True)
    parser.add_argument("--no-stop-on-offroad", dest="stop_on_offroad", action="store_false")
    parser.add_argument("--target-road-depth-m", type=float, default=55.0)
    parser.add_argument("--target-road-min-heading-deg", type=float, default=70.0)
    parser.add_argument("--target-road-reach-distance-m", type=float, default=12.0)
    parser.add_argument("--target-road-max-heading-error-deg", type=float, default=18.0)
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
    parser.add_argument("--speed-delta-threshold", type=float, default=0.8)
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
