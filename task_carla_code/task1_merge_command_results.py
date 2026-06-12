#!/usr/bin/env python3
"""Replace selected Task 1 command results and rebuild evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task1_aggregate_multimap import aggregate, write_csv, write_failure_analysis


def parse_commands(raw: str) -> list[str]:
    commands = [part.strip().lower().replace("_", " ") for part in raw.split(",") if part.strip()]
    if not commands:
        raise ValueError("At least one command is required.")
    return commands


def command_dir_name(command: str) -> str:
    return command.replace(" ", "_")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_map_summary(map_dir: Path, target_success_rate: float) -> dict[str, Any]:
    old_summary_path = map_dir / "summary.json"
    old_summary = load_json(old_summary_path) if old_summary_path.exists() else {}
    results = []
    for result_path in sorted(map_dir.glob("*/trial_*/result.json")):
        row = load_json(result_path)
        row["output_dir"] = str(result_path.parent)
        result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        results.append(row)

    successes = sum(1 for row in results if row.get("success"))
    total = len(results)
    success_rate = successes / total if total else 0.0
    summary = {
        "controller": old_summary.get("controller", "simlingo_lingoagent"),
        "executable_action_schema": old_summary.get(
            "executable_action_schema",
            ["steer", "throttle", "brake"],
        ),
        "checkpoint": old_summary.get("checkpoint"),
        "map": old_summary.get("map"),
        "total_trials": total,
        "successes": successes,
        "success_rate": success_rate,
        "target_success_rate": target_success_rate,
        "passed_target": success_rate >= target_success_rate if total else False,
        "results": results,
    }
    old_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in results for key in row})
    with (map_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    failures = [row for row in results if not row.get("success")]
    lines = [
        "# SimLingo Task 1 Failure Analysis",
        "",
        "Executable action: `carla.VehicleControl(steer, throttle, brake)`",
        f"Success rate: {success_rate:.3f}",
        "",
    ]
    if failures:
        lines.extend(
            f"- `{row.get('command')}` trial {row.get('trial_index')}: {row.get('reason')}"
            for row in failures
        )
    else:
        lines.append("No failures.")
    (map_dir / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def refresh_aggregate(root: Path, target_success_rate: float) -> dict[str, Any]:
    summary = aggregate(root, target_success_rate)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(root / "summary.csv", summary["results"])
    write_failure_analysis(root / "failure_analysis.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-map-dir", type=Path, required=True)
    parser.add_argument("--source-map-dir", type=Path, required=True)
    parser.add_argument("--commands", default="stop,speed up,slow down")
    parser.add_argument("--target-success-rate", type=float, default=0.80)
    args = parser.parse_args()

    target_map_dir = args.target_map_dir.resolve()
    source_map_dir = args.source_map_dir.resolve()
    if not target_map_dir.is_dir():
        raise FileNotFoundError(f"Target map directory does not exist: {target_map_dir}")
    if not source_map_dir.is_dir():
        raise FileNotFoundError(f"Source map directory does not exist: {source_map_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_root = target_map_dir / ".merge_backups" / stamp
    merged = []
    for command in parse_commands(args.commands):
        folder = command_dir_name(command)
        source = source_map_dir / folder
        target = target_map_dir / folder
        if not source.is_dir():
            raise FileNotFoundError(f"Missing source command directory: {source}")
        source_results = list(source.glob("trial_*/result.json"))
        if not source_results:
            raise RuntimeError(f"No result.json files found under source command directory: {source}")

        if target.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(backup_root / folder))
        shutil.copytree(source, target)
        merged.append(
            {
                "command": command,
                "source": str(source),
                "target": str(target),
                "trials": len(source_results),
            }
        )

    map_summary = write_map_summary(target_map_dir, args.target_success_rate)
    aggregate_summary = refresh_aggregate(target_map_dir.parent, args.target_success_rate)
    report = {
        "target_map_dir": str(target_map_dir),
        "source_map_dir": str(source_map_dir),
        "backup_root": str(backup_root) if backup_root.exists() else None,
        "merged": merged,
        "map_total_trials": map_summary["total_trials"],
        "map_success_rate": map_summary["success_rate"],
        "aggregate_total_trials": aggregate_summary["total_trials"],
        "aggregate_success_rate": aggregate_summary["success_rate"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
