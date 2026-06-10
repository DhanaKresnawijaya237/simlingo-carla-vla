#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def command_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in results:
        command = str(row.get("command", "unknown"))
        item = stats.setdefault(command, {"total": 0, "successes": 0, "success_rate": 0.0})
        item["total"] += 1
        item["successes"] += int(bool(row.get("success")))
    for item in stats.values():
        total = int(item["total"])
        item["success_rate"] = float(item["successes"] / total) if total else 0.0
    return dict(sorted(stats.items()))


def command_macro_stats(map_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_command: dict[str, list[float]] = {}
    for item in map_summaries:
        for command, stats in item.get("command_stats", {}).items():
            by_command.setdefault(command, []).append(float(stats.get("success_rate", 0.0)))
    out = {}
    for command, rates in sorted(by_command.items()):
        out[command] = {
            "map_count": len(rates),
            "average_map_success_rate": sum(rates) / len(rates) if rates else 0.0,
            "min_map_success_rate": min(rates) if rates else 0.0,
            "max_map_success_rate": max(rates) if rates else 0.0,
        }
    return out


def flatten_result(map_label: str, row: dict[str, Any]) -> dict[str, Any]:
    out = {"map_label": map_label}
    out.update(row)
    return out


def aggregate(root: Path, target_success_rate: float) -> dict[str, Any]:
    map_summaries = []
    all_results = []
    for summary_path in sorted(root.glob("*/summary.json")):
        map_dir = summary_path.parent
        summary = load_summary(summary_path)
        map_label = map_dir.name
        results = [flatten_result(map_label, row) for row in summary.get("results", [])]
        total = int(summary.get("total_trials", len(results)))
        successes = int(summary.get("successes", sum(1 for row in results if row.get("success"))))
        success_rate = float(summary.get("success_rate", successes / total if total else 0.0))
        map_summaries.append(
            {
                "map_label": map_label,
                "carla_map": summary.get("map"),
                "total_trials": total,
                "successes": successes,
                "success_rate": success_rate,
                "passed_target": success_rate >= target_success_rate if total else False,
                "command_stats": command_stats(results),
                "summary_path": str(summary_path),
            }
        )
        all_results.extend(results)

    total_trials = len(all_results)
    successes = sum(1 for row in all_results if row.get("success"))
    weighted_success_rate = successes / total_trials if total_trials else 0.0
    map_success_rates = [float(item["success_rate"]) for item in map_summaries if item["total_trials"]]
    macro_map_success_rate = sum(map_success_rates) / len(map_success_rates) if map_success_rates else 0.0
    return {
        "controller": "simlingo_lingoagent",
        "executable_action_schema": ["steer", "throttle", "brake"],
        "root": str(root),
        "maps": map_summaries,
        "map_count": len(map_summaries),
        "total_trials": total_trials,
        "successes": successes,
        "success_rate": weighted_success_rate,
        "macro_map_success_rate": macro_map_success_rate,
        "target_success_rate": target_success_rate,
        "passed_target": weighted_success_rate >= target_success_rate if total_trials else False,
        "command_stats": command_stats(all_results),
        "command_macro_stats": command_macro_stats(map_summaries),
        "results": all_results,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "map_label",
        "command",
        "trial_index",
        "success",
        "reason",
        "duration_sec",
        "frames",
        "distance_m",
        "heading_delta_deg",
        "max_speed_mps",
        "min_speed_mps",
        "final_speed_mps",
        "collision_count",
        "offroad_frames",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_failure_analysis(path: Path, summary: dict[str, Any]) -> None:
    failures = [row for row in summary["results"] if not row.get("success")]
    lines = [
        "# SimLingo Task 1 Multi-Map Failure Analysis",
        "",
        "Executable action: `carla.VehicleControl(steer, throttle, brake)`",
        f"Weighted success rate: {summary['success_rate']:.3f}",
        f"Average map success rate: {summary['macro_map_success_rate']:.3f}",
        f"Maps: {summary['map_count']}",
        f"Trials: {summary['total_trials']}",
        "",
        "## Per-Map Summary",
        "",
    ]
    for item in summary["maps"]:
        lines.append(
            f"- `{item['map_label']}`: {item['successes']}/{item['total_trials']} "
            f"({item['success_rate']:.3f})"
        )
    lines.extend(["", "## Per-Command Summary", ""])
    for command, item in summary["command_stats"].items():
        lines.append(f"- `{command}`: {item['successes']}/{item['total']} ({item['success_rate']:.3f})")
    lines.extend(["", "## Per-Command Average Across Maps", ""])
    for command, item in summary["command_macro_stats"].items():
        lines.append(
            f"- `{command}`: avg={item['average_map_success_rate']:.3f}, "
            f"min={item['min_map_success_rate']:.3f}, max={item['max_map_success_rate']:.3f} "
            f"over {item['map_count']} maps"
        )
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(
                f"- `{row.get('map_label')}` `{row.get('command')}` trial {row.get('trial_index')}: "
                f"{row.get('reason')}"
            )
    else:
        lines.append("No failures.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Task 1 SimLingo eval summaries across maps.")
    parser.add_argument("--root", type=Path, required=True, help="Parent folder containing <map>/summary.json files.")
    parser.add_argument("--target-success-rate", type=float, default=0.80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root
    summary = aggregate(root, args.target_success_rate)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(root / "summary.csv", summary["results"])
    write_failure_analysis(root / "failure_analysis.md", summary)
    print(json.dumps({k: summary[k] for k in ("map_count", "total_trials", "successes", "success_rate", "macro_map_success_rate", "passed_target")}, indent=2))
    return 0 if summary["passed_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
