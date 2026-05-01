#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def format_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_results(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain the expected list structure.")
    merged = None
    per_behavior = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("eval_type") == "Merged Behavior":
            merged = entry
        else:
            per_behavior.append(entry)
    return merged, per_behavior


def print_entry(title, entry):
    print(title)
    keys = [k for k in entry.keys() if k not in {"eval_type", "collision_info"}]
    for key in sorted(keys):
        print(f"  {key}: {format_metric(entry[key])}")


def main(argv):
    if len(argv) < 2:
        print("Usage: summarize_results.py <results.json> [more_results.json ...]")
        return 1

    for raw_path in argv[1:]:
        path = Path(raw_path)
        merged, per_behavior = load_results(path)
        print(f"== {path} ==")
        if merged is not None:
            print_entry("Merged", merged)
        else:
            print("Merged")
            print("  missing: true")
        for entry in per_behavior:
            title = str(entry.get("eval_type", "Behavior"))
            print_entry(title, entry)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
