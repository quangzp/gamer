import argparse
import ast
import json
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw challenge CSV files into GAMER MB dataset files.")
    parser.add_argument("--input_dir", type=Path, default=Path("./challenge_data"), help="Directory containing x_train.csv, y_train.csv, x_test.csv, jobs.json.")
    parser.add_argument("--output_root", type=Path, default=Path("./data"), help="Root directory where the converted dataset folder will be created.")
    parser.add_argument("--dataset", type=str, default="JobChallengeMB", help="Output dataset name.")
    parser.add_argument("--repair_text", action="store_true", default=False, help="Repair mojibake-like Unicode text conservatively by stripping combining marks.")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")


def parse_list_column(series: pd.Series, field_name: str) -> list[list[Any]]:
    parsed: list[list[Any]] = []
    for i, raw_value in enumerate(series.tolist()):
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Failed to parse row {i} in column {field_name}: {raw_value!r}") from exc
        if not isinstance(value, list):
            raise ValueError(f"Expected list in column {field_name} at row {i}, got {type(value)}")
        parsed.append(value)
    return parsed


def strip_combining_marks(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    filtered = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", filtered)


def repair_text_value(value: Any, repair_text: bool) -> Any:
    if isinstance(value, dict):
        return {str(k): repair_text_value(v, repair_text) for k, v in value.items()}
    if isinstance(value, list):
        return [repair_text_value(v, repair_text) for v in value]
    if isinstance(value, str):
        repaired = unicodedata.normalize("NFC", value)
        if repair_text:
            repaired = strip_combining_marks(repaired)
        return repaired
    return value


def compute_length_stats(sequences: dict[str, list[Any]]) -> dict[str, float | int]:
    lengths = [len(seq) for seq in sequences.values()]
    return {
        "count": len(lengths),
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "avg": float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }


def main():
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir = args.output_root / args.dataset
    ensure_dir(output_dir)

    x_train = pd.read_csv(input_dir / "x_train.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv")
    x_test = pd.read_csv(input_dir / "x_test.csv")
    jobs = load_json(input_dir / "jobs.json")

    required_x_cols = {"session_id", "job_ids", "actions"}
    required_y_cols = {"session_id", "job_id", "action"}
    if set(x_train.columns) != required_x_cols:
        raise ValueError(f"x_train.csv columns mismatch: expected {required_x_cols}, got {set(x_train.columns)}")
    if set(x_test.columns) != required_x_cols:
        raise ValueError(f"x_test.csv columns mismatch: expected {required_x_cols}, got {set(x_test.columns)}")
    if set(y_train.columns) != required_y_cols:
        raise ValueError(f"y_train.csv columns mismatch: expected {required_y_cols}, got {set(y_train.columns)}")
    if not isinstance(jobs, dict):
        raise ValueError(f"jobs.json must be a JSON object keyed by raw job_id, got {type(jobs)}")

    x_train = x_train.copy()
    x_test = x_test.copy()
    y_train = y_train.copy()
    x_train["job_ids"] = parse_list_column(x_train["job_ids"], "x_train.job_ids")
    x_train["actions"] = parse_list_column(x_train["actions"], "x_train.actions")
    x_test["job_ids"] = parse_list_column(x_test["job_ids"], "x_test.job_ids")
    x_test["actions"] = parse_list_column(x_test["actions"], "x_test.actions")

    merged_train = x_train.merge(y_train, on="session_id", how="inner", validate="one_to_one")
    if len(merged_train) != len(x_train) or len(merged_train) != len(y_train):
        raise ValueError("x_train and y_train do not align one-to-one on session_id.")

    raw_job_ids = sorted(int(job_id) for job_id in jobs.keys())
    raw_job_to_item = {str(raw_job_id): idx for idx, raw_job_id in enumerate(raw_job_ids)}
    item_to_raw_job = {str(item_id): str(raw_job_id) for raw_job_id, item_id in raw_job_to_item.items()}

    def to_dense_job_id(raw_job_id: Any) -> int:
        raw_job_str = str(int(raw_job_id))
        if raw_job_str not in raw_job_to_item:
            raise KeyError(f"Raw job id {raw_job_str} not found in jobs.json")
        return int(raw_job_to_item[raw_job_str])

    item_json: dict[str, Any] = {}
    missing_text_count = 0
    for raw_job_id in raw_job_ids:
        dense_id = raw_job_to_item[str(raw_job_id)]
        payload = jobs[str(raw_job_id)]
        repaired_payload = repair_text_value(payload, args.repair_text)
        if isinstance(repaired_payload, dict):
            title = repaired_payload.get("title", "")
            summary = repaired_payload.get("summary", "")
            if not isinstance(title, str):
                title = str(title)
            if not isinstance(summary, str):
                summary = str(summary)
            if title.strip() == "" and summary.strip() == "":
                missing_text_count += 1
            repaired_payload["raw_job_id"] = int(raw_job_id)
        item_json[str(dense_id)] = repaired_payload

    train_inter: dict[str, list[int]] = {}
    train_behavior: dict[str, list[str]] = {}
    train_uid_to_original_session: dict[str, int] = {}

    for row in merged_train.itertuples(index=False):
        session_id = str(int(row.session_id))
        history_job_ids = list(row.job_ids)
        history_actions = list(row.actions)
        if len(history_job_ids) != len(history_actions):
            raise ValueError(f"Mismatched history lengths for session_id={session_id}")
        full_job_ids = history_job_ids + [int(row.job_id)]
        full_actions = history_actions + [str(row.action)]
        train_inter[session_id] = [to_dense_job_id(job_id) for job_id in full_job_ids]
        train_behavior[session_id] = full_actions
        train_uid_to_original_session[session_id] = int(row.session_id)

    test_inter: dict[str, list[int]] = {}
    test_behavior: dict[str, list[str]] = {}
    test_uid_to_original_session: dict[str, int] = {}
    for row in x_test.itertuples(index=False):
        session_id = str(int(row.session_id))
        history_job_ids = list(row.job_ids)
        history_actions = list(row.actions)
        if len(history_job_ids) != len(history_actions):
            raise ValueError(f"Mismatched x_test history lengths for session_id={session_id}")
        test_inter[session_id] = [to_dense_job_id(job_id) for job_id in history_job_ids]
        test_behavior[session_id] = [str(action) for action in history_actions]
        test_uid_to_original_session[session_id] = int(row.session_id)

    behavior_level = {
        "view": 0,
        "apply": 1,
    }

    dump_json(output_dir / f"{args.dataset}.MB.inter.json", train_inter)
    dump_json(output_dir / f"{args.dataset}.MB.behavior.json", train_behavior)
    dump_json(output_dir / f"{args.dataset}.behavior_level.json", behavior_level)
    dump_json(output_dir / f"{args.dataset}.item.json", item_json)
    dump_json(output_dir / f"{args.dataset}.raw_job_to_item.json", {k: int(v) for k, v in raw_job_to_item.items()})
    dump_json(output_dir / f"{args.dataset}.item_to_raw_job.json", item_to_raw_job)
    dump_json(output_dir / f"{args.dataset}.uid_to_original_session.json", train_uid_to_original_session)

    dump_json(output_dir / f"{args.dataset}.challenge_test.inter.json", test_inter)
    dump_json(output_dir / f"{args.dataset}.challenge_test.behavior.json", test_behavior)
    dump_json(output_dir / f"{args.dataset}.challenge_test.uid_to_original_session.json", test_uid_to_original_session)

    summary = {
        "dataset": args.dataset,
        "repair_text": bool(args.repair_text),
        "train_sessions": len(train_inter),
        "challenge_test_sessions": len(test_inter),
        "catalog_jobs": len(item_json),
        "catalog_missing_title_and_summary": missing_text_count,
        "train_sequence_length": compute_length_stats(train_inter),
        "challenge_test_history_length": compute_length_stats(test_inter),
        "output_dir": str(output_dir.resolve()),
    }
    dump_json(output_dir / f"{args.dataset}.conversion_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
