#!/usr/bin/env python3
"""Prepare the fixed participant split and training-B majority baseline.

This CPU-only entry point supplies the file layout expected by the historical
experiment engines. Existing split/validation manifests are verified and kept
byte-for-byte. When absent, their reconstructed provenance is recorded explicitly.
Only training B and validation B/C answer cells are converted or analyzed;
test answer cells are never converted or used. CSV byte scanning and SHA256
verification necessarily read complete raw files.
"""
import argparse
import csv
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

from download_data import REPO_ID, REVISION, digest_file, verify_dataset

QIDS = ("QID163", "QID165", "QID167", "QID169", "QID183", "QID184",
        "QID192", "QID193", "QID196")
EXPECTED_SIZES = {"train": 1440, "validation": 309, "test": 309}
FIELDS = ("pid", "qid", "target_c", "reference_b", "prediction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_pid(value):
    value = str(value).strip()
    require(value and value.isdigit(), f"Invalid participant ID: {value!r}")
    return value


def read_ids(path):
    ids = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and "pid" in reader.fieldnames, "Missing PID column")
        for row in reader:
            ids.append(canonical_pid(row["pid"]))
    require(len(ids) == len(set(ids)) == 2058, "Expected 2,058 unique participants")
    return ids


def reconstructed_split(ids):
    """Exactly the established EDA recipe; string sorting is intentional."""
    ids = sorted(map(canonical_pid, ids))
    require(len(ids) == len(set(ids)) == 2058 and "574" in ids,
            "Split input must contain 2,058 unique PIDs including 574")
    remaining = [pid for pid in ids if pid != "574"]
    random.Random(42).shuffle(remaining)
    return {"train": ["574"] + remaining[:1439],
            "validation": remaining[1439:1748], "test": remaining[1748:]}


def resolve_split(ids, path, source=None):
    """Preserve supplied ordering, but refuse a changed participant protocol."""
    expected = reconstructed_split(ids)
    candidate = Path(source) if source else Path(path)
    if candidate.exists():
        split = {name: [] for name in EXPECTED_SIZES}
        seen = set()
        with candidate.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames and {"pid", "split"}.issubset(reader.fieldnames),
                    "Invalid participant split columns")
            for row in reader:
                pid, name = canonical_pid(row["pid"]), row["split"]
                require(pid not in seen and name in split, "Invalid/duplicate split row")
                split[name].append(pid)
                seen.add(pid)
        require(seen == set(map(str, ids)), "Dataset/split participant IDs differ")
        for name in split:
            require(set(split[name]) == set(expected[name]),
                    f"Existing {name} membership differs from the frozen seed-42 protocol")
        provenance = "existing_manifest_verified_and_preserved"
        if candidate.resolve() != Path(path).resolve():
            require(not Path(path).exists(), "Destination split already exists; do not overwrite it")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(candidate.read_bytes())
    else:
        require(not source, f"Supplied split file does not exist: {source}")
        split, provenance = expected, "reconstructed_seed42_pid574_reserved_train"
        rows = [{"pid": pid, "split": name} for name, people in split.items() for pid in people]
        write_csv(path, ("pid", "split"), rows)
    require({name: len(people) for name, people in split.items()} == EXPECTED_SIZES,
            "Incorrect participant split sizes")
    return split, provenance


def parse_selected(raw, pid, column):
    require(raw is not None, f"Malformed CSV cell: {pid}/{column}")
    stripped = raw.strip()
    if stripped.lower() in {"", "nan", "na", "n/a", "none", "null"}:
        return None
    try:
        number = float(stripped)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid numeric selected label: {pid}/{column}") from error
    require(math.isfinite(number), f"Non-finite selected label: {pid}/{column}")
    return int(number) if number.is_integer() else number


def selected_labels(path, pids, columns):
    """Discard unselected rows before examining any answer values."""
    wanted, columns = set(map(str, pids)), tuple(columns)
    rows, seen = {}, set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and set(("pid", *columns)).issubset(reader.fieldnames),
                "Missing requested response columns")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV headers")
        for row in reader:
            pid = canonical_pid(row["pid"])
            require(pid not in seen, "Duplicate response participant")
            seen.add(pid)
            if pid not in wanted:
                continue
            rows[pid] = {column: parse_selected(row[column], pid, column) for column in columns}
    require(set(rows) == wanted, "Missing requested participants")
    return rows


def mode_lowest(values):
    counts = Counter(value for value in values if value is not None)
    require(bool(counts), "Cannot fit majority to an empty training target")
    highest = max(counts.values())
    return min(value for value, count in counts.items() if count == highest)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def baseline_rows(train_b, validation_b, validation_c, validation_order, modes=None):
    modes = modes or {qid: mode_lowest(row[qid] for row in train_b.values()) for qid in QIDS}
    rows = []
    for pid in validation_order:
        for qid in QIDS:
            b, c = validation_b[pid][qid], validation_c[pid][qid]
            require(b is None or b in (1, 2), f"Invalid validation B code: {pid}/{qid}")
            require(c is None or c in (1, 2), f"Invalid validation C code: {pid}/{qid}")
            # The evaluation population is defined by observed C; no missing cells
            # are fabricated. B/C coverage is then required to match this protocol.
            if c is not None:
                rows.append(dict(pid=pid, qid=qid, target_c=c, reference_b=b,
                                 prediction=modes[qid]))
    require(len(rows) == 1545, "Expected the established 1,545 validation examples")
    require(all(row["reference_b"] in (1, 2) for row in rows), "B/C pair coverage changed")
    return rows, modes


def preserve_or_write_baseline(path, rows):
    path = Path(path)
    if not path.exists():
        write_csv(path, FIELDS, rows)
        return "reconstructed_training_B_mode_validation_C_manifest"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and set(FIELDS).issubset(reader.fieldnames),
                "Existing majority manifest is missing expected fields")
        actual = {}
        for row in reader:
            key = (row["pid"], row["qid"])
            require(key not in actual, "Duplicate existing majority key")
            actual[key] = {field: parse_selected(row[field], row["pid"], field)
                           for field in FIELDS[2:]}
    expected = {(row["pid"], row["qid"]): {field: row[field] for field in FIELDS[2:]}
                for row in rows}
    require(actual == expected, "Existing majority predictions/labels differ; refusing overwrite")
    return "existing_majority_manifest_verified_and_preserved"


def calculate_metrics(rows):
    metrics = []
    for qid in QIDS:
        group = [row for row in rows if row["qid"] == qid]
        correct = sum(row["prediction"] == row["target_c"] for row in group)
        human = sum(row["reference_b"] == row["target_c"] for row in group)
        metrics.append({"qid": qid, "examples": len(group),
                        "majority_accuracy": correct / len(group),
                        "human_pairs": len(group), "human_reference": human / len(group),
                        "majority_accuracy_paired": correct / len(group)})
    summary = {"evaluation_split": "validation", "evaluation_labels": "C",
               "training_labels": "B", "question_subset": list(QIDS), "examples": len(rows),
               "question_macro_accuracy": sum(row["majority_accuracy"] for row in metrics) / len(QIDS),
               "pooled_accuracy": sum(row["prediction"] == row["target_c"] for row in rows) / len(rows),
               "human_reference_macro_paired": sum(row["human_reference"] for row in metrics) / len(QIDS),
               "invalid_outputs": sum(row["prediction"] not in (1,2) for row in rows),
               "majority_macro_accuracy": sum(row["majority_accuracy"] for row in metrics) / len(QIDS),
               "final_test_evaluated": False}
    return metrics, summary


def prepare(root, dataset_root=None, supplied_split=None):
    root = Path(root).resolve()
    dataset = root / "data/Twin-2K-500"
    if dataset_root:
        source = Path(dataset_root).resolve()
        require(source.is_dir(), f"Local dataset directory does not exist: {source}")
        if dataset.exists():
            require(dataset.resolve() == source, "Dataset destination already points elsewhere")
        else:
            dataset.parent.mkdir(parents=True, exist_ok=True)
            dataset.symlink_to(source, target_is_directory=True)
    hashes = verify_dataset(dataset)
    csv_folder = dataset / "question_catalog_and_human_response_csv"
    earlier, later = csv_folder / "wave1_3_response.csv", csv_folder / "wave4_response.csv"
    ids = read_ids(earlier)
    require(set(read_ids(later)) == set(ids), "Earlier/later participants differ")
    split_path = root / "data/participant_split.csv"
    split, split_provenance = resolve_split(ids, split_path, supplied_split)
    train = selected_labels(earlier, split["train"], QIDS)
    require(all(value is None or value in (1, 2) for row in train.values() for value in row.values()),
            "Invalid training binary labels")
    train_count = sum(value is not None for row in train.values() for value in row.values())
    require(train_count == 7200, "Expected 7,200 training binary examples")
    modes = {qid: mode_lowest(row[qid] for row in train.values()) for qid in QIDS}
    # C is consulted only here, after modes have been fitted on training B.
    validation_b = selected_labels(earlier, split["validation"], QIDS)
    validation_c = selected_labels(later, split["validation"], QIDS)
    rows, modes = baseline_rows(train, validation_b, validation_c, split["validation"], modes)
    baseline = root / "outputs/majority_baseline_v1"
    provenance = preserve_or_write_baseline(baseline / "predictions.csv", rows)
    metrics, summary = calculate_metrics(rows)
    # Never replace legacy summaries: source engines may have recorded their hashes.
    if not (baseline / "metrics.csv").exists():
        write_csv(baseline / "metrics.csv", tuple(metrics[0]), metrics)
    if not (baseline / "summary.json").exists():
        (baseline / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with later.open(newline="", encoding="utf-8-sig") as handle:
        all_b_columns = tuple(column for column in next(csv.reader(handle)) if column != "pid")
    require(len(all_b_columns) == 126, "Expected 126 global B/C scalar columns")
    full_train = selected_labels(earlier, split["train"], all_b_columns)
    full_count = sum(value is not None for row in full_train.values() for value in row.values())
    require(full_count == 138264, "Expected 138,264 administered full-B training scalar examples")
    report = {"protocol": "twin-fixed-participants-training-B-validation-C-preparation-v1",
              "repo_id": REPO_ID, "revision": REVISION,
              "split_provenance": split_provenance, "baseline_provenance": provenance,
              "split_sizes": {name: len(people) for name, people in split.items()},
              "training_binary_examples": train_count, "validation_binary_examples": len(rows),
              "training_full_B_scalar_examples": full_count, "global_B_scalar_columns": len(all_b_columns),
              "majority_training_B_modes": modes, "majority_validation_metrics": summary,
              "dataset_sha256": hashes, "split_sha256": digest_file(split_path),
              "validation_manifest_sha256": digest_file(baseline / "predictions.csv"),
              "test_answer_cells_analyzed": 0,
              "scope_note": "CSV/hash byte scans cover files; only training B and validation B/C answer cells are converted or analyzed."}
    report_path = root / "data/preparation_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-root", "--dataset-dir", dest="dataset_root", type=Path,
                        help="Existing pinned local dataset repository")
    parser.add_argument("--split-manifest", type=Path, help="Existing original participant_split.csv")
    args = parser.parse_args()
    report = prepare(args.root, args.dataset_root, args.split_manifest)
    summary = report["majority_validation_metrics"]
    print(json.dumps({key: report[key] for key in ("split_provenance", "baseline_provenance", "split_sizes",
          "training_binary_examples", "validation_binary_examples", "training_full_B_scalar_examples")}, indent=2))
    print(f"Majority validation macro accuracy: {100*summary['question_macro_accuracy']:.2f}%")
    print(f"Majority validation pooled accuracy: {100*summary['pooled_accuracy']:.2f}%")
    print(f"Human B/C reference macro accuracy: {100*summary['human_reference_macro_paired']:.2f}%")
    print("No final test evaluation was performed.")


if __name__ == "__main__":
    main()
