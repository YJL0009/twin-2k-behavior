#!/usr/bin/env python3
"""Extend EDA using saved validation aggregates and optional frozen training data.

No model, GPU, or network access is required. New raw-answer analysis is restricted
to training A/B. The C parquet column and wave4 response CSV are never opened.
Saved validation human-retest aggregates are reported separately, with provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
DEFAULT_RESULTS = REPO / "docs/eda/source"
DEFAULT_OUTPUT = REPO.parent / "twin_work/eda_review"
GROUPS = (
    ("pricing_40", "Pricing", "Agreement ↑"),
    ("original_9_binary", "Pilot binary subset", "Agreement ↑"),
    ("family:matrix_nominal", "Nominal matrix", "Agreement ↑"),
    ("family:ordinal", "Ordinal", "Normalized absolute difference ↓"),
    ("family:matrix_ordinal", "Ordinal matrix", "Normalized absolute difference ↓"),
    ("family:bounded_numeric", "Bounded numeric", "Normalized absolute difference ↓"),
)
# Verified against the pinned catalog's preceding anchoring questions and the
# numeric targets. Anchors are questionnaire context, not reference answers.
UNBOUNDED_TARGETS = {
    "QID164": ("Number of African countries in the UN; preceding anchor: 12 countries", "countries"),
    "QID166": ("Number of African countries in the UN; preceding anchor: 65 countries", "countries"),
    "QID168": ("Height of the tallest redwood; preceding anchor: 85 feet", "feet"),
    "QID170": ("Height of the tallest redwood; preceding anchor: 1,000 feet", "feet"),
}
# Exact A definitions inspected in the hash-pinned source catalog. Sex assigned
# at birth must not be relabeled as gender; income is family income before tax.
DEMOGRAPHICS = {
    "US region": "QID11", "Sex assigned at birth": "QID12", "Age": "QID13",
    "Education": "QID14", "Race or origin": "QID15", "US citizenship": "QID16",
    "Family income before tax": "QID21", "Employment": "QID24",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def archived_retest(result_dir):
    path = Path(result_dir) / "comparison.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    config = json.loads((Path(result_dir) / "run_config.json").read_text())
    require(config["evaluation_split"] == "validation" and config["primary_reference"] == "C",
            "Expected saved validation-C results")
    require(config["test_evaluated"] is False, "Results contain a scored test set")
    selected = []
    for group, label, display_metric in GROUPS:
        candidates = [row for row in all_rows if row["stage"] == "condition_baseline"
                      and row["reference"] == "C" and row["reference_role"] == "primary"
                      and row["aggregation"] == "question_macro" and row["group"] == group]
        require(len(candidates) == 1, f"Expected exactly one primary aggregate for {group}")
        row = candidates[0]
        require(int(row["participants"]) == 309, "Validation participant count changed")
        score = float(row["human_B_C_reference_paired"])
        require(math.isfinite(score) and 0 <= score <= 1, "Invalid retest aggregate")
        selected.append({"split": "validation", "group": group, "label": label,
                         "description": label,
                         "unit": "agreement proportion" if display_metric.startswith("Agreement") else "normalized difference",
                         "metric": display_metric, "score": score,
                         "question_ids": int(row["questions"]),
                         "paired_scalar_responses": int(row["human_B_C_pairs"]),
                         "participants": int(row["participants"]),
                         "aggregation": "question_macro"})
    for qid, (description, unit) in UNBOUNDED_TARGETS.items():
        candidates = [row for row in all_rows if row["stage"] == "condition_baseline"
                      and row["reference"] == "C" and row["reference_role"] == "primary"
                      and row["aggregation"] == "single_question" and row["group"] == "qid:" + qid]
        require(len(candidates) == 1, f"Expected one unbounded numeric aggregate for {qid}")
        row = candidates[0]
        score = float(row["human_B_C_reference_paired"])
        require(math.isfinite(score) and score >= 0, "Invalid unbounded retest aggregate")
        selected.append({"split": "validation", "group": "qid:" + qid, "label": qid,
                         "description": description, "unit": unit,
                         "metric": "Raw mean absolute difference ↓", "score": score,
                         "question_ids": 1, "paired_scalar_responses": int(row["human_B_C_pairs"]),
                         "participants": int(row["participants"]), "aggregation": "single_question"})
    return selected, {name: sha256(Path(result_dir) / name)
                      for name in ("comparison.csv", "run_config.json")}


def frozen_split(ids, manifest=None):
    ids = sorted(map(str, ids))
    require(len(ids) == len(set(ids)) == 2058 and "574" in ids, "Expected frozen 2,058 IDs")
    remaining = [pid for pid in ids if pid != "574"]
    random.Random(42).shuffle(remaining)
    split = {"train": {"574", *remaining[:1439]}, "validation": set(remaining[1439:1748]),
             "test": set(remaining[1748:])}
    if manifest:
        supplied = {name: set() for name in split}
        seen = set()
        with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                require(row["split"] in split and row["pid"] not in seen, "Invalid split row")
                supplied[row["split"]].add(row["pid"])
                seen.add(row["pid"])
        require(supplied == split, "Manifest differs from the frozen participant protocol")
    return split


def observed(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(observed(v) for v in value.values())
    if isinstance(value, list):
        return any(observed(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def answer_category(answer):
    if not observed(answer):
        return "[No recorded answer]"
    if isinstance(answer, dict):
        if "SelectedText" in answer:
            answer = answer["SelectedText"]
        elif "Text" in answer:
            answer = answer["Text"]
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True)
    return str(answer).strip()


def demographic_definitions(catalog, explicit_map=None):
    """Resolve only unambiguous A question definitions; keep the exact wording."""
    questions = [q for q in catalog if q.get("source") == "wave1_3_persona_json"
                 and q.get("QuestionType") != "DB"]
    mapping, unresolved = {}, {}
    selectors = explicit_map or {label: {"qid": qid, "block": "Demographics"}
                                 for label, qid in DEMOGRAPHICS.items()}
    for label, selector in selectors.items():
        candidates = [q for q in questions if str(q["QuestionID"]) == selector["qid"]
                      and q.get("BlockName", "") == selector["block"]]
        unique = {(str(q.get("BlockName", "")), str(q["QuestionID"])): q for q in candidates}
        if len(unique) == 1:
            key, question = next(iter(unique.items()))
            mapping[key] = {"label": label, "question": question["QuestionText"]}
        else:
            unresolved[label] = {"candidate_count": len(unique),
                                 "reason": "no match" if not unique else "ambiguous"}
    return mapping, unresolved


def pool_small_categories(counts, minimum=5):
    """Publish aggregate categories only; merge small cells until >= minimum."""
    result = Counter(counts)
    small = [key for key, count in result.items() if 0 < count < minimum]
    if not small:
        return result
    pooled = sum(result.pop(key) for key in small)
    if pooled < minimum and result:
        other = min(result, key=lambda key: (result[key], key))
        pooled += result.pop(other)
    require(pooled >= minimum, "Cannot publish this small demographic distribution")
    result["Other categories (pooled to avoid counts below 5)"] = pooled
    return result


def training_audit(dataset_dir, split_manifest=None, demographic_map=None):
    import pyarrow as pa
    import pyarrow.dataset as ds

    sys.path.insert(0, str(REPO / "experiments/01_pilot"))
    from download_data import FILES
    engine_path = REPO / "experiments/05_embedding_selected/qwen_embedding_selected/native_engine"
    sys.path.insert(0, str(engine_path))
    base = importlib.import_module("run_liquid_baseline")
    schema = importlib.import_module("full_b_schema")
    inputs = importlib.import_module("full_b_data")
    metrics = importlib.import_module("full_b_metrics")

    root = Path(dataset_dir)
    required = [name for name in FILES if name != "question_catalog_and_human_response_csv/wave4_response.csv"]
    hashes = {}
    for name in required:
        path = root / name
        require(path.is_file(), f"Missing frozen EDA input: {path}")
        hashes[name] = sha256(path)
        require(hashes[name] == FILES[name], f"Pinned dataset hash mismatch: {name}")
    catalog = json.loads((root / "question_catalog_and_human_response_csv/question_catalog.json").read_text())
    source = ds.dataset([str(root / name) for name in required if name.endswith(".parquet")], format="parquet")
    ids = source.to_table(columns=["pid"])["pid"].to_pylist()
    split = frozen_split(ids, split_manifest)
    train = split["train"]
    explicit = json.loads(Path(demographic_map).read_text()) if demographic_map else None
    definitions, unresolved = demographic_definitions(catalog, explicit)
    demographics = {key: Counter() for key in definitions}
    a_counts = defaultdict(Counter)
    a_types, b_types, a_record_histogram = Counter(), Counter(), Counter()
    b_specs = {}
    a_question_text = {}
    for q in catalog:
        if q.get("source") == "wave1_3_persona_json" and q.get("QuestionType") != "DB":
            a_question_text[(str(q.get("BlockName", "")), str(q["QuestionID"]))] = q["QuestionText"]
    scanner = source.scanner(columns=["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A"],
                             filter=ds.field("pid").cast(pa.string()).isin(sorted(train)), batch_size=8)
    seen = set()
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            pid = str(row["pid"])
            require(pid in train and pid not in seen, "Unexpected training participant")
            seen.add(pid)
            present = set()
            records = 0
            for block, questions in base.blocks(row["wave1_3_persona_json"]):
                for q in questions:
                    if q["QuestionType"] == "DB":
                        continue
                    key = (str(block), str(q["QuestionID"]))
                    require(key not in present, f"Duplicate A definition for {key}")
                    present.add(key)
                    records += 1
                    answer = q.get("Answers")
                    count = a_counts[key]
                    count["present"] += 1
                    count["nonempty_answer"] += observed(answer)
                    a_types[q["QuestionType"]] += 1
                    a_question_text.setdefault(key, q["QuestionText"])
                    if q["QuestionType"] == "Matrix" and isinstance(answer, dict):
                        positions = answer.get("SelectedByPosition")
                        if isinstance(positions, list) and isinstance(q.get("Rows"), list):
                            count["matrix_declared_rows"] += len(q["Rows"])
                            count["matrix_observed_selections"] += len(positions)
                            count["short_matrix_sequences"] += len(positions) < len(q["Rows"])
                    if key in definitions:
                        demographics[key][answer_category(answer)] += 1
            for key in definitions:
                if key not in present:
                    demographics[key]["[Question record absent]"] += 1
            a_record_histogram[records] += 1
            # Strip B answers before deriving administered scalar schemas.
            units = base.survey_units(row["wave4_Q_wave1_3_A"], include_answers=False)
            items = schema.expand_units(units)
            b_specs[pid] = {item["item_id"]: item for item in items}
            b_types.update(item["family"] for item in items)
    require(seen == train, "Training dataset coverage differs from split")

    all_items = schema.catalog_items(catalog)
    labels = inputs.read_selected_labels(root / "question_catalog_and_human_response_csv/wave1_3_response.csv",
                                        train, [item["item_id"] for item in all_items])
    b_coverage = defaultdict(Counter)
    absent_nonempty = 0
    for pid in train:
        for item in all_items:
            key = (pid, item["item_id"])
            value = labels[key]
            stats = b_coverage[item["family"]]
            stats["global_cells"] += 1
            if item["item_id"] not in b_specs[pid]:
                stats["not_in_person_B_metadata"] += 1
                absent_nonempty += value is not None
                continue
            stats["administered_cells"] += 1
            if value is None:
                stats["missing_among_administered"] += 1
            elif metrics._valid(value, b_specs[pid][item["item_id"]]):
                stats["valid_among_administered"] += 1
            else:
                stats["invalid_among_administered"] += 1
    require(absent_nonempty == 0, "Nonempty B cell lacks participant target metadata; inspect before reporting")

    a_rows = []
    for key in sorted(a_question_text):
        counts = a_counts[key]
        a_rows.append({"block": key[0], "qid": key[1], "question": a_question_text[key],
                       "participants": len(train), "present": counts["present"],
                       "record_absent": len(train) - counts["present"],
                       "answer_empty_among_present": counts["present"] - counts["nonempty_answer"],
                       **{name: counts[name] for name in ("matrix_declared_rows", "matrix_observed_selections",
                                                         "short_matrix_sequences")}})
    demo_rows = []
    for key, counts in demographics.items():
        for category, count in pool_small_categories(counts).most_common():
            demo_rows.append({"split": "train", "variable": definitions[key]["label"], "block": key[0],
                              "qid": key[1], "question": definitions[key]["question"], "category": category,
                              "count": count, "denominator": len(train), "percent": 100 * count / len(train)})
    b_fields = ("global_cells", "not_in_person_B_metadata", "administered_cells", "valid_among_administered",
                "missing_among_administered", "invalid_among_administered")
    b_rows = [{"family": family, **{key: values[key] for key in b_fields}}
              for family, values in sorted(b_coverage.items())]
    metadata = {"revision": REVISION, "data_hashes": hashes, "raw_analysis_split": "train",
                "participants": len(train), "split_sizes": {k: len(v) for k, v in split.items()},
                "split_manifest_sha256": sha256(split_manifest) if split_manifest else None,
                "split_recipe": "string-sorted PIDs; seed 42; PID 574 reserved train; 1440/309/309",
                "parquet_columns_selected": ["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A"],
                "C_column_selected": False, "wave4_response_csv_opened": False,
                "test_answer_cells_analyzed": 0, "a_stored_record_histogram": dict(a_record_histogram),
                "a_record_types": dict(a_types), "unresolved_demographic_variables": unresolved,
                "demographic_selection": "explicit mapping" if explicit else "exact pinned-catalog Demographics block/QIDs",
                "demographic_small_cell_policy": "pool categories under 5; add next smallest category if pooled total is under 5"}
    return a_rows, demo_rows, b_rows, metadata


def markdown(retest, training=None):
    lines = ["# Extended EDA findings", "",
             "This audit separates **training-only profile and response analysis** from **previously scored validation human retest results**. The reserved test set is not evaluated.", "",
             "## Human B/C consistency on validation", "",
             "B and C are the same person's earlier and later answers to repeated questions. The table uses original aggregate files bundled under `docs/eda/source/` from the selected-answer retrieval run; it does not read new C answers. Scores first average within each question ID, then across IDs.", "",
             "| Question group | Question IDs | Paired scalar responses | Human B/C result | Measure |",
             "|---|---:|---:|---:|---|"]
    for row in retest:
        if row["aggregation"] != "question_macro":
            continue
        score = f"{100 * row['score']:.2f}%" if row["metric"].startswith("Agreement") else f"{row['score']:.4f}"
        lines.append(f"| {row['label']} | {row['question_ids']} | {row['paired_scalar_responses']:,} | {score} | {row['metric']} |")
    lines += ["", "All groups cover the same **309 validation participants**. Pair counts count scalar responses, not independent people. The nine pilot question IDs encode alternative branches: each participant contributes five binary scenarios. Ordered differences are divided by `K − 1`; bounded numeric differences use the allowed range. Lower difference means better retest consistency.", "",
              "The four **unbounded numeric targets** are reported separately in their original units; they have no shared normalization range and must not be averaged into the table above.", "",
              "| Question ID | Target and preceding anchor | Unit | Paired responses / participants | Mean absolute B/C difference ↓ |",
              "|---|---|---|---:|---:|"]
    for row in retest:
        if row["aggregation"] == "single_question":
            lines.append(f"| {row['label']} | {row['description']} | {row['unit']} | {row['paired_scalar_responses']} | {row['score']:.4f} |")
    lines += ["", "The anchors are numbers shown in the preceding survey question, not correct answers or bounds. Both country-count targets use countries; both tree-height targets use feet. The experiment's numeric schema treats these as nonnegative values without a fixed upper limit."]
    lines += ["", "These are **retest agreement/difference measures**, not psychometric reliability coefficients and not a strict prediction ceiling. Real preferences can change between waves; a model may also benefit from aggregate regularities. The validation split has been reused for development, so these figures are descriptive, not independent test evidence.", ""]
    if training is None:
        lines += ["## Training demographics and missingness", "",
                  "**Not recomputed in this report:** the source dataset is required to produce demographic counts and distinguish unassigned branches from missing administered responses. The original notebook establishes participant counts, question types, A/B definition differences, and a training-person leakage example; those outputs alone do not establish demographic representativeness or response completeness.", "",
                  "Run the command below with `--dataset-dir` to add verified training-only demographics and missingness. No demographic frequencies are inferred from the model-result archive.", ""]
    else:
        a_rows, demos, b_rows, metadata = training
        total_present = sum(row["present"] for row in a_rows)
        total_absent = sum(row["record_absent"] for row in a_rows)
        total_empty = sum(row["answer_empty_among_present"] for row in a_rows)
        short_matrices = sum(row["short_matrix_sequences"] for row in a_rows)
        short_entries = sum(row["matrix_declared_rows"] - row["matrix_observed_selections"] for row in a_rows)
        lines += ["## Training demographics", "",
                  f"Computed from A for **{metadata['participants']:,} training participants** at pinned dataset revision `{REVISION}`. Percentages use all training participants as the denominator. Categories smaller than five are pooled, with another category if necessary. Question definitions and the released categories are in `eda/training_demographics.csv`.", "",
                  "| Variable | Recorded category | Count | % of training participants |", "|---|---|---:|---:|"]
        for row in demos:
            category = row["category"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['variable']} | {category} | {row['count']:,} | {row['percent']:.2f}% |")
        if not demos:
            lines.append("| — | No unambiguous demographic questions resolved; see provenance | — | — |")
        if metadata["unresolved_demographic_variables"]:
            lines += ["", "Unresolved variables: " + ", ".join(metadata["unresolved_demographic_variables"]) + ". These are not treated as missing answers; the exact selector did not uniquely identify a question."]
        lines += ["", "## Training missingness and coverage", "",
                  f"A contains **{total_present:,} stored question records**, with **{total_empty:,} empty answer objects among present records** and **{total_absent:,} absent question records** relative to the catalog's question definitions. These are question-record counts, not scalar-answer counts. **{short_matrices:,} A matrix records have shortened answer sequences**, totaling **{short_entries:,} fewer selections than declared rows**; these are retained without assigning entries to guessed rows.", "",
                  "| Stored A question records per person | Training participants |", "|---|---:|"]
        for records, count in sorted(metadata["a_stored_record_histogram"].items(), key=lambda pair: int(pair[0]), reverse=True):
            lines.append(f"| {records} | {count:,} |")
        lines += ["",
                  "| B family | Global participant × item cells | Not represented in person's B metadata | Administered cells | Valid answers | Missing answers | Invalid answers |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in b_rows:
            lines.append("| " + " | ".join([row["family"], *[f"{row[key]:,}" for key in (
                "global_cells", "not_in_person_B_metadata", "administered_cells", "valid_among_administered",
                "missing_among_administered", "invalid_among_administered")]]) + " |")
        totals = {key: sum(row[key] for row in b_rows) for key in b_rows[0] if key != "family"}
        lines.append("| **Total** | " + " | ".join(f"{totals[key]:,}" for key in (
            "global_cells", "not_in_person_B_metadata", "administered_cells", "valid_among_administered",
            "missing_among_administered", "invalid_among_administered")) + " |")
        lines += ["", "The 126 global B scalar columns combine alternative survey branches. A blank in a target absent from the person's B metadata is separated from a blank in an administered target. Metadata absence is not proof of refusal or dropout; this audit does not infer its cause per cell. A-record absence likewise must not be relabeled as nonresponse without survey routing evidence.", ""]
    lines += ["## Representativeness and limits", "",
              "- A participant split prevents identity overlap, but it does not make this survey sample representative of a country, age group, or deployment population.",
              "- Training demographic frequencies describe this split only. No population benchmark, sampling weights, subgroup performance estimates, or causal claims are supplied by this audit.",
              "- Survey routing, incomplete matrix sequences, and repeated measures create unequal observation counts. The 126 global target columns must not be used as a per-person nonresponse denominator.",
              "- Repeated answers are correlated within a person and question. Pair counts are not independent sample sizes; uncertainty estimates would need participant-aware resampling and a declared question aggregation.",
              "- Test C remains reserved. This extension selects no C parquet column and never opens the wave-4 response CSV. Dataset hashing reads raw bytes for integrity only.", "",
              "## Reproduce", "",
              "From a fresh checkout, the original aggregate `comparison.csv` and `run_config.json` are already included under `docs/eda/source/`. They contain aggregate metrics, historical paths/settings, and source hashes, with no participant records. Their original bytes and SHA256 hashes are preserved in the audit provenance. `--results` defaults to this bundled folder; it can select another compatible aggregate folder explicitly.", "",
              "```bash", "# Rebuild archived validation retest tables using only bundled inputs (standard library)",
              "python scripts/extend_eda.py --output-dir ../twin_work/eda_archive_review", "",
              "# Add training A/B demographic and missingness analysis (CPU; pandas/pyarrow)",
              "python scripts/extend_eda.py \\",
              "  --dataset-dir /path/to/Twin-2K-500 \\",
              "  --output-dir ../twin_work/eda_full_review", "```", "",
              "The manifest is optional: without it the script reconstructs the established seed-42 split with participant 574 reserved for training. If supplied, membership must match exactly. The default demographics use exact question IDs in the pinned catalog's Demographics block. `--demographic-map` accepts a JSON mapping from display names to exact `block` and `qid` selectors; ambiguous matches are omitted, never guessed.", "",
              "The default output directory is `../twin_work/eda_review` when viewed from the repository root. Each run writes `EDA_FINDINGS.md`, aggregate CSV tables under `eda/`, and `eda/provenance.json` beneath its output directory. Ordinary review commands leave the fuller published report untouched. Updating the published report requires an explicit full-data run with `--output-dir docs`. Raw participant records are not written.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="Aggregate source folder (default: bundled docs/eda/source)")
    parser.add_argument("--dataset-dir", type=Path, help="Optional pinned raw dataset; enables training-only A/B audit")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--demographic-map", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Review output folder (default: ../twin_work/eda_review outside the repository)")
    args = parser.parse_args()
    require(args.dataset_dir or not (args.split_manifest or args.demographic_map),
            "Split/demographic mapping requires --dataset-dir")
    require(args.dataset_dir or args.output_dir.resolve() != (REPO / "docs").resolve(),
            "An archive-only run cannot replace the fuller published report; use an external --output-dir")
    retest, hashes = archived_retest(args.results)
    training = training_audit(args.dataset_dir, args.split_manifest, args.demographic_map) if args.dataset_dir else None
    target = args.output_dir / "eda"
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "validation_human_retest.csv", retest, tuple(retest[0]))
    provenance = {"protocol": "training-a-b-eda-and-archived-validation-retest-v1", "dataset_revision": REVISION,
                  "script_sha256": sha256(__file__), "validation_archive_hashes": hashes,
                  "new_raw_C_answers_read": False, "test_scored": False, "raw_training_audit_run": training is not None}
    if training:
        a_rows, demos, b_rows, metadata = training
        for name, rows, fallback_fields in (
            ("training_a_missingness.csv", a_rows, ()),
            ("training_demographics.csv", demos, ("split", "variable", "block", "qid", "question", "category", "count", "denominator", "percent")),
            ("training_b_coverage.csv", b_rows, ())):
            write_csv(target / name, rows, tuple(rows[0]) if rows else fallback_fields)
        provenance["training_audit"] = metadata
    write_json(target / "provenance.json", provenance)
    (args.output_dir / "EDA_FINDINGS.md").write_text(markdown(retest, training), encoding="utf-8")
    print(f"Wrote aggregate EDA findings: {args.output_dir / 'EDA_FINDINGS.md'}")
    print(f"Raw training audit: {training is not None}; new C access: false; test scored: false")


if __name__ == "__main__":
    main()
