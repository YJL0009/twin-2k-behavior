#!/usr/bin/env python3
"""Run fixed training-only product/price and A-nearest-neighbor baselines.

This CPU runner needs numpy, pandas and pyarrow, and the existing native full-B
engine sources. It never loads a language model or reads reserved test answers.
All prediction files are saved and hashed before validation B/C scoring begins.
"""
from __future__ import annotations

import argparse
import fcntl
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

if __package__:
    from .data_io import (canonical, digest, fit_reference_baselines, import_cpu_engines,
                          load_people, make_manifest, make_records, require,
                          train_item_specs, write_json, write_jsonl)
    from .baselines import build_baselines
else:
    from data_io import (canonical, digest, fit_reference_baselines, import_cpu_engines,
                         load_people, make_manifest, make_records, require,
                         train_item_specs, write_json, write_jsonl)
    from baselines import build_baselines

PROTOCOL = "fixed-epoch1-training-only-cpu-comparisons-v1"


def verify_retest_metadata(dataset_dir, pids, engine, out):
    """Run only after all predictions have been committed to disk."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    chunks = sorted((Path(dataset_dir) / "wave_split/chunks").glob("*.parquet"))
    source = ds.dataset([str(path) for path in chunks], format="parquet")
    scanner = source.scanner(columns=["pid", "wave4_Q_wave1_3_A", "wave4_Q_wave4_A"],
        filter=ds.field("pid").cast(pa.string()).isin(sorted(pids)), batch_size=1)
    seen, mismatches = set(), []
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            pid = str(row["pid"])
            require(pid in pids and pid not in seen, "Retest metadata participant mismatch")
            seen.add(pid)
            earlier = dict(engine.base.survey_units(row.pop("wave4_Q_wave1_3_A"), False))
            later = dict(engine.base.survey_units(row.pop("wave4_Q_wave4_A"), False))
            for qid in sorted(set(earlier) | set(later)):
                if earlier.get(qid) != later.get(qid):
                    mismatches.append({"pid": pid, "qid": qid})
    require(seen == set(pids), "Retest metadata coverage incomplete")
    write_json(Path(out) / "retest_metadata_check.json", {
        "participants": len(seen), "mismatches": mismatches,
        "policy": "B and C target text, options, rows/statements and instructions must match."})
    require(not mismatches, "B/C target metadata differ; validation scoring stopped")


def freeze_configuration(args, engine):
    dataset = args.dataset_dir.resolve()
    chunks = sorted((dataset / "wave_split/chunks").glob("*.parquet"))
    response = dataset / "question_catalog_and_human_response_csv"
    require(len(chunks) == 7, "Expected the seven pinned wave-split parquet chunks")
    inputs = [args.split.resolve(), *chunks, response / "wave1_3_response.csv",
              response / "wave4_response.csv"]
    require(all(path.is_file() for path in inputs), "Dataset, split or response file is missing")
    source_names = [Path(__file__).resolve(), Path(__file__).with_name("data_io.py").resolve(),
                    Path(__file__).with_name("baselines.py").resolve()]
    source_names += [Path(getattr(engine, key).__file__).resolve() for key in
                     ("base", "inputs", "schema", "metrics", "diag")]
    config = {
        "protocol": PROTOCOL, "evaluation_split": "validation", "primary_reference": "C",
        "secondary_reference": "B", "dataset_dir": str(dataset), "split": str(args.split.resolve()),
        "scripts_dir": str(engine.directory), "knn_k": 5, "demonstration_neighbors": 3,
        "selection": "Fixed methods and hyperparameters; no validation fitting or checkpoint selection",
        "train_label_source": "B cells for training participants only",
        "prediction_inputs": "A answers and target B metadata, with all embedded B answers stripped",
        "price_evaluation_scope": "40 pricing QIDs only",
        "nearest_neighbor_evaluation_scope": "all available scalar B items",
        "test_inputs_or_answers_loaded": False,
        "inputs_sha256": {str(path): engine.base.digest_file(path) for path in inputs},
        "source_sha256": {str(path): engine.base.digest_file(path) for path in source_names},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")},
    }
    path = args.out / "comparison_config.json"
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == config,
                "Existing output configuration differs; choose a fresh --out directory")
    else:
        write_json(path, config)
    return config


def assert_sources_unchanged(config, engine):
    for section in ("inputs_sha256", "source_sha256"):
        for name, expected in config[section].items():
            require(Path(name).is_file() and engine.base.digest_file(name) == expected,
                    f"Frozen input/source changed during this run: {name}")


def summarize_report(out, summaries, method_report):
    import pandas as pd
    lines = ["Training-only CPU baseline comparison complete", "",
             "309 validation participants; C is primary and B is secondary.",
             "Both methods and k=5 were fixed before reading validation labels.",
             "Product/price baseline uses training B and the actual target product identity and price only.",
             "Pricing QIDs are shuffled positions: the original per-QID mode ignores both product identity and price.",
             "Nearest neighbors use A similarity, exact target matching and training B answers.",
             "The fixed epoch-1 adapter has not been loaded or changed by this CPU run.",
             "No reserved test inputs/answers were materialized or evaluated.", ""]
    for method, summary in summaries.items():
        family = pd.read_csv(out / method / "scores/metrics_by_family.csv")
        questions = pd.read_csv(out / method / "scores/metrics_by_question.csv")
        lines += [f"Method: {method}; cases: {summary['prompts']}; QIDs: {summary['question_ids']}"]
        columns = ["family", "prediction_valid_rate_on_valid_labels", "question_macro_categorical_accuracy",
                   "question_macro_mode_categorical_accuracy", "question_macro_ordinal_nmae_invalid_penalty",
                   "question_macro_median_ordinal_nmae", "question_macro_bounded_nmae_invalid_penalty",
                   "question_macro_median_bounded_nmae"]
        lines += [family.loc[family.reference == "C", columns].round(6).to_string(index=False)]
        for label, mask in (("pricing_40", questions.qid.isin([f"QID9_{i}" for i in range(1, 41)])),
                            ("original_9_binary", questions.qid.isin([
                                "QID163", "QID165", "QID167", "QID169", "QID183", "QID184",
                                "QID192", "QID193", "QID196"]))):
            selected = questions[(questions.reference == "C") & mask]
            if len(selected):
                actual, reference = selected.categorical_accuracy.mean(), selected.mode_categorical_accuracy.mean()
                lines += [f"{label}: {actual:.6%}; training mode: {reference:.6%}; gain: {(actual-reference)*100:+.4f} pp"]
        numeric = questions[(questions.reference == "C") & (questions.family == "unbounded_numeric")]
        if len(numeric):
            lines += ["Unbounded numeric MAE, original units:", numeric[["qid", "numeric_valid_rate",
                      "numeric_mae_valid_only", "median_numeric_mae_on_model_valid"]].round(6).to_string(index=False)]
        lines += [""]
    lines += ["These are descriptive development-set comparisons, after prior validation experiments.",
              "Question-family metrics remain separate; there is no mixed-unit full-B score.",
              "A nearest-neighbor result alone does not establish that the adapter uses A.",
              "Run the GPU no-A, shuffled-A and retrieved-demonstration stages for that comparison.",
              "Retrieval/fit coverage is in baseline_fit_report.json; complete native metrics are in each scores/."]
    path = out / "report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"), flush=True)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/root/autodl-tmp/twin_qwen")
    parser.add_argument("--dataset-dir", type=Path, default=root / "data/Twin-2K-500")
    parser.add_argument("--split", type=Path, default=root / "data/participant_split.csv")
    parser.add_argument("--scripts-dir", type=Path, default=root / "qwen_full_b_lora_scripts")
    parser.add_argument("--out", type=Path, default=root / "outputs/epoch1_cpu_comparisons")
    return parser


def run(args):
    engine = import_cpu_engines(args.scripts_dir)
    config = freeze_configuration(args, engine)
    split = engine.inputs.read_split(args.split)
    print("Loading train/validation A and answer-free target metadata; test rows are filtered out.", flush=True)
    people = load_people(args.dataset_dir, split, engine)
    all_specs = train_item_specs(people, split["train"], engine)
    train_labels = engine.inputs.read_selected_labels(
        args.dataset_dir / "question_catalog_and_human_response_csv/wave1_3_response.csv",
        split["train"], all_specs)
    reference = fit_reference_baselines(people, split["train"], train_labels, engine)
    write_json(args.out / "training_baselines.json", reference)
    print("Fitting fixed product/price and five-neighbor baselines using training B only.", flush=True)
    fitted = build_baselines(people, train_labels, split["train"], split["validation"], k=5, demonstrations=3)
    require(set(fitted) >= {"knn", "price", "report"}, "Baseline implementation returned an invalid result")
    write_json(args.out / "baseline_fit_report.json", fitted["report"])
    stages = {}
    for method, key in (("nearest_neighbor", "knn"), ("product_price", "price")):
        manifest = make_manifest(people, split["validation"], engine, method)
        if method == "product_price":
            manifest = [spec for spec in manifest if spec.get("pricing")]
            require(len(manifest) == 12360 and len({s["qid"] for s in manifest}) == 40,
                    "Expected exactly 40 pricing items per validation participant")
        else:
            require(len(manifest) == 29690 and len({s["qid"] for s in manifest}) == 84,
                    "Expected the frozen 29,690-case full-B validation scope")
        records = make_records(fitted[key], manifest, engine, method)
        folder = args.out / method
        write_jsonl(folder / "validation_manifest.jsonl", manifest)
        write_jsonl(folder / "raw_predictions.jsonl", records)
        stages[method] = (records, manifest)
    assert_sources_unchanged(config, engine)
    # All methods' predictions are complete and durable before ANY validation labels are parsed.
    prediction_files = [args.out / method / filename for method in stages
                        for filename in ("validation_manifest.jsonl", "raw_predictions.jsonl")]
    frozen = {"protocol": PROTOCOL, "config_sha256": digest(config), "predictions_complete": True,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "validation_labels_read": False,
              "artifacts_sha256": {str(path.relative_to(args.out)): engine.base.digest_file(path)
                                   for path in prediction_files}}
    write_json(args.out / "predictions_frozen.json", frozen)
    print("All CPU predictions frozen. Now reading only validation B/C cells for scoring.", flush=True)
    verify_retest_metadata(args.dataset_dir, split["validation"], engine, args.out)
    response_dir = args.dataset_dir / "question_catalog_and_human_response_csv"
    labels_b = engine.inputs.read_selected_labels(response_dir / "wave1_3_response.csv", split["validation"], all_specs)
    labels_c = engine.inputs.read_selected_labels(response_dir / "wave4_response.csv", split["validation"], all_specs)
    summaries = {}
    for method, (records, manifest) in stages.items():
        folder = args.out / method
        summary = engine.metrics.score_full(records, manifest, labels_b, labels_c, reference,
            folder / "scores", trained_qids={spec["qid"] for spec in all_specs.values()},
            trained_scope="train_B_only_full_B_baseline")
        summary.update(comparison_protocol=PROTOCOL, method=method, evaluation_split="validation",
                       test_evaluated=False, config_sha256=digest(config),
                       predictions_sha256=engine.base.digest_file(folder / "raw_predictions.jsonl"))
        write_json(folder / "summary.json", summary)
        summaries[method] = summary
    assert_sources_unchanged(config, engine)
    for filename, expected in frozen["artifacts_sha256"].items():
        require(engine.base.digest_file(args.out / filename) == expected, "Predictions changed during scoring")
    summarize_report(args.out, summaries, fitted["report"])
    write_json(args.out / "evaluation_complete.json", {"protocol": PROTOCOL, "config_sha256": digest(config),
               "methods": list(stages), "test_evaluated": False,
               "predictions_frozen_sha256": engine.base.digest_file(args.out / "predictions_frozen.json"),
               "completed_utc": datetime.now(timezone.utc).isoformat()})


def main():
    args = parser().parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another CPU comparison process is using this output directory")
        run(args)


if __name__ == "__main__":
    main()
