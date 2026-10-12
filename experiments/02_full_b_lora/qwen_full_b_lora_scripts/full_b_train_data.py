#!/usr/bin/env python3
"""Prepare all valid training-B scalar responses and label-free validation inputs.

Uses only the established 1,440 training and 309 validation participants. Test
participant metadata and test response cells are never materialized. The parent
trainer owns optimization and checkpoint selection; this module never opens C.
"""
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

import run_liquid_baseline as base
import diagnose_liquid_outputs as diag
import full_b_schema as schema
import full_b_data as inputs

PROTOCOL = "twin-full-b-train-B-validation-inputs-v1"
require = base.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_state(value, item):
    """Return missing/invalid/valid under the same scalar domain as inference."""
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "invalid"
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return "invalid"
    if not math.isfinite(number):
        return "invalid"
    if item["family"] in ("mc", "ordinal", "matrix_nominal", "matrix_ordinal"):
        return "valid" if number.is_integer() and 1 <= number <= len(item["options"]) else "invalid"
    require(item["family"] in ("bounded_numeric", "unbounded_numeric"), "Unknown scalar family")
    if item.get("lower") is not None and number < item["lower"]:
        return "invalid"
    if item.get("upper") is not None and number > item["upper"]:
        return "invalid"
    return "valid"


def format_target(value, item):
    """Shortest ordinary/scientific numeric response with exact float round-trip."""
    require(label_state(value, item) == "valid", f"Cannot format invalid B target: {item['item_id']}")
    number = float(value)
    if item["family"] in ("mc", "ordinal", "matrix_nominal", "matrix_ordinal"):
        text = str(int(number))
    elif number.is_integer() and abs(number) < 1e16:
        text = str(int(number))
    else:
        text = repr(number)
    parsed = schema.parse_prediction(text, item)
    require(parsed is not None and float(parsed) == number,
            f"Formatted B target did not round-trip: {item['item_id']}")
    return text


def response_cap(item):
    return 8 if item["qid"] in schema.TRAINED_BINARY_QIDS else 16


def validate_prefix(encoded, record, max_input_tokens=32760):
    require(set(encoded) == {"input_ids", "attention_mask"}, "Inference prefix contains supervised labels")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
            "Invalid prompt token prefix")
    require(len(ids) == record["input_tokens"] <= max_input_tokens and mask == [1] * len(ids),
            "Prompt prefix length or attention mask changed")
    require(len(ids) + response_cap(record) <= 32768,
            f"{record['pid']}/{record['item_id']}: prompt plus frozen output cap exceeds Qwen context")


def encode_supervised(encoded, record, label_b, tokenizer, suffix_cache=None):
    """Append only the answer and exactly one EOS; all prompt positions masked."""
    validate_prefix(encoded, record)
    text = format_target(label_b, record)
    require(tokenizer.eos_token_id is not None, "Tokenizer EOS is required for supervised targets")
    cache = suffix_cache if suffix_cache is not None else {}
    if text not in cache:
        answer_ids = tokenizer.encode(text, add_special_tokens=False)
        require(isinstance(answer_ids, list) and answer_ids and
                all(type(x) is int and x >= 0 for x in answer_ids), "Answer tokenization returned invalid IDs")
        require(tokenizer.eos_token_id not in answer_ids, "Answer itself contains an EOS token")
        require(tokenizer.decode(answer_ids, skip_special_tokens=False) == text,
                "Answer tokenization changed the literal numeric target")
        cache[text] = answer_ids + [tokenizer.eos_token_id]
    suffix = list(cache[text])
    require(suffix[-1] == tokenizer.eos_token_id and suffix.count(tokenizer.eos_token_id) == 1,
            "Answer suffix must contain exactly one final EOS")
    require(len(suffix) <= response_cap(record),
            f"{record['pid']}/{record['item_id']}: numeric target needs {len(suffix)} tokens, "
            f"exceeding its frozen inference cap {response_cap(record)}")
    prefix = list(encoded["input_ids"])
    row = {"input_ids": prefix + suffix, "attention_mask": [1] * (len(prefix) + len(suffix)),
           "labels": [-100] * len(prefix) + suffix}
    require(len(row["input_ids"]) <= 32768, "Supervised example exceeds Qwen context")
    manifest = dict(record)
    manifest.update(label_b=float(label_b) if not float(label_b).is_integer() else int(float(label_b)),
                    answer_text=text, answer_token_ids=suffix[:-1], supervised_tokens=len(suffix),
                    total_tokens=len(row["input_ids"]), supervision_scope="all-valid-B-items")
    return manifest, row


def _source_jobs(source, pids):
    import pyarrow as pa
    import pyarrow.dataset as ds
    fields = ["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A"]
    scanner = source.scanner(columns=fields, batch_size=1,
                            filter=ds.field("pid").cast(pa.string()).isin(sorted(pids)))
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            pid = str(row["pid"])
            require(pid in pids, "Participant outside requested development split reached preparation")
            # Drop embedded B answer payloads before passing data to workers.
            b_metadata = base.survey_units(row["wave4_Q_wave1_3_A"], include_answers=False)
            yield pid, row["wave1_3_persona_json"], b_metadata


def load_prepared(out):
    """Load completed artifacts after verifying their configuration and hashes."""
    from datasets import Dataset
    out = Path(out).resolve()
    config = read_json(out / "preparation_config.json")
    summary = read_json(out / "preparation_summary.json")
    require(config["protocol"] == PROTOCOL and summary["protocol"] == PROTOCOL
            and base.digest_text(base.dumps(config)) == summary["config_hash"],
            "Full-B cached configuration changed")
    paths = {"train.arrow": out / "prepared/train.arrow",
             "validation.arrow": out / "prepared/validation.arrow",
             "training_manifest.jsonl": out / "training_manifest.jsonl",
             "validation_manifest.jsonl": out / "prepared/validation_manifest.jsonl",
             "training_baselines.json": out / "training_baselines.json"}
    require(set(summary["artifacts_sha256"]) == set(paths), "Full-B cache artifact list changed")
    for name, path in paths.items():
        require(path.is_file() and base.digest_file(path) == summary["artifacts_sha256"][name],
                f"Full-B prepared artifact changed: {name}")
    train = Dataset.from_file(str(paths["train.arrow"]))
    validation = Dataset.from_file(str(paths["validation.arrow"]))
    manifest = base.read_jsonl(paths["validation_manifest.jsonl"])
    require(len(train) == summary["train_examples"] and len(validation) == len(manifest)
            == summary["validation_examples"], "Full-B cached example counts changed")
    require(set(train.column_names) == {"input_ids", "attention_mask", "labels"}
            and set(validation.column_names) == {"input_ids", "attention_mask"}, "Arrow supervision schema changed")
    return train, validation, manifest, summary


def prepare_training(root, out, qwen_model, selection_model, workers=4,
                     max_input_tokens=32760, selection_budget=8192):
    """Return (training Dataset, validation Dataset, validation manifest, summary).

    ``train.arrow`` includes answer-only labels. ``validation.arrow`` contains
    only input IDs and masks. Validation includes all answerable B metadata
    records independent of B/C response presence. No C CSV is opened.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    from datasets import Dataset, Features, Sequence, Value
    from transformers import AutoTokenizer
    import full_b_metrics as metrics
    root, out = Path(root).resolve(), Path(out).resolve()
    qwen_model, selection_model = Path(qwen_model).resolve(), Path(selection_model).resolve()
    require(workers >= 1 and max_input_tokens == 32760 and selection_budget == 8192,
            "Keep the frozen context policy and use at least one preprocessing worker")
    split = inputs.read_split(root)
    train_pids, val_pids, test_pids = split["train"], split["validation"], split["test"]
    require(not (train_pids & val_pids or train_pids & test_pids or val_pids & test_pids), "Overlapping splits")
    all_pids = train_pids | val_pids | test_pids
    split_path = root / "data/participant_split.csv"
    response_path = root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave1_3_response.csv"
    catalog_path = Path(__file__).with_name("full_b_question_catalog.json")
    catalog = read_json(catalog_path)
    all_specs = {item["item_id"]: item for item in schema.catalog_items(catalog)}
    require(len(all_specs) == 126, "Expected all 126 possible scalar B items")
    chunks = sorted((root / "data/Twin-2K-500/wave_split/chunks").glob("*.parquet"))
    require(len(chunks) == 7 and response_path.is_file(), "Missing seven source chunks or earlier-wave B CSV")
    source_paths = [split_path, response_path, catalog_path, *chunks]
    scripts = [Path(__file__), Path(inputs.__file__), Path(schema.__file__), Path(base.__file__),
               Path(diag.__file__), Path(metrics.__file__)]
    config = {"protocol": PROTOCOL, "root": str(root), "qwen_model": str(qwen_model),
              "selection_model": str(selection_model), "selection_budget": selection_budget,
              "max_input_tokens": max_input_tokens, "train_pids": sorted(train_pids),
              "validation_pids": sorted(val_pids), "test_participant_metadata_loaded": False,
              "training_target_scope": "all scalar B items with valid training responses",
              "training_label_source": "training-participant B only; no C",
              "validation_label_source": "none during preparation",
              "target_serialization": "strict categorical integer or shortest round-trip finite number; answer+EOS",
              "context_policy": "whole A units with original Liquid8192 selection; exact binary9 prompt policy",
              "inputs_sha256": {str(path): base.digest_file(path) for path in source_paths},
              "scripts_sha256": {path.name: base.digest_file(path) for path in scripts},
              "qwen_tokenizer_sha256": inputs.tokenizer_files_sha256(qwen_model),
              "selection_tokenizer_sha256": inputs.tokenizer_files_sha256(selection_model),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "tokenizers", "datasets", "pyarrow")}}
    config_hash = base.digest_text(base.dumps(config))
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "preparation_config.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Full-B preparation configuration changed; choose a new output")
    else:
        base.write_json(config_path, config)
    prepared = out / "prepared"
    prepared.mkdir(exist_ok=True)
    paths = {"train.arrow": prepared / "train.arrow", "validation.arrow": prepared / "validation.arrow",
             "training_manifest.jsonl": out / "training_manifest.jsonl",
             "validation_manifest.jsonl": prepared / "validation_manifest.jsonl",
             "training_baselines.json": out / "training_baselines.json"}
    summary_path = out / "preparation_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        require(summary["config_hash"] == config_hash, "Full-B prepared configuration hash changed")
        for name, path in paths.items():
            require(path.is_file() and base.digest_file(path) == summary["artifacts_sha256"][name],
                    f"Full-B prepared artifact changed: {name}")
        train = Dataset.from_file(str(paths["train.arrow"]))
        validation = Dataset.from_file(str(paths["validation.arrow"]))
        manifest = base.read_jsonl(paths["validation_manifest.jsonl"])
        require(len(train) == summary["train_examples"] and len(validation) == len(manifest)
                == summary["validation_examples"], "Full-B prepared example counts changed")
        require(set(train.column_names) == {"input_ids", "attention_mask", "labels"}
                and set(validation.column_names) == {"input_ids", "attention_mask"}, "Arrow supervision schema changed")
        return train, validation, manifest, summary
    source = ds.dataset([str(path) for path in chunks], format="parquet")
    ids = [str(pid) for pid in source.to_table(columns=["pid"])["pid"].to_pylist()]
    require(len(ids) == len(set(ids)) == 2058 and set(ids) == all_pids, "Dataset and frozen split IDs differ")
    labels = inputs.read_selected_labels(response_path, train_pids, all_specs)
    baseline_values = {key: [] for key in all_specs}
    for (pid, key), value in labels.items():
        require(pid in train_pids, "Nontraining B value reached baseline fitting")
        baseline_values[key].append(value)
    baselines = metrics.build_baselines(baseline_values, all_specs)
    base.write_json(paths["training_baselines.json"], baselines)
    del baseline_values
    tokenizer = AutoTokenizer.from_pretrained(qwen_model, local_files_only=True, padding_side="right")
    require(tokenizer.eos_token_id is not None, "Qwen tokenizer lacks EOS")
    train_features = Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
                               "labels": Sequence(Value("int32"))})
    val_features = Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8"))})
    totals = {}; val_manifest = []; suffix_cache = {}; label_counts = Counter(); label_counts_by_item = {}
    coverage = {}; max_prompt = 0; max_supervised = 0; audit = []
    for split_name, pids, features, arrow_name, manifest_name in (
            ("train", train_pids, train_features, "train.arrow", "training_manifest.jsonl"),
            ("validation", val_pids, val_features, "validation.arrow", "validation_manifest.jsonl")):
        arrow_path, manifest_path = paths[arrow_name], paths[manifest_name]
        arrow_tmp = arrow_path.with_suffix(".arrow.tmp")
        manifest_tmp = manifest_path.with_suffix(".jsonl.tmp")
        seen, keys, included_pids, buffer = set(), set(), set(), []
        count, candidates, families, source_counts, qid_counts = 0, 0, Counter(), Counter(), Counter()
        a_min, a_max = None, 0
        with pa.OSFile(str(arrow_tmp), "wb") as sink, manifest_tmp.open("w", encoding="utf-8") as handle:
            with pa.ipc.new_stream(sink, features.arrow_schema) as writer:
                prepared_people = inputs.stream_people(_source_jobs(source, pids), selection_model, qwen_model,
                                                       workers, selection_budget, max_input_tokens)
                for pid, examples, n_questions in prepared_people:
                    require(pid in pids and pid not in seen and pid not in test_pids,
                            "Duplicate, test or unexpected participant reached development preparation")
                    seen.add(pid); source_counts[n_questions] += 1
                    for record, encoded in examples:
                        validate_prefix(encoded, record, max_input_tokens)
                        require(not diag.has_answers(record), "Target answer entered the prompt manifest")
                        key = (pid, record["item_id"])
                        require(key not in keys and record["pid"] == pid and record["item_id"] in all_specs,
                                "Unexpected or duplicate scalar target")
                        keys.add(key); candidates += 1
                        if split_name == "train":
                            require(key in labels, "Training-B CSV lacks requested participant/item pair")
                            label = labels[key]
                            state = label_state(label, record)
                            label_counts[state] += 1
                            label_counts_by_item.setdefault(record["item_id"], Counter())[state] += 1
                            if state != "valid":
                                continue
                            record, encoded = encode_supervised(encoded, record, label, tokenizer, suffix_cache)
                            max_supervised = max(max_supervised, len(encoded["input_ids"]))
                            if len(audit) < 8 or (record["family"] not in {r["family"] for r in audit}):
                                audit.append({k: record[k] for k in
                                    ("pid", "item_id", "family", "prompt_hash", "input_tokens", "answer_text",
                                     "answer_token_ids", "supervised_tokens", "total_tokens")})
                        else:
                            val_manifest.append(record)
                        handle.write(base.dumps(record) + "\n")
                        buffer.append(encoded)
                        count += 1; included_pids.add(pid); families[record["family"]] += 1
                        qid_counts[record["qid"]] += 1
                        a_min = record["a_units_selected"] if a_min is None else min(a_min, record["a_units_selected"])
                        a_max = max(a_max, record["a_units_selected"])
                        max_prompt = max(max_prompt, record["input_tokens"])
                        if len(buffer) >= 16:
                            writer.write_table(pa.Table.from_pylist(buffer, schema=features.arrow_schema))
                            buffer.clear()
                    if len(seen) % 20 == 0 or len(seen) == len(pids):
                        print(f"Prepared full-B {split_name}: {len(seen)}/{len(pids)} participants; "
                              f"{count} examples ({candidates} metadata targets)", flush=True)
                if buffer:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=features.arrow_schema))
        require(seen == pids and count > 0, f"Incomplete {split_name} preparation")
        if split_name == "validation":
            require(sum(qid_counts[q] for q in base.QIDS) == 1545, "Original validation binary subset changed")
        arrow_tmp.replace(arrow_path); manifest_tmp.replace(manifest_path)
        totals[split_name] = count
        coverage[split_name] = {"participants_seen": len(seen), "participants_with_examples": len(included_pids),
            "metadata_scalar_targets": candidates, "included_examples": count, "family_examples": dict(families),
            "qid_examples": dict(qid_counts), "source_question_counts": dict(source_counts),
            "a_units_selected_min": a_min, "a_units_selected_max": a_max}
    require(sum(label_counts.values()) == coverage["train"]["metadata_scalar_targets"]
            and label_counts["valid"] == totals["train"], "Training label filtering accounting mismatch")
    summary = {"protocol": PROTOCOL, "config_hash": config_hash,
        "training_target_scope": "all scalar B items with valid training responses",
        "train_examples": totals["train"], "validation_examples": totals["validation"],
        "train_participants": len(train_pids), "validation_participants": len(val_pids),
        "train_participants_with_examples": coverage["train"]["participants_with_examples"],
        "training_label_counts": dict(label_counts),
        "training_label_counts_by_item": {key: dict(value) for key, value in label_counts_by_item.items()},
        "coverage": coverage, "max_input_tokens_observed": max_prompt,
        "max_train_total_tokens": max_supervised, "eos_token_id": tokenizer.eos_token_id,
        "masking": "all prompt tokens -100; all answer tokens and one final EOS supervised",
        "mask_audit": audit, "validation_response_labels_opened": False, "C_response_csv_opened": False,
        "test_participant_metadata_loaded": False,
        "artifacts_sha256": {name: base.digest_file(path) for name, path in paths.items()}}
    base.write_json(summary_path, summary)
    train = Dataset.from_file(str(paths["train.arrow"]))
    validation = Dataset.from_file(str(paths["validation.arrow"]))
    require(len(train) == summary["train_examples"] and len(validation) == len(val_manifest)
            == summary["validation_examples"], "Final Arrow example counts mismatch")
    return train, validation, val_manifest, summary
