#!/usr/bin/env python3
"""Prepare participant-grouped, fixed-content Qwen CV without reading C answers.

The original completed Qwen run supplies audited prompt token prefixes. Its
training labels are stripped before pooling with the frozen validation prompts.
Only B labels belonging to a fold's training participants are then read to build
that fold's supervised cache and majority reference. Test participants never
enter the pool. Participant 574 is training-only in every fold.
"""
import argparse
import csv
import importlib.metadata
import json
import random
from collections import Counter
from pathlib import Path

import run_liquid_baseline as base
import qwen_full_data as original

PROTOCOL = "qwen-three-fold-fixed-content-v1"
SPLIT_SEED = 20261010
require = base.require
read_json = original.read_json
hash_config = lambda value: base.digest_text(base.dumps(value))


def resolve_under(root, path):
    path = Path(path)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def make_folds(dev_pids, seed=SPLIT_SEED):
    """Assign each non-anchor participant once; never balance using B or C."""
    ids = [str(pid) for pid in dev_pids]
    require(len(ids) == len(set(ids)) and "574" in ids, "Invalid development participant IDs")
    shuffled = sorted(pid for pid in ids if pid != "574")
    require(len(shuffled) >= 3, "Need at least three non-anchor participants")
    random.Random(seed).shuffle(shuffled)
    quotient, remainder = divmod(len(shuffled), 3)
    assignment, start = {"574": -1}, 0
    for fold in (1, 2, 3):
        size = quotient + int(fold <= remainder)
        assignment.update({pid: fold for pid in shuffled[start:start + size]})
        start += size
    require(start == len(shuffled) and set(assignment) == set(ids), "Incomplete CV assignment")
    return assignment


def read_split(path):
    import pandas as pd
    split = pd.read_csv(path, dtype={"pid": "string"})
    require(split.pid.notna().all() and split.pid.is_unique
            and split["split"].value_counts().to_dict() ==
                {"train": 1440, "validation": 309, "test": 309}, "Original participant split changed")
    require(split.set_index("pid").at["574", "split"] == "train", "Participant 574 must remain training-only")
    return split


def verify_hashes(hashes, description):
    require(bool(hashes), f"Missing {description} hashes")
    for filename, digest in hashes.items():
        path = Path(filename)
        require(path.is_file() and base.digest_file(path) == digest,
                f"{description} artifact missing or changed: {path}")


def check_prefix(ids, mask, record, special):
    require(isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids),
            "Prompt must contain a flat nonnegative integer token list")
    require(0 < len(ids) == int(record["input_tokens"]) <= 32760
            and mask == [1] * len(ids), "Prompt token count or attention mask changed")
    start, end = special["system_prefix_ids"], special["assistant_prefix_ids"]
    require(ids[:len(start)] == start and ids[-len(end):] == end,
            "Prompt chat boundary changed; expected system/user followed by assistant generation prefix")
    require(isinstance(record["prompt_hash"], str) and len(record["prompt_hash"]) == 64,
            "Prompt message hash is missing")


def strip_supervision(row, record, special):
    """Audit answer-only masking before removing every supervised token."""
    ids, mask, labels = list(row["input_ids"]), list(row["attention_mask"]), list(row["labels"])
    n = int(record["input_tokens"])
    label = int(record["label_b"])
    require(label in (1, 2), "Invalid source B label")
    suffix = [special["digit_ids"][str(label)], special["eos_token_id"]]
    require(len(ids) == len(mask) == len(labels) == n + 2
            and labels == [-100] * n + suffix and ids[-2:] == suffix
            and mask == [1] * len(ids), "Source cache no longer supervises precisely digit plus EOS")
    prefix = {"input_ids": ids[:-2], "attention_mask": mask[:-2]}
    check_prefix(prefix["input_ids"], prefix["attention_mask"], record, special)
    return prefix


def _arrow_write(path, rows, supervised=False):
    import pyarrow as pa
    from datasets import Features, Sequence, Value
    features = {"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8"))}
    if supervised:
        features["labels"] = Sequence(Value("int32"))
    schema = Features(features).arrow_schema
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp, buffer = path.with_suffix(path.suffix + ".tmp"), []
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
        for row in rows:
            require(set(row) == set(features), "Unexpected columns in prompt/training cache")
            buffer.append(row)
            if len(buffer) == 16:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    tmp.replace(path)


def prepare_cv(root, source_run, cv_root):
    """Create audited, label-free development pool; returns immutable CV config."""
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoTokenizer
    root = Path(root).resolve()
    source_run, cv_root = resolve_under(root, source_run), resolve_under(root, cv_root)
    require(cv_root != source_run and source_run not in cv_root.parents
            and cv_root not in source_run.parents and root / "data" not in cv_root.parents,
            "CV output must be separate from the current run and source data")
    source = read_json(source_run / "run_config.json")
    complete = read_json(source_run / "training_complete.json")
    prep = read_json(source_run / "preprocessing_summary.json")
    selection = read_json(source_run / "best_checkpoint.json")
    source_hash = hash_config(source)
    require(source["protocol"] == original.PROTOCOL
            and complete["protocol"] == original.PROTOCOL
            and complete["config_hash"] == prep["config_hash"] == source_hash,
            "Current Qwen run must finish training and validation before CV")
    require(selection["training_complete_sha256"] == base.digest_file(source_run / "training_complete.json")
            and selection["test_evaluated"] is False
            and len(selection["all_epochs"]) == source["epochs"], "Current validation evaluation is incomplete")
    require(source["epochs"] == 3 and source["training_examples"] == prep["examples"] == 7200
            and prep["participants"] == 1440 and prep["training_labels"] == "B"
            and source["max_input_tokens"] == 32760 and source["max_new_tokens"] == 8
            and source["learning_rate"] == 1e-5 and source["micro_batch_size"] == 1
            and source["gradient_accumulation"] == 16, "Unexpected current Qwen training protocol")
    split_path = root / "data/participant_split.csv"
    b_path = root / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave1_3_response.csv"
    for path in (split_path, b_path):
        require(base.digest_file(path) == source["inputs_sha256"][str(path)], "Source split/B CSV changed")
    split = read_split(split_path)
    train_pids = set(split.loc[split["split"].eq("train"), "pid"])
    val_pids = set(split.loc[split["split"].eq("validation"), "pid"])
    dev_pids = train_pids | val_pids
    model = Path(source["model"])
    require(not (model / "adapter_config.json").exists()
            and original.model_files_sha256(model) == source["model_files_sha256"]
            and source["model_files_sha256"].get("model.safetensors") == original.OFFICIAL_WEIGHTS_SHA256,
            "CV must initialize every fold from the verified original Qwen weights")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, padding_side="right")
    require(base.digest_text(tokenizer.chat_template) == source["chat_template_hash"], "Source tokenizer template changed")
    special = {"eos_token_id": tokenizer.eos_token_id,
        "digit_ids": {str(d): tokenizer.encode(str(d), add_special_tokens=False) for d in (1, 2)},
        "system_prefix_ids": tokenizer.encode("<|im_start|>system\n", add_special_tokens=False),
        "assistant_prefix_ids": tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)}
    require(special["eos_token_id"] is not None and all(len(v) == 1 for v in special["digit_ids"].values()),
            "Expected one-token digits and an EOS token")
    special["digit_ids"] = {k: v[0] for k, v in special["digit_ids"].items()}
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "datasets", "accelerate", "pandas", "pyarrow")}
    require(versions == source["versions"] and versions["transformers"] == "5.2.0",
            "Use the unchanged liquid_tf52 environment from the current Qwen run")
    artifacts = {"train.arrow": source_run / "prepared/train.arrow",
                 "training_manifest.csv": source_run / "training_manifest.csv",
                 "validation_prompts.jsonl": source_run / "prepared/validation_prompts.jsonl"}
    for name, path in artifacts.items():
        require(base.digest_file(path) == prep["artifacts_sha256"][name], f"Source prepared artifact changed: {name}")
    source_paths = list(artifacts.values()) + [source_run / name for name in
        ("run_config.json", "training_complete.json", "preprocessing_summary.json", "best_checkpoint.json")]
    source_hashes = {str(path): base.digest_file(path) for path in source_paths}
    inputs_hashes = {str(path): base.digest_file(path) for path in (split_path, b_path)}
    cv_root.mkdir(parents=True, exist_ok=True)
    if (cv_root / "cv_config.json").exists():
        cached, _, _, _ = load_cv(cv_root)
        require(cached["root"] == str(root) and cached["source_run"] == str(source_run)
                and cached["source_artifacts_sha256"] == source_hashes
                and cached["inputs_sha256"] == inputs_hashes and cached["versions"] == versions,
                "Existing CV source/configuration changed; choose a new output directory")
        return cached
    manifest = pd.read_csv(artifacts["training_manifest.csv"], dtype={"pid": "string", "qid": "string"})
    train = Dataset.from_file(str(artifacts["train.arrow"]))
    require(len(train) == len(manifest) == 7200 and set(manifest.pid) == train_pids
            and not manifest[["pid", "qid"]].duplicated().any()
            and set(manifest.qid) == set(base.QIDS), "Original training cache coverage changed")
    prompts = base.read_jsonl(artifacts["validation_prompts.jsonl"])
    require(len(prompts) == 1545 and {str(p["pid"]) for p in prompts} == val_pids,
            "Original validation prompt coverage changed")
    records = []
    for row in manifest.to_dict("records"):
        records.append({"pid": str(row["pid"]), "qid": str(row["qid"]),
                        "prompt_hash": row["prompt_hash"], "input_tokens": int(row["input_tokens"]),
                        "source_split": "train"})
    for prompt in prompts:
        require(base.digest_text(base.dumps(prompt["messages"])) == prompt["prompt_hash"]
                and [m["role"] for m in prompt["messages"]] == ["system", "user"]
                and prompt["messages"][0]["content"] == source["system"], "Frozen validation prompt changed")
        records.append({"pid": str(prompt["pid"]), "qid": str(prompt["qid"]),
                        "prompt_hash": prompt["prompt_hash"], "input_tokens": int(prompt["input_tokens"]),
                        "source_split": "validation"})
    pool_manifest = pd.DataFrame(records)
    require(len(pool_manifest) == 8745 and set(pool_manifest.pid) == dev_pids
            and not pool_manifest[["pid", "qid"]].duplicated().any()
            and set(pool_manifest.qid) == set(base.QIDS)
            and pool_manifest.groupby("pid").size().eq(5).all(), "Development pool includes missing, extra, or test cases")
    prepared = cv_root / "prepared"
    prepared.mkdir(exist_ok=True)
    paths = {"pool.arrow": prepared / "pool.arrow", "pool_manifest.csv": prepared / "pool_manifest.csv",
             "folds.csv": prepared / "folds.csv"}
    def rows():
        for index, row in enumerate(train):
            yield strip_supervision(row, manifest.iloc[index].to_dict(), special)
        for prompt in prompts:
            ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
                        add_generation_prompt=True, return_dict=False, truncation=False)
            row = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            check_prefix(ids, row["attention_mask"], prompt, special)
            yield row
    _arrow_write(paths["pool.arrow"], rows())
    pool_manifest.to_csv(paths["pool_manifest.csv"], index=False)
    assignment = make_folds(dev_pids)
    pd.DataFrame([{"pid": pid, "validation_fold": assignment[pid]}
                  for pid in sorted(assignment)]).to_csv(paths["folds.csv"], index=False)
    config = {"protocol": PROTOCOL, "root": str(root), "source_run": str(source_run),
        "cv_root": str(cv_root), "model": str(model), "model_files_sha256": source["model_files_sha256"],
        "qids": list(base.QIDS), "system": source["system"], "versions": versions,
        "max_input_tokens": 32760, "max_new_tokens": 8, "epochs": 3,
        "epoch_policy": "fixed third epoch for every fold; no validation checkpoint selection",
        "learning_rate": 1e-5, "micro_batch_size": 1, "gradient_accumulation": 16,
        "dataloader_num_workers": 4, "num_workers": 4, "seed": 42, "split_seed": SPLIT_SEED,
        "optimizer": "adamw_torch", "weight_decay": 0.0, "warmup_fraction": 0.05,
        "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
        "parameter_dtype": "torch.float32", "precision": "bf16-if-supported-otherwise-fp16 autocast",
        "trainable_parameters_expected": original.EXPECTED_PARAMETERS,
        "initialization": "fresh original Qwen2.5-0.5B-Instruct for each fold",
        "loss": "digit plus EOS only; all prompt and padding labels -100",
        "development_participants": 1749, "pool_examples": 8745, "fold_count": 3,
        "training_only_pids": ["574"], "test_participants_excluded": 309,
        "context_policy": "exact current-run A content; source Qwen prefixes with digit/EOS stripped; no refill or truncation",
        "special_token_ids": special, "source_artifacts_sha256": source_hashes,
        "inputs_sha256": inputs_hashes,
        "artifacts_sha256": {name: base.digest_file(path) for name, path in paths.items()},
        "data_script_sha256": base.digest_file(__file__),
        "validation_label_policy": "C read only by separate evaluator after predictions; B reference scored separately",
        "interpretation": "development CV after earlier validation use; final test remains untouched"}
    base.write_json(cv_root / "cv_config.json", config)
    load_cv(cv_root)
    print(f"Prepared label-free Qwen CV pool: 1749 participants, {len(pool_manifest)} examples; test excluded.", flush=True)
    return config


def load_cv(cv_root):
    """Verify cache/config hashes and return config, pool, manifest, assignments."""
    import pandas as pd
    from datasets import Dataset
    cv_root = Path(cv_root).resolve()
    config = read_json(cv_root / "cv_config.json")
    require(config["protocol"] == PROTOCOL and config["cv_root"] == str(cv_root)
            and config["epochs"] == 3 and config["split_seed"] == SPLIT_SEED
            and config["training_only_pids"] == ["574"]
            and config["data_script_sha256"] == base.digest_file(__file__), "CV protocol/script changed")
    verify_hashes(config["source_artifacts_sha256"], "Current Qwen source")
    verify_hashes(config["inputs_sha256"], "CV source data")
    for name, digest in config["artifacts_sha256"].items():
        require(name in {"pool.arrow", "pool_manifest.csv", "folds.csv"}
                and base.digest_file(cv_root / "prepared" / name) == digest, f"CV prepared cache changed: {name}")
    pool = Dataset.from_file(str(cv_root / "prepared/pool.arrow"))
    manifest = pd.read_csv(cv_root / "prepared/pool_manifest.csv", dtype={"pid": "string", "qid": "string"})
    folds = pd.read_csv(cv_root / "prepared/folds.csv", dtype={"pid": "string"})
    require(set(pool.column_names) == {"input_ids", "attention_mask"}
            and list(manifest.columns) == ["pid", "qid", "prompt_hash", "input_tokens", "source_split"]
            and len(pool) == len(manifest) == config["pool_examples"] == 8745,
            "CV feature pool must contain no answer labels")
    split = read_split(Path(config["root"]) / "data/participant_split.csv")
    dev_pids = set(split.loc[split["split"].isin(["train", "validation"]), "pid"])
    require(set(manifest.pid) == set(folds.pid) == dev_pids and len(folds) == 1749
            and folds.pid.is_unique and not manifest[["pid", "qid"]].duplicated().any()
            and manifest.groupby("pid").size().eq(5).all()
            and dict(zip(folds.pid, folds.validation_fold)) == make_folds(dev_pids),
            "CV participant split or row coverage changed")
    require(Counter(folds.validation_fold) == {-1: 1, 1: 583, 2: 583, 3: 582}, "Unexpected fold sizes")
    return config, pool, manifest, folds


def read_fold_b(response_path, all_pids, train_pids):
    """Read IDs first; skip every held-out/test row before parsing B columns."""
    import pandas as pd
    ids = pd.read_csv(response_path, usecols=["pid"], dtype={"pid": "string"})["pid"]
    require(ids.notna().all() and ids.is_unique and set(ids) == set(all_pids), "B response IDs differ from original split")
    skipped = [index + 1 for index, pid in enumerate(ids) if pid not in train_pids]
    labels = pd.read_csv(response_path, skiprows=skipped, usecols=["pid", *base.QIDS],
                         dtype={"pid": "string"}).set_index("pid")
    require(labels.index.is_unique and set(labels.index) == set(train_pids), "Fold B training selection failed")
    for qid in base.QIDS:
        answers = labels[qid].dropna()
        require(len(answers) > 0 and answers.isin([1, 2]).all(), f"Invalid fold training B labels: {qid}")
    return labels


def majority_from_manifest(manifest):
    counts = {qid: {str(d): int((manifest.qid.eq(qid) & manifest.label_b.eq(d)).sum())
                    for d in (1, 2)} for qid in base.QIDS}
    require(all(sum(row.values()) > 0 for row in counts.values()), "Missing question in fold training labels")
    predictions = {qid: (1 if row["1"] >= row["2"] else 2) for qid, row in counts.items()}
    return predictions, counts


def prepare_fold(cv_root, fold_number):
    """Return (fold_config, supervised train Dataset), using only fold-training B."""
    import pandas as pd
    from datasets import Dataset
    require(type(fold_number) is int and fold_number in (1, 2, 3), "Fold number must be 1, 2, or 3")
    cv_root = Path(cv_root).resolve()
    config, pool, manifest, folds = load_cv(cv_root)
    val_pids = set(folds.loc[folds.validation_fold.eq(fold_number), "pid"])
    train_pids = set(folds.pid) - val_pids
    train_indices = manifest.index[manifest.pid.isin(train_pids)].tolist()
    val_indices = manifest.index[manifest.pid.isin(val_pids)].tolist()
    require("574" in train_pids and "574" not in val_pids and not train_pids & val_pids
            and len(train_indices) + len(val_indices) == len(pool), "Invalid participant fold")
    out = cv_root / f"fold_{fold_number}"
    out.mkdir(exist_ok=True)
    paths = {"train.arrow": out / "prepared/train.arrow", "training_manifest.csv": out / "training_manifest.csv",
             "majority.json": out / "majority.json"}
    expected = {"protocol": PROTOCOL, "fold_number": fold_number, "cv_config_hash": hash_config(config),
        "train_pids": sorted(train_pids), "val_pids": sorted(val_pids), "train_indices": train_indices,
        "val_indices": val_indices, "train_examples": len(train_indices), "val_examples": len(val_indices)}
    if (out / "fold_config.json").exists():
        saved = read_json(out / "fold_config.json")
        require(all(saved.get(k) == v for k, v in expected.items()), "Existing fold configuration changed")
        for name, path in paths.items():
            require(base.digest_file(path) == saved["train_artifacts_sha256"][name], f"Fold training cache changed: {name}")
        train = Dataset.from_file(str(paths["train.arrow"]))
        require(len(train) == expected["train_examples"], "Fold training cache length changed")
        return saved, train
    split = read_split(Path(config["root"]) / "data/participant_split.csv")
    b_path = Path(config["root"]) / "data/Twin-2K-500/question_catalog_and_human_response_csv/wave1_3_response.csv"
    labels = read_fold_b(b_path, set(split.pid), train_pids)
    records = []
    for index in train_indices:
        record = manifest.iloc[index].to_dict()
        answer = labels.at[record["pid"], record["qid"]]
        require(not pd.isna(answer) and answer in (1, 2), "Pool question lacks a corresponding fold training B answer")
        record.update(pool_index=index, label_b=int(answer), total_tokens=int(record["input_tokens"]) + 2)
        records.append(record)
    training_manifest = pd.DataFrame(records)
    require(len(training_manifest) == int(labels.notna().sum().sum()), "Fold B answer coverage differs from prompt pool")
    majority, counts = majority_from_manifest(training_manifest)
    special = config["special_token_ids"]
    def supervised_rows():
        for record in records:
            prefix = pool[int(record["pool_index"])]
            check_prefix(prefix["input_ids"], prefix["attention_mask"], record, special)
            suffix = [special["digit_ids"][str(record["label_b"])], special["eos_token_id"]]
            n = len(prefix["input_ids"])
            yield {"input_ids": prefix["input_ids"] + suffix,
                   "attention_mask": [1] * (n + 2), "labels": [-100] * n + suffix}
    _arrow_write(paths["train.arrow"], supervised_rows(), supervised=True)
    training_manifest.to_csv(paths["training_manifest.csv"], index=False)
    base.write_json(paths["majority.json"], {"predictions": majority, "counts": counts,
        "source": "this fold's training B labels only", "tie_break": "smaller digit (1)"})
    fold_config = dict(expected, majority_prediction=majority, training_b_counts=counts,
        train_artifacts_sha256={name: base.digest_file(path) for name, path in paths.items()})
    base.write_json(out / "fold_config.json", fold_config)
    print(f"Fold {fold_number}: train {len(train_pids)} participants/{len(train_indices)} examples; "
          f"validation {len(val_pids)} participants/{len(val_indices)} examples", flush=True)
    return fold_config, Dataset.from_file(str(paths["train.arrow"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--source-run", type=Path, default=Path("outputs/qwen_full_v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_cv_v1"))
    parser.add_argument("--fold", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    args.root = args.root.resolve()
    out = resolve_under(args.root, args.out)
    prepare_cv(args.root, args.source_run, out)
    for fold in ([args.fold] if args.fold else [1, 2, 3]):
        prepare_fold(out, fold)


if __name__ == "__main__":
    main()
