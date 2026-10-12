#!/usr/bin/env python3
"""Two-stage validation: frozen Qwen3-Embedding-0.6B kNN, then Qwen2.5-0.5B few-shot answers.

No fine-tuning, adapters, reranker, test evaluation, or validation-label-based
neighbor selection. Resume by repeating exactly the same command. --stage knn
scores stage 1 alone; a later --stage all reuses its embeddings and predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from support import data_io, reporting
from support.run_cpu_comparisons import verify_retest_metadata
import runtime as rt

PROTOCOL = "qwen-embedding06-shared-a-definitions-v1"
ROOT = Path("/root/autodl-tmp/twin_qwen")
DEFAULT_MODELS = Path("/root/autodl-tmp/models")
CPU_STAGES = ("condition_baseline", "embedding_knn")
LLM_STAGES = ("qwen_no_demo", "qwen_retrieved")
require = data_io.require


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--embedding-model", type=Path, default=DEFAULT_MODELS / "Qwen3-Embedding-0.6B")
    p.add_argument("--generator-model", type=Path, default=DEFAULT_MODELS / "Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", type=Path, default=Path("outputs/qwen_embedding06_shared_v3"))
    p.add_argument("--embedding-cache", type=Path)
    p.add_argument("--stage", choices=("all", "knn"), default="all")
    p.add_argument("--k", type=int, default=5, help="Same fixed neighbors for voting and demonstrations")
    p.add_argument("--embedding-batch-size", type=int, default=8)
    p.add_argument("--embedding-max-tokens", type=int, default=2048, help="All A is covered by chunks; never first-N truncation")
    p.add_argument("--generation-batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4, help="CPU preparation threads; no CUDA worker processes")
    p.add_argument("--max-input-tokens", type=int, default=12288)
    p.add_argument("--own-a-budget", type=int, default=1600,
                   help="Whole-unit selection budget measured as original Q+A tokens before sharing definitions")
    p.add_argument("--donor-a-budget", type=int, default=1600,
                   help="Per-neighbor whole-unit selection budget measured as original Q+A tokens before sharing definitions")
    p.add_argument("--wait-for-idle", action="store_true", help="Wait for current training/comparison locks and GPU jobs")
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--score-only", action="store_true")
    return p


def resolve_args(args):
    args.root = args.root.resolve()
    args.dataset_dir = (args.dataset_dir or args.root / "data/Twin-2K-500").resolve()
    args.split = (args.split or args.root / "data/participant_split.csv").resolve()
    args.out = (args.out if args.out.is_absolute() else args.root / args.out).resolve()
    args.embedding_cache = (args.embedding_cache or args.root / "data/qwen_embedding06_cache_v1").resolve()
    args.embedding_model, args.generator_model = args.embedding_model.resolve(), args.generator_model.resolve()
    require(1 <= args.k <= 10 and args.workers >= 1, "Choose k in 1..10 and positive workers")
    require(args.embedding_batch_size >= 1 and args.generation_batch_size >= 1, "Batch sizes must be positive")
    require(256 <= args.embedding_max_tokens <= 32768 and 256 <= args.max_input_tokens <= 32752,
            "Invalid embedding/generation input token budget")
    require(args.own_a_budget > 0 and args.donor_a_budget > 0, "A snippet budgets must be positive")
    protected = [args.dataset_dir, args.embedding_model, args.generator_model,
                 args.root / "outputs/qwen_full_b_lora_v4_b16_a2",
                 args.root / "outputs/qwen_epoch1_comparisons_v1"]
    require(all(args.out != p and p not in args.out.parents for p in protected),
            "Use a new output directory outside existing model, data, training and comparison directories")
    return args


def preflight(args, engine):
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "numpy", "pandas", "pyarrow")}
    major_minor = tuple(int(value) for value in versions["transformers"].split(".")[:2])
    require(major_minor >= (4, 51), "Qwen3 embedding requires transformers>=4.51; use your liquid_tf52 environment")
    split = engine.inputs.read_split(args.split)
    chunks = sorted((args.dataset_dir / "wave_split/chunks").glob("*.parquet"))
    responses = args.dataset_dir / "question_catalog_and_human_response_csv"
    files = [args.split, *chunks, responses / "wave1_3_response.csv", responses / "wave4_response.csv"]
    require(len(chunks) == 7 and all(p.is_file() for p in files), "Pinned dataset/split files are incomplete")
    for path, expected_type in ((args.embedding_model, "qwen3"), (args.generator_model, "qwen2")):
        require((path / "config.json").exists(), f"Downloaded model missing: {path}")
        config = rt.read_json(path / "config.json")
        require(config.get("model_type") == expected_type, f"Wrong model type at {path}")
        require(not (path / "adapter_config.json").exists(), "Use original downloaded models, not adapters")
    embedding_config = rt.read_json(args.embedding_model / "config.json")
    require(embedding_config.get("hidden_size") == 1024 and embedding_config.get("num_hidden_layers") == 28,
            "Expected Qwen3-Embedding-0.6B architecture, not 4B/8B")
    from tokenizer_smoke import check_tokenizers
    print("Checking actual local tokenizers on CPU before loading model weights...", flush=True)
    args.tokenizer_smoke_report = check_tokenizers(args.embedding_model, args.generator_model,
                                                  embedding_max_tokens=args.embedding_max_tokens)
    print(f"Tokenizer checks passed: native embedding template "
          f"{args.tokenizer_smoke_report['special_token_layout']}; A content and padding preserved.", flush=True)
    require(shutil.disk_usage(args.root).free >= 4 * 2**30, "Need at least 4 GiB free for caches, prepared inputs and outputs")
    if not args.score_only:
        import torch
        require(torch.cuda.is_available(), "CUDA is required to create embeddings and run Qwen answers")
    print(f"Preflight: {len(split['train'])} train / {len(split['validation'])} validation / "
          f"{len(split['test'])} reserved test; k={args.k}; CPU workers={args.workers}; "
          f"embedding batch={args.embedding_batch_size}; generation batch={args.generation_batch_size}.", flush=True)
    return split, files, versions


def freeze_run(args, files, versions):
    print("Fingerprinting local model weights, dataset and source code...", flush=True)
    source_dir = Path(__file__).resolve().parent
    sources = sorted(p for p in source_dir.rglob("*") if p.suffix in {".py", ".json"}
                     and "__pycache__" not in p.parts and not p.name.startswith("tests_")
                     and p.name != "PACKAGE_SHA256.json")
    config = {
        "protocol": PROTOCOL, "evaluation_split": "validation", "primary_reference": "C", "secondary_reference": "B",
        "dataset_dir": str(args.dataset_dir), "split": str(args.split),
        "embedding_model": str(args.embedding_model), "generator_model": str(args.generator_model),
        "embedding_files_sha256": rt.model_files(args.embedding_model),
        "generator_files_sha256": rt.model_files(args.generator_model),
        "inputs_sha256": {str(path): rt.sha256(path) for path in files},
        "source_sha256": {str(path.relative_to(source_dir)): rt.sha256(path) for path in sources},
        "versions": versions, "k": args.k,
        "tokenizer_smoke": args.tokenizer_smoke_report,
        "embedding_batch_size": args.embedding_batch_size, "embedding_max_tokens": args.embedding_max_tokens,
        "generation_batch_size": args.generation_batch_size, "max_input_tokens": args.max_input_tokens,
        "own_a_budget": args.own_a_budget, "donor_a_budget": args.donor_a_budget,
        "planned_methods": list(CPU_STAGES + LLM_STAGES), "training_labels": "B, training participants only",
        "candidate_policy": "exact target text/options/schema; actual product and exact price, not pricing QID position",
        "retrieval_inputs": "A-only profiles; all complete A units covered by embedding chunks; no B/C labels or PIDs",
        "prompt_policy": "shared exact A definitions once; per-person answer references; same selected donors as voting; matched own-A for no-demo control; output one scalar only",
        "a_budget_semantics": "original complete Q+A unit tokens before dictionary compression; unchanged lexical selection",
        "generator": "original Qwen2.5-0.5B-Instruct; no adapter or fine-tuning",
        "parameter_budget_note": "The 0.6B embedding component is an additional retrieval baseline, separate from the <0.5B fine-tuning POC.",
        "test_evaluated": False,
    }
    rt.freeze(args.out / "run_config.json", config)
    return config


def assert_unchanged(args, config):
    for path, expected in config["inputs_sha256"].items():
        require(rt.sha256(path) == expected, f"Dataset or split changed: {path}")
    rt.verify_files(Path(__file__).resolve().parent, config["source_sha256"])
    rt.verify_files(args.embedding_model, config["embedding_files_sha256"])
    rt.verify_files(args.generator_model, config["generator_files_sha256"])


def save_cpu_predictions(args, engine, people, split, train_labels, config):
    done_path = args.out / "neighbor_predictions_complete.json"
    if done_path.exists():
        rt.verify_files(args.out, rt.read_json(done_path)["artifacts_sha256"])
        rows = engine.base.read_jsonl(args.out / "embedding_knn/raw_predictions.jsonl")
        neighbors = {(str(row["pid"]), row["item_id"]): row["algorithm_details"] for row in rows}
        print("Reusing frozen neighbor votes and eligibility baseline.", flush=True)
        return neighbors
    require(not args.score_only, "Neighbor predictions incomplete; run without --score-only")
    from embedding_index import build_profile_embeddings
    from neighbors import build_neighbors
    print("Stage 1: building/reusing A-only Qwen3-Embedding-0.6B vectors...", flush=True)
    embedded = build_profile_embeddings(people, sorted(split["train"], key=lambda p: (len(p), p)),
        sorted(split["validation"], key=lambda p: (len(p), p)),
        args.embedding_model, args.embedding_cache, batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_tokens, workers=args.workers,
        model_fingerprint=config["embedding_files_sha256"])
    rt.write_json(args.out / "embedding_report.json", embedded["report"])
    print("Selecting top-k within each exact target condition, then voting/median aggregation...", flush=True)
    fitted = build_neighbors(people, train_labels, split["train"], split["validation"], embedded, k=args.k)
    rt.write_json(args.out / "neighbor_report.json", fitted["report"])
    paths = ["embedding_report.json", "neighbor_report.json"]
    for name, key in (("embedding_knn", "knn"), ("condition_baseline", "condition_baseline")):
        manifest = data_io.make_manifest(people, split["validation"], engine, name)
        require(len(manifest) == 29690, "Full-B validation case count changed")
        records = data_io.make_records(fitted[key], manifest, engine, name)
        folder = args.out / name
        rt.write_jsonl(folder / "manifest.jsonl", manifest)
        rt.write_jsonl(folder / "raw_predictions.jsonl", records)
        paths.extend([f"{name}/manifest.jsonl", f"{name}/raw_predictions.jsonl"])
    rt.write_json(done_path, {"protocol": PROTOCOL, "config_sha256": data_io.digest(config),
        "rows": 29690, "artifacts_sha256": rt.artifact_manifest(args.out, paths),
        "completed_utc": rt.now(), "validation_answers_used_for_predictions": False})
    return fitted["knn"]


def prepare_llm(args, engine, people, split, train_labels, neighbors):
    done = [args.out / name / "prepared_complete.json" for name in LLM_STAGES]
    if not all(path.exists() for path in done):
        require(not args.score_only, "Qwen prompts incomplete; run without --score-only")
        require(not any((args.out / name / "unique_predictions.jsonl").exists() for name in LLM_STAGES),
                "Cannot rebuild partly inferred prompts; use a fresh output directory")
        from transformers import AutoTokenizer
        from retrieval_prompts import iter_prompt_pairs
        tokenizer = AutoTokenizer.from_pretrained(str(args.generator_model), local_files_only=True, padding_side="left")
        manifest = data_io.make_manifest(people, split["validation"], engine, "paired_qwen_inputs")
        writers = [rt.PreparedWriter(args.out / name, engine) for name in LLM_STAGES]
        sample_index = []
        sampled_kinds = set()
        print("Stage 2 preparation: shared A definitions, compact per-person answers, matched no-demo control...", flush=True)
        for pair in iter_prompt_pairs(engine, tokenizer, manifest, people, neighbors, train_labels, split["train"],
                max_input_tokens=args.max_input_tokens, own_budget=args.own_a_budget,
                donor_budget=args.donor_a_budget, k=args.k, workers=args.workers):
            for writer, (record, encoded) in zip(writers, pair):
                writer.add(record, encoded)
            first = pair[0][0]
            kind = "legacy_binary" if first.get("trained_binary") else first["family"]
            if kind not in sampled_kinds:
                sampled_kinds.add(kind)
                for name, (record, encoded) in zip(LLM_STAGES, pair):
                    relative = f"prompt_examples/{name}_{kind}.txt"
                    destination = args.out / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(tokenizer.decode(encoded["input_ids"], skip_special_tokens=False),
                                           encoding="utf-8")
                    sample_index.append({"method": name, "kind": kind, "pid": record["pid"],
                        "item_id": record["item_id"], "input_tokens": record["input_tokens"],
                        "repeated_input_tokens_same_a": record["repeated_input_tokens_same_a"],
                        "prompt_hash": record["prompt_hash"], "file": relative,
                        "file_sha256": rt.sha256(destination)})
        rt.write_json(args.out / "prompt_examples/index.json", {
            "description": "First prepared example of each question family; decoded exact prepared token IDs. Local only; includes survey content.",
            "examples": sample_index})
        for writer in writers:
            require(len(writer.manifest) == len(manifest), "Prompt preparation lost cases")
            writer.finish()
        print(f"Actual prompt examples saved in {args.out / 'prompt_examples'}", flush=True)
        del tokenizer
    stages = {name: rt.prepared(args.out / name) for name in LLM_STAGES}
    for name, (_, manifest) in stages.items():
        require(len(manifest) == 29690, f"Incomplete prepared stage {name}")
    control, retrieved = (stages[name][1] for name in LLM_STAGES)
    for left, right in zip(control, retrieved):
        require((left['pid'], left['item_id'], left['matched_own_a_prompt_hash']) ==
                (right['pid'], right['item_id'], right['matched_own_a_prompt_hash']),
                "No-demo/retrieval own-A pairing changed")
        require(left['shared_own_semantic_hash'] == right['shared_own_semantic_hash'] and
                left['shared_own_answer_ids'] == right['shared_own_answer_ids'],
                "Shared question/answer mapping differs between paired own-A prompts")
        expected = neighbors[(str(right['pid']), right['item_id'])]['neighbors']
        require(right['retrieval_donor_pids'] == [entry['pid'] for entry in expected],
                "Prompt donors differ from the voting neighbors")
        require(right['retrieval_source_items'] ==
                [{"pid": entry['pid'], "item_id": entry['item_id']} for entry in expected],
                "Prompt source question IDs differ from the voting neighbors")
    rt.write_json(args.out / "paired_demonstration_coverage.json", {
        "cases": len(retrieved), "same_neighbors_as_vote": True, "matched_own_A": True,
        "demonstrations_per_prompt": dict(Counter(row['demonstrations_in_prompt'] for row in retrieved)),
        "empty_donor_A_snippets_per_prompt": dict(Counter(row['paired_demo_empty_a_count'] for row in retrieved)),
        "control_demonstrations_per_prompt": dict(Counter(row['demonstrations_in_prompt'] for row in control)),
        "retrieved_input_tokens_min": min(row['input_tokens'] for row in retrieved),
        "retrieved_input_tokens_max": max(row['input_tokens'] for row in retrieved),
    })
    compression = {"protocol": PROTOCOL,
        "comparison": "Actual shared prompt vs repeated question text using the SAME FINAL selected A units, donors, B labels and target; counts include chat template.",
        "selection_budget": "Full Q+A equivalent tokens before sharing definitions; compression does not automatically add more A responses.",
        "methods": {}}
    for name, (_, rows) in stages.items():
        shared = [row['input_tokens'] for row in rows]
        repeated = [row['repeated_input_tokens_same_a'] for row in rows]
        savings = [before - after for before, after in zip(repeated, shared)]
        compression['methods'][name] = {
            "cases": len(rows), "shared_input_tokens_mean": sum(shared) / len(rows),
            "shared_input_tokens_min": min(shared), "shared_input_tokens_max": max(shared),
            "repeated_input_tokens_mean_same_a": sum(repeated) / len(rows),
            "token_savings_mean_same_a": sum(savings) / len(rows),
            "token_savings_fraction_of_repeated_total": sum(savings) / sum(repeated),
            "shorter_cases": sum(value > 0 for value in savings),
            "equal_length_cases": sum(value == 0 for value in savings),
            "longer_cases": sum(value < 0 for value in savings),
            "shared_definition_count_mean": sum(row['shared_definition_count'] for row in rows) / len(rows),
            "shared_definition_reuse_count_mean": sum(row['shared_definition_reuse_count'] for row in rows) / len(rows)}
    rt.write_json(args.out / "prompt_compression_report.json", compression)
    retrieved_stats = compression['methods']['qwen_retrieved']
    print(f"Shared retrieved prompts: mean {retrieved_stats['shared_input_tokens_mean']:.0f} tokens; "
          f"{retrieved_stats['token_savings_fraction_of_repeated_total']:+.1%} token saving against repeated text with the same A.", flush=True)
    return stages


def infer_llm(args, engine, stages):
    pending = [name for name in LLM_STAGES if not (args.out / name / "generation_complete.json").exists()]
    model = tokenizer = None
    if pending:
        require(not args.score_only, "Qwen predictions incomplete; run without --score-only")
        rt.release_cuda()
        model, tokenizer, info = rt.load_generator(args.generator_model)
        rt.write_json(args.out / "generator_loading.json", info)
    try:
        for name, (dataset, manifest) in stages.items():
            rt.generate_stage(args.out / name, dataset, manifest, engine, model, tokenizer,
                              args.generation_batch_size)
    finally:
        if model is not None:
            del model, tokenizer
            rt.release_cuda()


def collect_results(args, method_names):
    names = ["run_config.json", "embedding_report.json", "neighbor_report.json", "report.txt",
             "reporting_scope.txt", "evaluation_complete.json", "comparison.csv", "prediction_agreement.csv",
             "comparison_coverage.csv", "paired_demonstration_coverage.json", "prompt_compression_report.json"]
    files = [args.out / name for name in names if (args.out / name).is_file()]
    for name in method_names:
        folder = args.out / name
        files.extend(path for path in [folder / "summary.json", *sorted((folder / "scores").glob("metrics_*.csv"))]
                     if path.is_file())
    # Per-person vectors, inputs, answers and raw prediction records stay local.
    target = args.out / "aggregate_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.name == "embedding_report.json":
                info = rt.read_json(path)
                info = {k: v for k, v in info.items()
                        if k not in {"profiles", "profile_reports", "per_profile", "profile_statistics", "a_profile_hashes"}}
                archive.writestr(path.name, json.dumps(info, indent=2, ensure_ascii=False))
            else:
                archive.write(path, str(path.relative_to(args.out)))
        for path in sorted((args.out / 'llm_control_comparison').glob('*')):
            if path.suffix in {'.csv', '.txt'}:
                archive.write(path, str(path.relative_to(args.out)))
    return target


def score(args, engine, split, all_specs, reference, config, method_names):
    jobs, paths = {}, []
    for name in method_names:
        folder = args.out / name
        manifest = engine.base.read_jsonl(folder / "manifest.jsonl")
        records = engine.base.read_jsonl(folder / "raw_predictions.jsonl")
        require(len(records) == len(manifest) == 29690, f"Incomplete predictions in {name}")
        require({(str(r['pid']), r['item_id']) for r in records} ==
                {(str(r['pid']), r['item_id']) for r in manifest}, "Prediction keys differ")
        jobs[name] = (records, manifest)
        paths.extend([f"{name}/manifest.jsonl", f"{name}/raw_predictions.jsonl"])
    assert_unchanged(args, config)
    frozen = rt.artifact_manifest(args.out, paths)
    rt.write_json(args.out / f"predictions_frozen_{args.stage}.json", {
        "methods": list(method_names), "artifacts_sha256": frozen,
        "config_sha256": data_io.digest(config), "completed_utc": rt.now()})
    print("Predictions frozen. Reading validation B/C only now for scoring; test remains untouched.", flush=True)
    verify_retest_metadata(args.dataset_dir, split["validation"], engine, args.out)
    response = args.dataset_dir / "question_catalog_and_human_response_csv"
    labels_b = engine.inputs.read_selected_labels(response / "wave1_3_response.csv", split["validation"], all_specs)
    labels_c = engine.inputs.read_selected_labels(response / "wave4_response.csv", split["validation"], all_specs)
    for name, (records, manifest) in jobs.items():
        folder = args.out / name
        summary = engine.metrics.score_full(records, manifest, labels_b, labels_c, reference, folder / "scores",
            trained_qids={spec["qid"] for spec in all_specs.values()}, trained_scope="training_B_reference_pool_no_fine_tuning")
        summary.update(protocol=PROTOCOL, method=name, evaluation_split="validation", test_evaluated=False,
                       config_sha256=data_io.digest(config))
        rt.write_json(folder / "summary.json", summary)
    reporting.report(args.out, {name: args.out / name for name in method_names},
                     original_name="condition_baseline")
    if set(LLM_STAGES) <= set(method_names):
        reporting.report(args.out / 'llm_control_comparison',
            {name: args.out / name for name in LLM_STAGES}, original_name='qwen_no_demo')
    scope = ("Qwen embedding retrieval experiment: validation only.\n"
             "Stage 1: frozen Qwen3-Embedding-0.6B, exact-condition training neighbors, B vote/median.\n"
             "Stage 2: original Qwen2.5-0.5B-Instruct, same neighbors as demonstrations; no adapter.\n"
             "Prompt: exact A question definitions are shared once; each participant has separate answer references.\n"
             "No fine-tuning or reranker. k and budgets were fixed before evaluation.\n"
             "Compare Qwen retrieved vs matched no-demo, and embedding kNN vs condition_baseline.\n"
             "The 0.6B retriever is an additional component; this is separate from the <0.5B fine-tuning POC.\n"
             "Metrics are separated by question family. Human B/C consistency is a reference, not a strict ceiling.\n"
             "This is a previously used development set. No reserved test evaluation occurred.\n")
    (args.out / "reporting_scope.txt").write_text(scope, encoding="utf-8")
    rt.verify_files(args.out, frozen)
    rt.write_json(args.out / "evaluation_complete.json", {"protocol": PROTOCOL, "stage": args.stage,
        "methods": list(method_names), "config_sha256": data_io.digest(config), "test_evaluated": False,
        "completed_utc": rt.now()})
    print(scope, flush=True)
    print((args.out / "report.txt").read_text(encoding="utf-8"), flush=True)
    print(f"Results: {collect_results(args, method_names)}", flush=True)


def run(args, engine, split, files, versions):
    config = freeze_run(args, files, versions)
    rt.write_json(args.out / "status.json", {"state": "loading_profiles", "updated_utc": rt.now()})
    print("Loading training/validation A and answer-free targets; reserved test rows are excluded...", flush=True)
    people = data_io.load_people(args.dataset_dir, split, engine)
    all_specs = data_io.train_item_specs(people, split["train"], engine)
    train_labels = engine.inputs.read_selected_labels(
        args.dataset_dir / "question_catalog_and_human_response_csv/wave1_3_response.csv", split["train"], all_specs)
    reference = data_io.fit_reference_baselines(people, split["train"], train_labels, engine)
    rt.freeze(args.out / "training_baselines.json", reference)
    neighbors = save_cpu_predictions(args, engine, people, split, train_labels, config)
    method_names = list(CPU_STAGES)
    if args.stage == "all":
        stages = prepare_llm(args, engine, people, split, train_labels, neighbors)
        del people, train_labels, neighbors
        infer_llm(args, engine, stages)
        method_names.extend(LLM_STAGES)
    score(args, engine, split, all_specs, reference, config, method_names)
    rt.write_json(args.out / "status.json", {"state": "complete", "stage": args.stage, "updated_utc": rt.now()})


def main():
    args = resolve_args(parser().parse_args())
    engine = rt.load_engine()
    split, files, versions = preflight(args, engine)
    if args.check_only:
        print("Preflight passed. No embeddings, inference, training, or evaluation launched.", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another retrieval job already owns {args.out}")
        require((args.out / "run_config.json").exists() or
                all(p.name in {".run.lock", "status.json"} for p in args.out.iterdir()),
                "Output contains unrelated files; choose a new --out directory")
        guard = nullcontext() if args.score_only else rt.resources(args.root, args.wait_for_idle, args.out / "status.json")
        try:
            with guard:
                run(args, engine, split, files, versions)
        except BaseException as error:
            rt.write_json(args.out / "status.json", {"state": "failed", "error": str(error), "updated_utc": rt.now()})
            raise


if __name__ == "__main__":
    main()
