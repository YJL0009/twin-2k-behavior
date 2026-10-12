"""Offline I/O, streamed inputs and resumable native scalar generation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

from support import data_io

require = data_io.require
write_json = data_io.write_json
write_jsonl = data_io.write_jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(path, config):
    path = Path(path)
    if path.exists():
        require(read_json(path) == config,
                f"Configuration changed: {path}. Use a fresh --out; cached embeddings can still be reused.")
    else:
        write_json(path, config)


def load_engine():
    folder = Path(__file__).resolve().parent / "native_engine"
    engine = data_io.import_cpu_engines(folder)
    engine.data = importlib.import_module("full_b_train_data")
    engine.generation = importlib.import_module("run_qwen_full_b_test")
    return engine


def model_files(folder):
    folder = Path(folder).resolve()
    require(folder.is_dir() and (folder / "config.json").is_file(), f"Model directory missing: {folder}")
    require(not (folder / "adapter_config.json").exists(), f"Use the original base model, not an adapter: {folder}")
    patterns = ("*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.jinja", "*.tiktoken")
    files = sorted({p for pattern in patterns for p in folder.glob(pattern) if p.is_file()})
    require(any(p.suffix in {".safetensors", ".bin"} for p in files), f"Model weights missing: {folder}")
    require((folder / "tokenizer_config.json").is_file() and
            any((folder / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")),
            f"Tokenizer incomplete: {folder}")
    return {p.name: sha256(p) for p in files}


def verify_files(folder, hashes):
    for name, expected in hashes.items():
        path = (Path(folder) / name).resolve()
        require(path.is_file() and sha256(path) == expected, f"Frozen file changed: {path}")


def artifact_manifest(folder, names):
    return {str(name): sha256(Path(folder) / name) for name in names}


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return sorted({int(value.strip()) for value in result.stdout.splitlines()
                   if value.strip().isdigit() and int(value.strip()) != os.getpid()})


@contextmanager
def resources(root, wait, status_path):
    """Honor the existing comparison/training locks before touching CUDA."""
    root = Path(root)
    old = root / "outputs/qwen_full_b_lora_v4_b16_a2"
    paths = [p for p in (old / ".pipeline.lock", old / ".training.lock") if p.exists()]
    own = root / "outputs/.qwen_embedding_retrieval_gpu.lock"
    own.parent.mkdir(parents=True, exist_ok=True)
    paths.append(own)
    while True:
        handles, busy = [], None
        try:
            for path in paths:
                handle = path.open("a+")
                handles.append(handle)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    busy = f"existing job lock {path}"
                    break
            if not busy:
                processes = gpu_processes()
                if processes:
                    busy = "GPU processes " + ", ".join(map(str, processes))
            if not busy:
                break
        except BaseException:
            for handle in handles:
                handle.close()
            raise
        for handle in handles:
            handle.close()
        require(wait, f"Waiting is required: {busy}. Use --wait-for-idle; no job was stopped.")
        print(f"Waiting for {busy}; rechecking in 30 seconds.", flush=True)
        write_json(status_path, {"state": "waiting", "reason": busy, "updated_utc": now()})
        time.sleep(30)
    try:
        yield
    finally:
        for handle in handles:
            handle.close()


class PreparedWriter:
    def __init__(self, folder, engine):
        import pyarrow as pa
        self.folder, self.engine, self.pa = Path(folder), engine, pa
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "inputs.arrow"
        self.tmp = self.folder / "inputs.arrow.tmp"
        self.schema = pa.schema([("input_ids", pa.list_(pa.int32())),
                                 ("attention_mask", pa.list_(pa.int8()))])
        self.sink = pa.OSFile(str(self.tmp), "wb")
        self.writer = pa.ipc.new_stream(self.sink, self.schema)
        self.manifest, self.buffer = [], []

    def add(self, record, encoded):
        self.engine.data.validate_prefix(encoded, record, max_input_tokens=32760)
        self.manifest.append(record)
        self.buffer.append(encoded)
        if len(self.buffer) >= 64:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(self.pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def finish(self):
        self.flush()
        self.writer.close()
        self.sink.close()
        self.tmp.replace(self.path)
        write_jsonl(self.folder / "manifest.jsonl", self.manifest)
        write_json(self.folder / "prepared_complete.json", {
            "rows": len(self.manifest),
            "artifacts_sha256": artifact_manifest(self.folder, ["inputs.arrow", "manifest.jsonl"])})


class ArrowRows:
    def __init__(self, path):
        import pyarrow as pa
        self.source = pa.memory_map(str(path), "r")
        self.table = pa.ipc.open_stream(self.source).read_all()
        require(set(self.table.column_names) == {"input_ids", "attention_mask"}, "Unexpected prepared columns")

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, index):
        return {key: self.table[key][index].as_py() for key in self.table.column_names}


def prepared(folder):
    folder = Path(folder)
    info = read_json(folder / "prepared_complete.json")
    verify_files(folder, info["artifacts_sha256"])
    rows = ArrowRows(folder / "inputs.arrow")
    with (folder / "manifest.jsonl").open(encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle]
    require(len(rows) == len(manifest) == info["rows"], "Prepared case coverage differs")
    return rows, manifest


class RowView:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def unique_inputs(dataset, manifest, engine):
    first, unique, mapping = {}, [], []
    for index, spec in enumerate(manifest):
        key = (spec["prompt_hash"], engine.data.response_cap(spec))
        if key not in first:
            first[key] = index
            unique.append(index)
        original = first[key]
        if original != index:
            require(dataset[index] == dataset[original], "Hash-equal prompts have different token IDs")
        mapping.append(original)
    # Grouping by output cap also occurs in the native inference engine.
    unique.sort(key=lambda i: (engine.data.response_cap(manifest[i]), manifest[i]["input_tokens"], i))
    return unique, mapping


def generate_stage(folder, dataset, manifest, engine, model, tokenizer, batch_size):
    folder = Path(folder)
    raw = folder / "raw_predictions.jsonl"
    if (folder / "generation_complete.json").exists():
        done = read_json(folder / "generation_complete.json")
        require(sha256(raw) == done["raw_sha256"], "Completed predictions were changed")
        records = engine.base.read_jsonl(raw)
        require(len(engine.generation.validate_records(records, manifest)) == len(manifest), "Stage incomplete")
        return records
    indices, mapping = unique_inputs(dataset, manifest, engine)
    unique_manifest = [manifest[i] for i in indices]
    print(f"Generating {folder.name}: {len(manifest):,} cases, {len(indices):,} unique prompts, batch {batch_size}.", flush=True)
    rows = engine.generation.infer(model, tokenizer, RowView(dataset, indices), unique_manifest,
        folder / "unique_predictions.jsonl", batch_size=batch_size, status_path=folder / "status.json")
    by_key = {(str(row["pid"]), row["item_id"]): row for row in rows}
    expanded = []
    for spec, original_index in zip(manifest, mapping):
        source = manifest[original_index]
        row = dict(by_key[(str(source["pid"]), source["item_id"])])
        row.update({key: spec[key] for key in ("pid", "item_id", "qid", "family", "prompt_hash", "input_tokens")})
        expanded.append(row)
    require(len(engine.generation.validate_records(expanded, manifest)) == len(manifest), "Incomplete generation")
    write_jsonl(raw, expanded)
    write_json(folder / "generation_complete.json", {"rows": len(expanded), "unique_prompts": len(indices),
               "raw_sha256": sha256(raw), "batch_size": batch_size, "completed_utc": now()})
    return expanded


def load_generator(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side="left")
    require(tokenizer.eos_token_id is not None, "Generator tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, loading = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
        dtype=dtype, attn_implementation="sdpa", output_loading_info=True)
    require(not any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            f"Generator loading mismatch: {loading}")
    count = sum(p.numel() for p in model.parameters())
    require(count == 494_032_768, f"Expected original Qwen2.5-0.5B-Instruct (494,032,768 parameters), got {count:,}")
    model.requires_grad_(False)
    return model.to("cuda").eval(), tokenizer, {"parameters": count, "dtype": str(dtype), "adapter": None}


def release_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
