"""Frozen local Qwen3-Embedding-0.6B embeddings of A profiles only.

Every A unit is retained. Units are packed into token-bounded chunks; an
oversized unit is explicitly split by token IDs. Each chunk uses last-token
pooling and L2 normalization. A profile is the L2-normalized content-token-
weighted mean of its chunk embeddings. This chunk aggregation is our stated
experiment design, not an official Qwen pooling rule for long profiles.

Training documents have no instruction. Validation queries use QUERY_PREFIX.
No participant identifier, target item, B answer or C answer is embedded.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

PROTOCOL = "qwen3-embedding-profile-chunks-native-specials-v2"
QUERY_PREFIX = (
    "Instruct: Retrieve survey participants with similar preferences, traits, "
    "and earlier responses.\nQuery:"
)
EXPECTED_DIMENSION = 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _loading_info_json(value):
    """Normalize diagnostic containers without weakening checkpoint checks.

    Transformers releases can return sets for loading-info fields. Keep this
    conversion confined to the report, after validating the original values;
    model inputs, cache identities and strict JSON encoding remain unchanged.
    """
    if isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value),
                 "Loading diagnostics must have string dictionary keys")
        return {key: _loading_info_json(child) for key, child in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_loading_info_json(child) for child in value), key=_json)
    if isinstance(value, (tuple, list)):
        return [_loading_info_json(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported loading diagnostic type: {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def render_a_profile(person):
    """Only the caller's explicitly separated A units enter rendered text."""
    from support.prompts import compact_a_unit

    units = person["a_units"]
    _require(bool(units), "Cannot embed an empty A profile")
    texts = []
    for unit in units:
        _require(isinstance(unit, (tuple, list)) and len(unit) == 2,
                 "A units must be (question identifier, unit payload) pairs")
        text = compact_a_unit(unit[1])
        _require(bool(text.strip()), "A unit rendered to empty text")
        texts.append(text)
    return texts


def _special_token_layout(tokenizer):
    """Verify the tokenizer's native single-sequence prefix/suffix template.

    Some official Qwen embedding tokenizers append EOS through their Rust
    postprocessor. Read that behavior via encode rather than hard-coding token
    IDs or using build_inputs_with_special_tokens (absent on some backends).
    Reject a processor that modifies content or has a text-dependent template.
    """
    probes = ("Embedding tokenizer check: alpha 17.", "Second survey response: beta 42!",
              "多语言问卷回答。 café", QUERY_PREFIX + "A survey response.")
    layout = None
    for probe in probes:
        raw = tokenizer.encode(probe, add_special_tokens=False)
        native = tokenizer.encode(probe, add_special_tokens=True)
        _require(isinstance(raw, list) and isinstance(native, list) and raw and
                 all(type(token) is int and token >= 0 for token in raw + native),
                 "Embedding tokenizer must return flat integer IDs")
        positions = [start for start in range(len(native) - len(raw) + 1)
                     if native[start:start + len(raw)] == raw]
        _require(len(positions) == 1,
                 "Embedding tokenizer postprocessing modifies content or repeats it; cannot safely pack chunks")
        start = positions[0]
        candidate = {"prefix_ids": native[:start], "suffix_ids": native[start + len(raw):]}
        if layout is None:
            layout = candidate
        _require(candidate == layout,
                 "Embedding tokenizer has a text-dependent special-token template; cannot safely pack chunks")
    return dict(layout, probe_count=len(probes))


def prepare_profile_texts(tokenizer, texts, *, is_query, max_length=2048, special_token_layout=None):
    """Tokenize complete units and pack deterministically without truncation.

    Tokenization is per whole unit including its leading separator. Concatenating
    those IDs preserves every rendered unit. Oversized units are the only units
    split across chunks, and their counts are reported explicitly.
    """
    _require(isinstance(max_length, int) and max_length >= 64, "max_length must be >= 64")
    _require(bool(texts) and all(isinstance(t, str) and t.strip() for t in texts),
             "Expected nonempty rendered A units")
    layout = special_token_layout if special_token_layout is not None else _special_token_layout(tokenizer)
    special_prefix, special_suffix = list(layout["prefix_ids"]), list(layout["suffix_ids"])
    prefix_ids = tokenizer.encode(QUERY_PREFIX if is_query else "", add_special_tokens=False)
    budget = max_length - len(prefix_ids) - len(special_prefix) - len(special_suffix)
    _require(budget > 0, "Instruction and special tokens consume the whole chunk budget")
    payload = tokenizer(["\n\n" + text for text in texts], add_special_tokens=False,
                        padding=False, truncation=False, return_attention_mask=False)
    encoded_units = payload["input_ids"]
    _require(len(encoded_units) == len(texts), "Tokenizer dropped A units")
    chunks, weights, current = [], [], []
    split_units = split_fragments = total_tokens = 0

    def append_chunk(content_ids):
        ids = special_prefix + list(prefix_ids) + list(content_ids) + special_suffix
        _require(0 < len(ids) <= max_length, "Chunk exceeds declared token budget")
        _require(all(type(x) is int and x >= 0 for x in ids), "Expected flat token IDs")
        chunks.append(ids)
        weights.append(len(content_ids))

    for ids in encoded_units:
        _require(bool(ids), "Tokenizer produced an empty A unit")
        total_tokens += len(ids)
        if len(ids) > budget:
            if current:
                append_chunk(current)
                current = []
            split_units += 1
            for start in range(0, len(ids), budget):
                append_chunk(ids[start:start + budget])
                split_fragments += 1
        else:
            if len(current) + len(ids) > budget:
                append_chunk(current)
                current = []
            current.extend(ids)
    if current:
        append_chunk(current)
    _require(sum(weights) == total_tokens and bool(chunks), "A content was lost during packing")
    return {
        "chunks": chunks,
        "weights": weights,
        "a_content_hash": _digest(texts),
        "a_units": len(texts),
        "content_tokens": total_tokens,
        "chunk_count": len(chunks),
        "oversized_units_split": split_units,
        "oversized_unit_fragments": split_fragments,
        "maximum_input_tokens": max(map(len, chunks)),
        "instruction_tokens": len(prefix_ids),
        "special_prefix_ids": special_prefix,
        "special_suffix_ids": special_suffix,
        "special_tokens_per_chunk": len(special_prefix) + len(special_suffix),
        "truncated_content_tokens": 0,
    }


def prepare_profile(tokenizer, person, *, is_query, max_length=2048):
    return prepare_profile_texts(tokenizer, render_a_profile(person), is_query=is_query,
                                 max_length=max_length)


def last_token_pool(last_hidden_states, attention_mask):
    """Last real token for either left or right padding; fails on empty inputs."""
    import torch

    _require(last_hidden_states.ndim == 3 and attention_mask.ndim == 2,
             "Unexpected hidden-state or attention-mask shape")
    _require(tuple(last_hidden_states.shape[:2]) == tuple(attention_mask.shape),
             "Hidden-state and attention-mask dimensions differ")
    mask = attention_mask.bool()
    _require(bool(mask.any(dim=1).all().item()), "Cannot pool an empty sequence")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    return last_hidden_states[torch.arange(mask.shape[0], device=mask.device), last]


def weighted_profile_vector(chunk_vectors, weights):
    """Normalize chunks, content-token-weight their mean, then normalize it."""
    vectors = np.asarray(chunk_vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _require(vectors.ndim == 2 and len(vectors) > 0 and weights.shape == (len(vectors),),
             "Invalid chunk vectors or weights")
    _require(np.isfinite(vectors).all() and np.isfinite(weights).all() and (weights > 0).all(),
             "Nonfinite vector or invalid token weight")
    norms = np.linalg.norm(vectors, axis=1)
    _require((norms > 1e-12).all(), "Cannot normalize a zero chunk embedding")
    pooled = ((vectors / norms[:, None]) * weights[:, None]).sum(axis=0) / weights.sum()
    norm = np.linalg.norm(pooled)
    _require(np.isfinite(norm) and norm > 1e-12, "Profile embeddings cancelled to zero")
    return (pooled / norm).astype(np.float32)


def _validate_vector(vector, dimension):
    vector = np.asarray(vector)
    _require(vector.dtype == np.float32 and vector.shape == (dimension,),
             "Cached embedding dtype or dimension differs")
    _require(np.isfinite(vector).all(), "Cached embedding contains nonfinite values")
    _require(abs(float(np.linalg.norm(vector)) - 1.0) < 2e-5,
             "Cached embedding is not L2 normalized")
    return vector


def save_profile_cache(path, vector, metadata, *, dimension=EXPECTED_DIMENSION):
    vector = _validate_vector(np.asarray(vector), dimension)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False)
    try:
        with handle:
            np.savez_compressed(handle, vector=vector, metadata=np.array(_json(metadata)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def load_profile_cache(path, expected_metadata, *, dimension=EXPECTED_DIMENSION):
    with np.load(path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"vector", "metadata"}, "Unexpected embedding cache members")
        metadata = json.loads(str(saved["metadata"].item()))
        for key, value in expected_metadata.items():
            _require(metadata.get(key) == value, f"Embedding cache metadata differs: {key}")
        vector = _validate_vector(saved["vector"], dimension).copy()
    return vector, metadata


def _bounded_map(function, records, workers):
    """Ordered thread map with at most 2*workers profiles in flight."""
    if workers == 1:
        for record in records:
            yield function(record)
        return
    iterator = iter(records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(2 * workers):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass


def _tokenizer_fingerprint(model_path):
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model")
    files = {name: _sha_file(model_path / name) for name in names if (model_path / name).is_file()}
    _require("tokenizer.json" in files or "tokenizer.model" in files,
             "Local embedding tokenizer files were not found")
    return files


def _model_fingerprint(model_path):
    names = [p for p in model_path.rglob("*") if p.is_file()
             and not any(part.startswith(".") for part in p.relative_to(model_path).parts)
             and (p.suffix in {".safetensors", ".bin"}
                  or p.name in {"config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"})]
    _require(any(p.suffix in {".safetensors", ".bin"} for p in names),
             "No local model weights found")
    result = {}
    for path in sorted(names):
        print(f"Fingerprinting embedding file: {path.name}", flush=True)
        result[str(path.relative_to(model_path))] = _sha_file(path)
    return result


def build_profile_embeddings(people, train_pids, query_pids, model_path, cache_dir, *,
                             batch_size=8, max_length=2048, workers=4,
                             model_fingerprint=None):
    """Return canonical train/query matrices and a reproducible, resumable report.

    Participant IDs are ordered by ``(len(pid), pid)`` after string conversion;
    the returned ID lists define the row order of the corresponding matrices.
    This also makes embedding batches independent of input set iteration order.

    ``model_fingerprint`` may be a caller-computed JSON-serializable SHA256
    manifest. When absent, all weight/config files are hashed here. Cache identity
    includes source A text hashes, tokenizer/model identity, role, chunk policy,
    numerical runtime and embedding batch size. Completed profiles are written
    atomically, so an interrupted run resumes without recomputing them.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from support import prompts as profile_prompts

    start = time.monotonic()
    _require(type(batch_size) is int and batch_size >= 1, "batch_size must be positive")
    _require(type(workers) is int and workers >= 1, "workers must be positive")
    train_pids = sorted(map(str, train_pids), key=lambda pid: (len(pid), pid))
    query_pids = sorted(map(str, query_pids), key=lambda pid: (len(pid), pid))
    _require(bool(train_pids) and bool(query_pids), "Both train and query participants are required")
    _require(len(set(train_pids)) == len(train_pids) and len(set(query_pids)) == len(query_pids),
             "Duplicate participant identifiers")
    _require(not set(train_pids).intersection(query_pids), "Train/query participant leakage")
    all_pids = train_pids + query_pids
    _require(all(pid in people for pid in all_pids), "Missing requested A profile")
    model_path = Path(model_path).resolve()
    _require(model_path.is_dir(), "Embedding model must be a downloaded local directory")
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    _require(config.model_type == "qwen3" and config.hidden_size == EXPECTED_DIMENSION,
             "Expected Qwen3-Embedding-0.6B: model_type=qwen3 and hidden_size=1024")
    _require(max_length <= config.max_position_embeddings, "Chunk length exceeds model context window")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                              trust_remote_code=False, padding_side="left")
    _require(tokenizer.pad_token_id is not None, "Embedding tokenizer needs its official pad token")
    _require(getattr(tokenizer, "is_fast", False), "Use the downloaded fast tokenizer for parallel preparation")
    special_layout = _special_token_layout(tokenizer)
    print(f"Embedding tokenizer special-token template: prefix={special_layout['prefix_ids']}; "
          f"suffix={special_layout['suffix_ids']} (retained on every chunk).", flush=True)
    fingerprint = model_fingerprint if model_fingerprint is not None else _model_fingerprint(model_path)
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"
    dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if is_cuda else torch.float32)
    if not is_cuda:
        print("Embedding on CPU: CUDA is unavailable; this can be substantially slower.", flush=True)
    try:
        transformer_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformer_version = "unknown"
    configuration = {
        "protocol": PROTOCOL, "model_fingerprint": fingerprint,
        "implementation_sha256": {"embedding_index.py": _sha_file(Path(__file__)),
                                   "support/prompts.py": _sha_file(Path(profile_prompts.__file__))},
        "tokenizer_files_sha256": _tokenizer_fingerprint(model_path),
        "dimension": EXPECTED_DIMENSION, "query_prefix": QUERY_PREFIX,
        "special_token_layout": special_layout,
        "special_token_policy": "native single-sequence prefix/suffix on every chunk; excluded from A pooling weights",
        "document_prefix": "", "max_length": max_length,
        "unit_tokenization": "each compact A unit with leading two newlines; no special tokens",
        "chunk_pooling": "last nonpadding token; float32 L2 normalization",
        "profile_pooling": "content-token-weighted mean of normalized chunks; L2 normalize",
        "oversized_units": "split contiguous content token IDs; never truncate",
        "batch_size": batch_size, "dtype": str(dtype), "device": device,
        "device_name": torch.cuda.get_device_name(0) if is_cuda else "cpu",
        "torch_version": str(torch.__version__), "transformers_version": transformer_version,
        "cuda_version": torch.version.cuda, "attention": "sdpa",
    }
    config_hash = _digest(configuration)
    cache_root = Path(cache_dir) / config_hash
    cache_root.mkdir(parents=True, exist_ok=True)
    config_path = cache_root / "config.json"
    if config_path.exists():
        _require(json.loads(config_path.read_text()) == configuration, "Cache configuration differs")
    else:
        _atomic_json(config_path, configuration)
    train_pid_set = set(train_pids)
    roles = {pid: ("document" if pid in train_pid_set else "query") for pid in all_pids}
    vectors, statistics, records = {}, {}, []
    last_progress = time.monotonic()
    print(f"Checking A-only embedding cache for {len(all_pids):,} profiles...", flush=True)
    for index, pid in enumerate(all_pids, 1):
        texts = render_a_profile(people[pid])
        identity = {"config_hash": config_hash, "pid": pid, "role": roles[pid],
                    "a_content_hash": _digest(texts)}
        path = cache_root / ("profile_" + _digest(identity) + ".npz")
        if path.exists():
            vector, metadata = load_profile_cache(path, identity)
            vectors[pid], statistics[pid] = vector, metadata["statistics"]
        else:
            records.append((pid, texts, identity, path))
        if index % 100 == 0 or time.monotonic() - last_progress >= 10 or index == len(all_pids):
            print(f"A profiles checked: {index:,}/{len(all_pids):,}; cache hits={len(vectors):,}; "
                  f"elapsed={time.monotonic() - start:.1f}s", flush=True)
            last_progress = time.monotonic()
    cache_hits = len(vectors)
    computed_chunks = 0
    loading_report_path = cache_root / "model_loading.json"
    model_loading = json.loads(loading_report_path.read_text()) if loading_report_path.exists() else None
    if records:
        print(f"Loading frozen embedding model on {device}, {dtype}; new profiles={len(records):,}; "
              f"batch={batch_size}; CPU preparation threads={workers}", flush=True)
        model, loading_info = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                                        trust_remote_code=False, dtype=dtype,
                                                        attn_implementation="sdpa",
                                                        output_loading_info=True)
        _require(not loading_info.get("missing_keys") and not loading_info.get("mismatched_keys")
                 and not loading_info.get("error_msgs"),
                 f"Embedding checkpoint did not load completely: {loading_info}")
        unexpected = loading_info.get("unexpected_keys", [])
        _require(set(unexpected).issubset({"lm_head.weight"}),
                 f"Unexpected embedding checkpoint tensors: {unexpected}")
        # AutoModel intentionally omits a causal LM head if one is present in
        # the official checkpoint. The hidden-state backbone must load fully.
        model_loading = {"parameter_count": sum(p.numel() for p in model.parameters()),
                         "loading_checks": _loading_info_json(loading_info),
                         "unused_lm_head_permitted": True,
                         "architecture": type(model).__name__}
        _atomic_json(loading_report_path, model_loading)
        model = model.to(device).eval()
        model.requires_grad_(False)
        state, batch_ids, batch_keys = {}, [], []
        preparation_local = threading.local()

        def prepare(record):
            pid, texts, identity, path = record
            # HF wrappers can mutate backend padding/truncation state. Each
            # worker owns a clone; inference padding uses the original tokenizer.
            if not hasattr(preparation_local, "tokenizer"):
                preparation_local.tokenizer = copy.deepcopy(tokenizer)
            prepared = prepare_profile_texts(preparation_local.tokenizer, texts,
                                              is_query=roles[pid] == "query",
                                              max_length=max_length, special_token_layout=special_layout)
            return pid, identity, path, prepared

        def flush():
            nonlocal computed_chunks, last_progress
            if not batch_ids:
                return
            padded = tokenizer.pad({"input_ids": batch_ids,
                                    "attention_mask": [[1] * len(ids) for ids in batch_ids]},
                                   padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in padded.items()}
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, return_dict=True)
                pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                chunk_vectors = pooled.cpu().numpy()
            _require(chunk_vectors.shape == (len(batch_ids), EXPECTED_DIMENSION),
                     "Embedding output dimension differs from Qwen3-Embedding-0.6B")
            for (pid, chunk_index), vector in zip(batch_keys, chunk_vectors):
                entry = state[pid]
                entry["vectors"][chunk_index] = vector
                entry["remaining"] -= 1
                if entry["remaining"] == 0:
                    profile = weighted_profile_vector(entry["vectors"], entry["weights"])
                    metadata = dict(entry["identity"], statistics=entry["statistics"])
                    save_profile_cache(entry["path"], profile, metadata)
                    vectors[pid], statistics[pid] = profile, entry["statistics"]
                    del state[pid]
            computed_chunks += len(batch_ids)
            batch_ids.clear()
            batch_keys.clear()
            if computed_chunks % 64 < batch_size or time.monotonic() - last_progress >= 10:
                print(f"Embedded profiles: {len(vectors):,}/{len(all_pids):,}; "
                      f"new chunks={computed_chunks:,}; elapsed={time.monotonic() - start:.1f}s", flush=True)
                last_progress = time.monotonic()

        try:
            for pid, identity, path, prepared in _bounded_map(prepare, records, workers):
                stat = {key: value for key, value in prepared.items() if key not in {"chunks", "weights"}}
                state[pid] = {"identity": identity, "path": path, "statistics": stat,
                              "weights": prepared["weights"], "vectors": [None] * prepared["chunk_count"],
                              "remaining": prepared["chunk_count"]}
                for chunk_index, ids in enumerate(prepared["chunks"]):
                    batch_ids.append(ids)
                    batch_keys.append((pid, chunk_index))
                    if len(batch_ids) == batch_size:
                        flush()
            flush()
            _require(not state, "Some profile embeddings were not completed")
        finally:
            del model
            if is_cuda:
                torch.cuda.empty_cache()
    _require(set(vectors) == set(all_pids), "Embedding participant coverage differs")
    _require(model_loading is not None, "Missing model-loading provenance for cached embeddings")
    input_hashes = {pid: stat["a_content_hash"] for pid, stat in statistics.items()}
    report = {
        "configuration": configuration, "config_hash": config_hash,
        "model_loading": model_loading,
        "cache_directory": str(cache_root), "cached_profiles_reused": cache_hits,
        "new_profiles_embedded": len(records), "new_chunks_embedded": computed_chunks,
        "train_participants": len(train_pids), "query_participants": len(query_pids),
        "a_profile_hashes": input_hashes, "all_a_profile_hashes_sha256": _digest(input_hashes),
        "total_chunks": sum(s["chunk_count"] for s in statistics.values()),
        "total_a_content_tokens": sum(s["content_tokens"] for s in statistics.values()),
        "oversized_units_split": sum(s["oversized_units_split"] for s in statistics.values()),
        "oversized_unit_fragments": sum(s["oversized_unit_fragments"] for s in statistics.values()),
        "maximum_input_tokens": max(s["maximum_input_tokens"] for s in statistics.values()),
        "truncated_content_tokens": 0, "elapsed_seconds": time.monotonic() - start,
        "profile_statistics": statistics,
    }
    report_path = cache_root / ("run_" + _digest({"train": train_pids, "query": query_pids,
                                                "source": input_hashes}) + ".json")
    _atomic_json(report_path, report)
    print(f"Embeddings complete: {len(all_pids):,} profiles; cache reused={cache_hits:,}; "
          f"chunks computed={computed_chunks:,}; elapsed={time.monotonic() - start:.1f}s", flush=True)
    return {"train_pids": train_pids, "query_pids": query_pids,
            "train_vectors": np.stack([vectors[pid] for pid in train_pids]).astype(np.float32),
            "query_vectors": np.stack([vectors[pid] for pid in query_pids]).astype(np.float32),
            "report": report}
