"""Leakage-controlled inputs and native scoring adapters for epoch-1 comparisons."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def import_cpu_engines(directory):
    """No torch, transformers or datasets import is required for CPU baselines."""
    directory = Path(directory).resolve()
    names = {"base": "run_liquid_baseline", "inputs": "full_b_data",
             "schema": "full_b_schema", "metrics": "full_b_metrics", "diag": "diagnose_liquid_outputs"}
    require(all((directory / (name + ".py")).is_file() for name in names.values()),
            f"Native full-B engine files missing in {directory}")
    sys.path.insert(0, str(directory))
    modules = {key: importlib.import_module(name) for key, name in names.items()}
    require(all(Path(module.__file__).resolve().parent == directory for module in modules.values()),
            "An unrelated native module was imported; use a clean Python process")
    return SimpleNamespace(directory=directory, **modules)


def load_people(dataset_dir, split, engine):
    """Load train and validation A plus answer-free B targets, never test or C.

    The parquet scanner selects only these three fields and filters participant
    IDs before materialization. B answer fields are removed immediately and are
    never used for target selection. All B/C labels must be read separately.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    split = {name: set(map(str, split[name])) for name in ("train", "validation", "test")}
    require(all(split.values()), "Empty participant split")
    require(not (split["train"] & split["validation"] or split["train"] & split["test"]
                 or split["validation"] & split["test"]), "Participant splits overlap")
    requested = split["train"] | split["validation"]
    chunks = sorted((Path(dataset_dir) / "wave_split/chunks").glob("*.parquet"))
    require(bool(chunks), "Wave-split parquet files are missing")
    source = ds.dataset([str(path) for path in chunks], format="parquet")
    scanner = source.scanner(columns=["pid", "wave1_3_persona_json", "wave4_Q_wave1_3_A"],
        filter=ds.field("pid").cast(pa.string()).isin(sorted(requested)), batch_size=1)
    people, global_b = {}, set()
    started, last_log = time.monotonic(), time.monotonic()
    diag = getattr(engine, "diag", engine.inputs.diag)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            pid = str(row["pid"])
            require(pid in requested and pid not in people, "Unexpected or duplicate participant")
            # Strip embedded B answers before any question expansion or filtering.
            b_units = engine.base.survey_units(row.pop("wave4_Q_wave1_3_A"), include_answers=False)
            require(not any(diag.has_answers(unit) for _, unit in b_units), "B answers survived stripping")
            a_units = engine.base.survey_units(row.pop("wave1_3_persona_json"), include_answers=True)
            require(len(dict(b_units)) == len(b_units), f"Duplicate B question identifiers for {pid}")
            # A QIDs are reused across sections for different questions. Keep all
            # A units; retrieval must qualify identity by substantive metadata.
            items = engine.schema.expand_units(b_units)
            item_map = {str(item["item_id"]): item for item in items}
            require(len(items) == len(item_map) and not any(diag.has_answers(item) for item in items),
                    f"Duplicate scalar item or target answer leak for {pid}")
            global_b.update(str(qid) for qid, _ in b_units)
            people[pid] = {"a_units": a_units, "items": item_map, "targets": dict(b_units)}
            if len(people) % 100 == 0 or time.monotonic() - last_log >= 10 or len(people) == len(requested):
                print(f"Loaded A/target metadata: {len(people):,}/{len(requested):,} participants "
                      f"| elapsed {time.monotonic()-started:.1f}s", flush=True)
                last_log = time.monotonic()
    require(set(people) == requested, "Train/validation participants are missing from the dataset")
    for pid, person in people.items():
        require(not (set(str(qid) for qid, _ in person["a_units"]) & global_b),
                f"A includes a repeated B target question for participant {pid}")
    return people


def make_manifest(people, pids, engine, method):
    """Label-free scoring cases; hashes describe this algorithm, not LLM prompts."""
    manifest = []
    for pid in sorted(map(str, pids), key=lambda value: (len(value), value)):
        person = people[pid]
        a_hash = digest(person["a_units"])
        for item_id, item in person["items"].items():
            require(not engine.inputs.diag.has_answers(item), "Target schema contains answers")
            spec = dict(item)
            spec.update(pid=pid, input_tokens=0, context_policy=method,
                        prompt_hash=digest({"method": method, "a_sha256": a_hash,
                                            "target": item, "pid": pid}))
            manifest.append(spec)
    require(len({(row["pid"], row["item_id"]) for row in manifest}) == len(manifest),
            "Duplicate participant/item manifest key")
    return manifest


def make_records(outputs, manifest, engine, method):
    require(set(outputs) >= {(spec["pid"], spec["item_id"]) for spec in manifest},
            "Predictions do not cover all selected cases")
    records = []
    for spec in manifest:
        key = (spec["pid"], spec["item_id"])
        detail = dict(outputs[key])
        prediction = detail.pop("prediction")
        record = {name: spec[name] for name in
                  ("pid", "item_id", "qid", "family", "prompt_hash", "input_tokens")}
        record.update(prediction=prediction, valid=bool(engine.metrics._valid(prediction, spec)),
                      raw_output="" if prediction is None else str(prediction),
                      method=method, algorithm_details=detail)
        records.append(record)
    return records


def train_item_specs(people, pids, engine):
    specs = {}
    for pid in sorted(pids):
        for item_id, item in people[pid]["items"].items():
            require(item_id not in specs or engine.metrics._schema_signature(specs[item_id])
                    == engine.metrics._schema_signature(item), f"Inconsistent schema for {item_id}")
            specs[item_id] = item
    return specs


def fit_reference_baselines(people, train_pids, train_labels, engine):
    specs = train_item_specs(people, train_pids, engine)
    grouped = {item_id: [] for item_id in specs}
    for pid in sorted(train_pids):
        for item_id in people[pid]["items"]:
            grouped[item_id].append(train_labels.get((pid, item_id)))
    return engine.metrics.build_baselines(grouped, specs)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)
