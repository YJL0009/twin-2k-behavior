#!/usr/bin/env python3
"""Frozen full-B scoring; no model fitting, prompt changes, or label-derived tuning.

Public contract
---------------
build_baselines(train_labels, specs): train_labels maps item_id to numeric B
values from TRAIN participants only. specs maps item_id to item schema. Returns
item_id -> {mode, median, n, observed_n, invalid_n, missing_n}.

score_full(records, specs, labels_b, labels_c, baselines, out): records and specs
are lists with a unique (pid, item_id) key. A record has prediction (float or
None); a spec has qid, family, options, lower, upper and optional sub_id,
question_type, block_name. Labels are dicts keyed by (str(pid), str(item_id)).
All records must have a matching spec, and every spec must have a record.
The caller must freeze predictions BEFORE supplying held-out labels here.
Optional trained_qids and trained_scope override the original nine-binary-QID
training declaration for later full-B models; omitting them preserves v1 scope.

Families: mc, ordinal, matrix_nominal, matrix_ordinal, bounded_numeric,
unbounded_numeric. First four use 1-based option positions. Nonnegative
unbounded numeric answers have no survey-defined upper limit.

Invalid categorical predictions are wrong; invalid ordinal/bounded predictions
receive normalized error 1. Numeric MAE/MedAE are explicitly valid-only, with a
median baseline on the identical valid mask. A separately named fallback
system substitutes the training median for invalid numeric predictions and is
scored on all valid reference labels. No arbitrary maximum error is invented
for unbounded quantities. Raw errors with different units are never pooled.

None/NaN references are missing. Other malformed/nonfinite/out-of-domain
references are counted invalid and excluded from metrics, never clamped.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "full-b-frozen-item-scoring-v1"
TRAINED_QIDS = frozenset(("QID163", "QID165", "QID167", "QID169", "QID183",
                          "QID184", "QID192", "QID193", "QID196"))
CATEGORICAL = frozenset(("mc", "ordinal", "matrix_nominal", "matrix_ordinal"))
ORDINAL = frozenset(("ordinal", "matrix_ordinal"))
NUMERIC = frozenset(("bounded_numeric", "unbounded_numeric"))
FAMILIES = CATEGORICAL | NUMERIC


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value):
    if isinstance(value, (bool, np.bool_)) or _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _validate_spec(spec):
    family = spec["family"]
    _require(family in FAMILIES, f"Unsupported family: {family}")
    if family in CATEGORICAL:
        _require(isinstance(spec.get("options"), list) and len(spec["options"]) >= 2,
                 f"{spec['item_id']}: categorical options missing")
    if family == "bounded_numeric":
        lo, hi = _number(spec.get("lower")), _number(spec.get("upper"))
        _require(lo is not None and hi is not None and hi > lo,
                 f"{spec['item_id']}: invalid numeric bounds")


def _valid(value, spec):
    number = _number(value)
    if number is None:
        return False
    family = spec["family"]
    if family in CATEGORICAL:
        return number.is_integer() and 1 <= number <= len(spec["options"])
    if family == "bounded_numeric":
        return float(spec["lower"]) <= number <= float(spec["upper"])
    return number >= 0


def _label_state(value, spec):
    if _is_missing(value):
        return "missing"
    return "valid" if _valid(value, spec) else "invalid"


def _schema_signature(spec):
    return json.dumps({key: spec.get(key) for key in
                       ("qid", "family", "options", "lower", "upper", "sub_id")},
                      ensure_ascii=False, sort_keys=True)


def build_baselines(train_labels: dict, specs: dict) -> dict:
    """Fit only from supplied TRAIN-B labels; tie-breaking for mode is lowest."""
    _require(set(train_labels) <= set(specs), "Training labels have unknown item IDs")
    output = {}
    for item_id, spec in specs.items():
        _validate_spec(spec)
        raw = list(train_labels.get(item_id, []))
        states = [_label_state(value, spec) for value in raw]
        values = [_number(value) for value, state in zip(raw, states) if state == "valid"]
        mode = None
        if values:
            counts = Counter(values)
            mode = min(counts, key=lambda value: (-counts[value], value))
        output[str(item_id)] = {
            "mode": mode,
            "median": float(np.median(values)) if values else None,
            "n": len(values),
            "observed_n": sum(state != "missing" for state in states),
            "invalid_n": states.count("invalid"),
            "missing_n": states.count("missing"),
        }
    return output


def _mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _median(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _make_row(record, spec, labels_b, labels_c, baseline, trained_qids=TRAINED_QIDS,
              trained_scope="trained_9_qids"):
    key = (str(spec["pid"]), str(spec["item_id"]))
    family = spec["family"]
    prediction = record.get("prediction")
    pred_valid = _valid(prediction, spec)
    number = _number(prediction)
    mode, median = baseline.get("mode"), baseline.get("median")
    baseline_n = int(baseline.get("n", 0))
    if baseline_n:
        _require(_valid(mode, spec), f"{key}: invalid baseline mode")
        med_number = _number(median)
        _require(med_number is not None, f"{key}: invalid baseline median")
        if family in CATEGORICAL:
            _require(1 <= med_number <= len(spec["options"]), "Ordinal median outside range")
        else:
            _require(_valid(med_number, spec), f"{key}: baseline median outside range")
    else:
        _require(mode is None and median is None, f"{key}: empty baseline has values")
    row = {"pid": key[0], "item_id": key[1], "qid": str(spec["qid"]),
           "sub_id": spec.get("sub_id"), "question_type": spec.get("question_type"),
           "family": family, "block_name": spec.get("block_name"),
           "training_scope": trained_scope if spec["qid"] in trained_qids else "new_qids",
           "prediction": number, "prediction_valid": pred_valid,
           "prediction_state": "valid" if pred_valid else "invalid",
           "raw_output": record.get("raw_output"), "baseline_mode": mode,
           "baseline_median": median, "baseline_training_n": baseline_n}
    for ref, labels in (("B", labels_b), ("C", labels_c)):
        target = labels.get(key)
        state = _label_state(target, spec)
        row[f"{ref}_label_state"] = state
        row[f"{ref}_label"] = _number(target)
        row[f"{ref}_label_raw"] = str(target) if state == "invalid" else None
        for metric in ("correct", "mode_correct", "ordinal_nmae", "median_ordinal_nmae",
                       "bounded_nmae", "median_bounded_nmae", "numeric_absolute_error",
                       "median_numeric_absolute_error", "numeric_fallback_absolute_error"):
            row[f"{ref}_{metric}"] = np.nan
        if state != "valid":
            continue
        target = _number(target)
        if family in CATEGORICAL:
            row[f"{ref}_correct"] = float(pred_valid and number == target)
            if baseline_n:
                row[f"{ref}_mode_correct"] = float(mode == target)
        if family in ORDINAL:
            width = len(spec["options"]) - 1
            row[f"{ref}_ordinal_nmae"] = abs(number - target) / width if pred_valid else 1.0
            if baseline_n:
                row[f"{ref}_median_ordinal_nmae"] = abs(median - target) / width
        if family == "bounded_numeric":
            width = float(spec["upper"]) - float(spec["lower"])
            row[f"{ref}_bounded_nmae"] = abs(number - target) / width if pred_valid else 1.0
            if baseline_n:
                row[f"{ref}_median_bounded_nmae"] = abs(median - target) / width
        if family in NUMERIC:
            if pred_valid:
                row[f"{ref}_numeric_absolute_error"] = abs(number - target)
            if baseline_n:
                row[f"{ref}_median_numeric_absolute_error"] = abs(median - target)
            fallback = number if pred_valid else median
            if fallback is not None:
                row[f"{ref}_numeric_fallback_absolute_error"] = abs(fallback - target)
    paired = row["B_label_state"] == "valid" and row["C_label_state"] == "valid"
    row["paired_B_C"] = paired
    for metric in ("correct", "ordinal_nmae", "bounded_nmae", "numeric_absolute_error"):
        row[f"human_{metric}"] = np.nan
    if paired:
        b, c = row["B_label"], row["C_label"]
        if family in CATEGORICAL:
            row["human_correct"] = float(b == c)
        if family in ORDINAL:
            row["human_ordinal_nmae"] = abs(b - c) / (len(spec["options"]) - 1)
        if family == "bounded_numeric":
            row["human_bounded_nmae"] = abs(b - c) / (float(spec["upper"]) - float(spec["lower"]))
        if family in NUMERIC:
            row["human_numeric_absolute_error"] = abs(b - c)
    return row


SCALAR_METRICS = (
    "categorical_accuracy", "mode_categorical_accuracy", "categorical_gain_pp",
    "ordinal_nmae_invalid_penalty", "median_ordinal_nmae",
    "bounded_nmae_invalid_penalty", "median_bounded_nmae",
    "numeric_mae_valid_only", "numeric_medae_valid_only", "median_numeric_mae_on_model_valid",
    "median_numeric_medae_on_model_valid", "median_numeric_mae_all", "numeric_fallback_mae_all",
    "human_categorical_accuracy", "model_categorical_accuracy_paired", "mode_categorical_accuracy_paired",
    "human_ordinal_nmae", "model_ordinal_nmae_paired", "median_ordinal_nmae_paired",
    "human_bounded_nmae", "model_bounded_nmae_paired", "median_bounded_nmae_paired",
    "human_numeric_mae", "model_numeric_mae_paired_valid", "human_numeric_mae_on_model_valid",
    "median_numeric_mae_paired_valid", "numeric_fallback_mae_paired_all",
)


def _metrics(frame, ref, allow_raw_numeric):
    valid = frame[frame[f"{ref}_label_state"] == "valid"]
    cats = valid[valid["family"].isin(CATEGORICAL)]
    num = valid[valid["family"].isin(NUMERIC)]
    num_valid = num[num["prediction_valid"]]
    pair = frame[frame["paired_B_C"]] if ref == "C" else frame.iloc[:0]
    pair_num_valid = pair[pair["family"].isin(NUMERIC) & pair["prediction_valid"]]
    result = {
        "reference": ref, "prompts": len(frame), "participants": int(frame["pid"].nunique()),
        "questions": int(frame["qid"].nunique()), "items": int(frame["item_id"].nunique()),
        "observed_labels": int((frame[f"{ref}_label_state"] != "missing").sum()),
        "valid_labels": len(valid), "missing_labels": int((frame[f"{ref}_label_state"] == "missing").sum()),
        "invalid_labels": int((frame[f"{ref}_label_state"] == "invalid").sum()),
        "invalid_predictions_all_prompts": int((~frame["prediction_valid"]).sum()),
        "invalid_predictions_on_valid_labels": int((~valid["prediction_valid"]).sum()),
        "prediction_valid_rate_on_valid_labels": _mean(valid["prediction_valid"]),
        "labels_with_baseline": int((valid["baseline_training_n"] > 0).sum()),
        "categorical_examples": len(cats), "numeric_examples": len(num),
        "numeric_valid_predictions": len(num_valid),
        "numeric_valid_rate": _mean(num["prediction_valid"]),
        "paired_B_C": len(pair), "raw_numeric_errors_aggregated": bool(allow_raw_numeric),
    }
    mappings = {
        "categorical_accuracy": (valid, f"{ref}_correct"),
        "mode_categorical_accuracy": (valid, f"{ref}_mode_correct"),
        "ordinal_nmae_invalid_penalty": (valid, f"{ref}_ordinal_nmae"),
        "median_ordinal_nmae": (valid, f"{ref}_median_ordinal_nmae"),
        "bounded_nmae_invalid_penalty": (valid, f"{ref}_bounded_nmae"),
        "median_bounded_nmae": (valid, f"{ref}_median_bounded_nmae"),
        "human_categorical_accuracy": (pair, "human_correct"),
        "model_categorical_accuracy_paired": (pair, "C_correct"),
        "mode_categorical_accuracy_paired": (pair, "C_mode_correct"),
        "human_ordinal_nmae": (pair, "human_ordinal_nmae"),
        "model_ordinal_nmae_paired": (pair, "C_ordinal_nmae"),
        "median_ordinal_nmae_paired": (pair, "C_median_ordinal_nmae"),
        "human_bounded_nmae": (pair, "human_bounded_nmae"),
        "model_bounded_nmae_paired": (pair, "C_bounded_nmae"),
        "median_bounded_nmae_paired": (pair, "C_median_bounded_nmae"),
    }
    if allow_raw_numeric:
        mappings.update({
            "numeric_mae_valid_only": (num_valid, f"{ref}_numeric_absolute_error"),
            "median_numeric_mae_on_model_valid": (num_valid, f"{ref}_median_numeric_absolute_error"),
            "median_numeric_mae_all": (num, f"{ref}_median_numeric_absolute_error"),
            "numeric_fallback_mae_all": (num, f"{ref}_numeric_fallback_absolute_error"),
            "human_numeric_mae": (pair, "human_numeric_absolute_error"),
            "model_numeric_mae_paired_valid": (pair_num_valid, "C_numeric_absolute_error"),
            "human_numeric_mae_on_model_valid": (pair_num_valid, "human_numeric_absolute_error"),
            "median_numeric_mae_paired_valid": (pair_num_valid, "C_median_numeric_absolute_error"),
            "numeric_fallback_mae_paired_all": (pair, "C_numeric_fallback_absolute_error"),
        })
    for name in SCALAR_METRICS:
        result[name] = None
    for name, (subset, column) in mappings.items():
        result[name] = _mean(subset[column])
    accuracy, baseline_accuracy = result["categorical_accuracy"], result["mode_categorical_accuracy"]
    # A gain is comparable only if the baseline covers every categorical label.
    if accuracy is not None and baseline_accuracy is not None and cats[f"{ref}_mode_correct"].notna().all():
        result["categorical_gain_pp"] = 100 * (accuracy - baseline_accuracy)
    if allow_raw_numeric:
        result["numeric_medae_valid_only"] = _median(num_valid[f"{ref}_numeric_absolute_error"])
        result["median_numeric_medae_on_model_valid"] = _median(num_valid[f"{ref}_median_numeric_absolute_error"])
    return result


def _aggregate(frame, group_keys, question_macro=False):
    output = []
    groups = frame.groupby(group_keys, sort=True, dropna=False) if group_keys else [((), frame)]
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_keys, key))
        for ref in ("C", "B"):
            allow_raw = group["qid"].nunique() == 1
            result = {**identity, **_metrics(group, ref, allow_raw)}
            if question_macro:
                per_question = [_metrics(qgroup, ref, True) for _, qgroup in group.groupby("qid")]
                for metric in SCALAR_METRICS:
                    # Different units must never be averaged into a raw numeric score.
                    if "numeric_" in metric and not allow_raw:
                        continue
                    values = [q[metric] for q in per_question if q[metric] is not None]
                    result[f"question_macro_{metric}"] = float(np.mean(values)) if values else None
                    result[f"question_macro_{metric}_questions"] = len(values)
            output.append(result)
    return pd.DataFrame(output)


def _matrix_exact(frame, specs):
    expected = {}
    for spec in specs:
        if spec["family"] in {"matrix_nominal", "matrix_ordinal"}:
            expected.setdefault(spec["qid"], set()).add(str(spec["item_id"]))
    result = []
    for qid, item_ids in expected.items():
        qframe = frame[frame["qid"] == qid]
        for ref in ("C", "B"):
            n, correct, baseline_n, baseline_correct = 0, 0, 0, 0
            excluded, human_n, human_correct, model_paired, mode_paired = 0, 0, 0, 0, 0
            mode_paired_n = 0
            for _, person in qframe.groupby("pid"):
                complete = set(person["item_id"]) == item_ids
                complete = complete and (person[f"{ref}_label_state"] == "valid").all()
                if not complete:
                    excluded += 1
                    continue
                n += 1
                correct += int((person[f"{ref}_correct"] == 1).all())
                if person[f"{ref}_mode_correct"].notna().all():
                    baseline_n += 1
                    baseline_correct += int((person[f"{ref}_mode_correct"] == 1).all())
                if ref == "C" and person["paired_B_C"].all():
                    human_n += 1
                    human_correct += int((person["human_correct"] == 1).all())
                    model_paired += int((person["C_correct"] == 1).all())
                    if person["C_mode_correct"].notna().all():
                        mode_paired_n += 1
                        mode_paired += int((person["C_mode_correct"] == 1).all())
            result.append({"qid": qid, "reference": ref, "expected_items": len(item_ids),
                           "complete_reference_vectors": n, "excluded_incomplete_reference_vectors": excluded,
                           "vector_exact_accuracy": correct / n if n else None,
                           "baseline_complete_vectors": baseline_n,
                           "mode_vector_exact_accuracy": baseline_correct / baseline_n if baseline_n else None,
                           "human_complete_pairs": human_n,
                           "human_vector_exact_accuracy": human_correct / human_n if human_n else None,
                           "model_vector_exact_accuracy_paired": model_paired / human_n if human_n else None,
                           "mode_paired_vectors": mode_paired_n,
                           "mode_vector_exact_accuracy_paired": mode_paired / mode_paired_n if mode_paired_n else None})
    return pd.DataFrame(result)


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def score_full(records: list, specs: list, labels_b: dict, labels_c: dict,
               baselines: dict, out: Path, trained_qids=None, trained_scope="trained_full_b") -> dict:
    """Write scored rows and separate metrics; return a JSON-serializable summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    legacy_scope = trained_qids is None
    trained_qids = TRAINED_QIDS if legacy_scope else frozenset(map(str, trained_qids))
    trained_scope = "trained_9_qids" if legacy_scope else str(trained_scope)
    _require(trained_qids and trained_scope, "Training coverage declaration must not be empty")
    _require(len(specs) > 0, "No evaluation specs")
    rec_map = {(str(r["pid"]), str(r["item_id"])): r for r in records}
    spec_map = {(str(s["pid"]), str(s["item_id"])): s for s in specs}
    _require(len(rec_map) == len(records), "Duplicate prediction keys")
    _require(len(spec_map) == len(specs), "Duplicate evaluation specs")
    _require(set(rec_map) == set(spec_map), "Prediction/spec coverage mismatch")
    signatures = {}
    rows = []
    for key, spec in spec_map.items():
        _validate_spec(spec)
        item = str(spec["item_id"])
        signature = _schema_signature(spec)
        _require(item not in signatures or signatures[item] == signature,
                 f"{item}: scoring schema differs between participants")
        signatures[item] = signature
        _require(item in baselines, f"Missing baseline declaration: {item}")
        _require(int(baselines[item].get("n", 0)) > 0,
                 f"{item}: no valid training-B labels; cannot compare a frozen baseline or define median fallback")
        rows.append(_make_row(rec_map[key], spec, labels_b, labels_c, baselines[item], trained_qids, trained_scope))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "scored_predictions.csv", index=False)
    by_item = _aggregate(frame, ["item_id", "qid", "family", "training_scope"])
    by_question = _aggregate(frame, ["qid", "family", "training_scope"])
    by_family = _aggregate(frame, ["family"], question_macro=True)
    by_scope = _aggregate(frame, ["training_scope", "family"], question_macro=True)
    matrices = _matrix_exact(frame, specs)
    for name, data in (("metrics_by_item", by_item), ("metrics_by_question", by_question),
                       ("metrics_by_family", by_family), ("metrics_by_training_scope", by_scope),
                       ("matrix_exact", matrices)):
        data.to_csv(out / f"{name}.csv", index=False)
    human_columns = [col for col in by_question.columns if col.startswith("human_")
                     or col.endswith("_paired") or col in ("qid", "family", "training_scope", "paired_B_C")]
    by_question.loc[by_question["reference"] == "C", human_columns].to_csv(
        out / "human_reference_by_question.csv", index=False)
    overall = {ref: _metrics(frame, ref, False) for ref in ("C", "B")}
    # Overall categorical accuracy is a category-only diagnostic, never a full-B mixed score.
    for ref in overall:
        qcat = by_question[(by_question["reference"] == ref) & by_question["family"].isin(CATEGORICAL)]
        overall[ref]["categorical_question_macro_accuracy"] = _mean(qcat["categorical_accuracy"])
        overall[ref]["mode_categorical_question_macro_accuracy"] = _mean(qcat["mode_categorical_accuracy"])
    summary = _clean_json({
        "protocol": PROTOCOL, "primary_reference": "C", "secondary_reference": "B",
        "prompts": len(frame), "participants": int(frame["pid"].nunique()),
        "question_ids": int(frame["qid"].nunique()), "item_ids": int(frame["item_id"].nunique()),
        "trained_qids": sorted(trained_qids), "C": overall["C"], "B": overall["B"],
        "family_metrics": by_family.to_dict(orient="records"),
        "training_scope_metrics": by_scope.to_dict(orient="records"),
        "notes": [
            "No combined accuracy/error score for full B; interpret families separately.",
            "Categorical invalid answers are wrong. Ordinal/bounded invalid answers receive normalized error 1.",
            "Numeric MAE/MedAE valid-only are conditional on valid predictions; compare their matched-mask baseline and coverage.",
            "Numeric fallback MAE evaluates a separate system substituting each training-B median for invalid predictions.",
            "Raw numeric errors are provided per item/question only, never pooled across different question units.",
            "Missing and observed invalid reference labels are counted separately and excluded, never clamped.",
            "Family question-macro metrics first average eligible participant/item cases within each QID.",
            "Matrix vector exact match requires all declared rows and valid reference labels; invalid model rows fail.",
            "Human reference uses valid B/C pairs; it is a reference, not a hard performance ceiling.",
            ("The frozen checkpoint was trained on nine binary QIDs; other question IDs are expanded task coverage."
             if legacy_scope else f"Training coverage is explicitly declared as {trained_scope}; untrained QIDs remain separate."),
        ],
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary
