#!/usr/bin/env python3
"""Compare frozen native full-B scores without mixing question units.

The caller must finish generation and native scoring before invoking this
module. This module never reads source answer CSVs, fits a method, selects a
checkpoint, or changes predictions. Stages must contain the identical original
validation keys, or exactly its complete pricing subset. A price-only method is
therefore never silently compared with an all-B aggregate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PRICING_QIDS = frozenset(f"QID9_{i}" for i in range(1, 41))
OLD_NINE_QIDS = frozenset(("QID163", "QID165", "QID167", "QID169", "QID183",
                           "QID184", "QID192", "QID193", "QID196"))
CATEGORICAL = frozenset(("mc", "ordinal", "matrix_nominal", "matrix_ordinal"))
# native metric, baseline metric, scalar-row suffix, baseline suffix, direction
PRIMARY = {
    "mc": ("categorical_accuracy", "mode_categorical_accuracy", "correct", "mode_correct", "higher"),
    "matrix_nominal": ("categorical_accuracy", "mode_categorical_accuracy", "correct", "mode_correct", "higher"),
    "ordinal": ("ordinal_nmae_invalid_penalty", "median_ordinal_nmae", "ordinal_nmae", "median_ordinal_nmae", "lower"),
    "matrix_ordinal": ("ordinal_nmae_invalid_penalty", "median_ordinal_nmae", "ordinal_nmae", "median_ordinal_nmae", "lower"),
    "bounded_numeric": ("bounded_nmae_invalid_penalty", "median_bounded_nmae", "bounded_nmae", "median_bounded_nmae", "lower"),
    "unbounded_numeric": ("numeric_mae_valid_only", "median_numeric_mae_on_model_valid", "numeric_absolute_error", "median_numeric_absolute_error", "lower"),
}
HUMAN = {"mc": "human_categorical_accuracy", "matrix_nominal": "human_categorical_accuracy",
         "ordinal": "human_ordinal_nmae", "matrix_ordinal": "human_ordinal_nmae",
         "bounded_numeric": "human_bounded_nmae", "unbounded_numeric": "human_numeric_mae"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _bool(series, column):
    mapping = {True: True, False: False, "True": True, "False": False,
               "true": True, "false": False, 1: True, 0: False}
    result = series.map(mapping)
    _require(result.notna().all(), f"Malformed boolean values: {column}")
    return result.astype(bool)


def _score_folder(path):
    path = Path(path).resolve()
    if (path / "scored_predictions.csv").is_file():
        return path
    _require((path / "scores/scored_predictions.csv").is_file(),
             f"Complete native scores missing: {path}")
    return path / "scores"


def _load(path):
    path = _score_folder(path)
    rows = pd.read_csv(path / "scored_predictions.csv", dtype={"pid": str, "item_id": str, "qid": str})
    required = {"pid", "item_id", "qid", "family", "prediction", "prediction_valid",
                "baseline_mode", "baseline_median", "baseline_training_n", "paired_B_C",
                "B_label", "C_label", "B_label_state", "C_label_state"}
    _require(required <= set(rows.columns), f"Native scored-prediction columns missing: {path}")
    _require(len(rows) > 0 and not rows.duplicated(["pid", "item_id"]).any(),
             f"Empty or duplicate scored keys: {path}")
    _require(rows[["pid", "item_id", "qid", "family"]].notna().all().all(), "Missing scored identity")
    _require(set(rows.family) <= set(PRIMARY), "Unsupported scored question family")
    rows["prediction_valid"] = _bool(rows.prediction_valid, "prediction_valid")
    rows["paired_B_C"] = _bool(rows.paired_B_C, "paired_B_C")
    rows = rows.set_index(["pid", "item_id"], drop=False).sort_index()
    questions = pd.read_csv(path / "metrics_by_question.csv", dtype={"qid": str})
    families = pd.read_csv(path / "metrics_by_family.csv")
    _require(not questions.duplicated(["qid", "reference"]).any(), "Duplicate native question metrics")
    _require(not families.duplicated(["family", "reference"]).any(), "Duplicate native family metrics")
    _require(set(questions.reference) == set(families.reference) == {"B", "C"},
             "Both primary C and secondary B native metrics are required")
    _require(set(questions.qid) == set(rows.qid) and set(families.family) == set(rows.family),
             "Native metric tables do not cover the scored rows")
    for ref in ("C", "B"):
        _require(set(questions[questions.reference == ref].qid) == set(rows.qid)
                 and set(families[families.reference == ref].family) == set(rows.family),
                 f"Native {ref} metrics have incomplete question/family coverage")
    return {"path": path, "rows": rows, "questions": questions, "families": families}


def _identical_values(left, right):
    return bool(((left == right) | (left.isna() & right.isna())).all())


def _validate_pair(stage, original, name):
    rows, base = stage["rows"], original["rows"]
    keys, original_keys = set(rows.index), set(base.index)
    pricing_keys = set(base[base.qid.isin(PRICING_QIDS)].index)
    _require(keys == original_keys or (bool(pricing_keys) and keys == pricing_keys),
             f"{name}: stage must match all original keys or the complete pricing subset")
    aligned = base.loc[rows.index]
    for column in ("qid", "family", "B_label", "C_label", "B_label_state", "C_label_state",
                   "baseline_mode", "baseline_median", "baseline_training_n"):
        _require(_identical_values(rows[column], aligned[column]),
                 f"{name}: mismatched original labels, schemas or frozen baseline in {column}")
    return "all_original_validation_cases" if keys == original_keys else "pricing_40_only"


def _macro(frame, column):
    return _mean(frame.groupby("qid")[column].mean())


def _gain(model, comparison, direction):
    if model is None or comparison is None:
        return None
    return (model - comparison) if direction == "higher" else (comparison - model)


def _metric_row(name, scope, group_name, family, ref, frame, qmetrics, original):
    metric, baseline_metric, suffix, baseline_suffix, direction = PRIMARY[family]
    _require(not (family == "unbounded_numeric" and frame.qid.nunique() != 1),
             "Raw numeric units may only be compared within one question")
    chosen = qmetrics[(qmetrics.reference == ref) & qmetrics.qid.isin(frame.qid)]
    _require(set(chosen.qid) == set(frame.qid), f"{name}/{group_name}/{ref}: missing native question metric")
    score, baseline = _mean(chosen[metric]), _mean(chosen[baseline_metric])
    valid = frame[frame[f"{ref}_label_state"] == "valid"]
    scalar_col, baseline_col = f"{ref}_{suffix}", f"{ref}_{baseline_suffix}"
    score_rows = valid[valid.prediction_valid] if family == "unbounded_numeric" else valid
    # A stale native table must not generate a plausible-looking comparison.
    computed = _macro(score_rows, scalar_col)
    _require((score is None and computed is None) or
             (score is not None and computed is not None and abs(score - computed) < 1e-10),
             f"{name}/{group_name}/{ref}: native metric table differs from scored predictions")
    baseline_computed = _macro(score_rows, baseline_col)
    _require((baseline is None and baseline_computed is None) or
             (baseline is not None and baseline_computed is not None
              and abs(baseline - baseline_computed) < 1e-10),
             f"{name}/{group_name}/{ref}: native baseline table differs from scored predictions")
    if family != "unbounded_numeric":
        matched = score_rows
    else:
        # Each model's published valid-only MAE can use a different mask. Compare
        # this stage with original on their common valid prediction mask only.
        matched = score_rows[original.loc[score_rows.index, "prediction_valid"]]
    original_matched = original.loc[matched.index]
    model_matched = _macro(matched, scalar_col)
    original_score = _macro(original_matched, scalar_col)
    gain_baseline = _gain(score, baseline, direction)
    gain_original = _gain(model_matched, original_score, direction)
    return {
        "stage": name, "stage_scope": scope, "reference": ref,
        "reference_role": "primary" if ref == "C" else "secondary",
        "group": group_name, "family": family,
        "aggregation": "single_question" if frame.qid.nunique() == 1 else "question_macro",
        "metric": metric, "better": direction,
        "prompts": len(frame), "participants": frame.pid.nunique(), "questions": frame.qid.nunique(),
        "valid_labels": len(valid), "valid_predictions": int(frame.prediction_valid.sum()),
        "invalid_predictions": int((~frame.prediction_valid).sum()),
        "prediction_valid_rate": float(frame.prediction_valid.mean()),
        "valid_predictions_on_valid_labels": int(valid.prediction_valid.sum()),
        "score": score, "training_B_baseline": baseline,
        "gain_over_training_B_baseline": gain_baseline,
        "gain_over_training_B_baseline_pp": None if direction == "lower" or gain_baseline is None else 100 * gain_baseline,
        "original_comparison_cases": len(matched),
        "stage_score_on_original_comparison_mask": model_matched,
        "original_score_on_comparison_mask": original_score,
        "gain_over_original_on_comparison_mask": gain_original,
        "gain_over_original_pp": None if direction == "lower" or gain_original is None else 100 * gain_original,
        "human_B_C_reference_paired": _mean(chosen[HUMAN[family]]) if ref == "C" else None,
        "human_B_C_pairs": int(frame.paired_B_C.sum()) if ref == "C" else 0,
    }


def _agreement(name, scope, frame, original, group):
    aligned = original.loc[frame.index]
    valid_pair = frame.prediction_valid & aligned.prediction_valid
    same = valid_pair & (frame.prediction == aligned.prediction)
    both_invalid = ~frame.prediction_valid & ~aligned.prediction_valid
    return {"stage": name, "stage_scope": scope, "group": group, "cases": len(frame),
            "both_predictions_valid": int(valid_pair.sum()),
            "same_valid_predictions": int(same.sum()),
            "same_valid_prediction_rate_all_cases": float(same.mean()) if len(frame) else None,
            "agreement_among_both_valid": float(same.sum() / valid_pair.sum()) if valid_pair.any() else None,
            "both_invalid": int(both_invalid.sum()),
            "changed_prediction_or_validity": int((~(same | both_invalid)).sum()),
            "note": "Exact parsed-answer agreement; two invalid outputs are not counted as an agreed valid answer."}


def report(rootout, stages: dict, original_name="adapter_original", *, stage_metadata=None):
    """Write comparison CSVs and a concise text report; return their paths.

    ``stages`` maps method name to native score directory (or its parent).
    ``stage_metadata`` optionally supplies retrieval-coverage dictionaries; those
    are copied verbatim to a separate JSON and are not confused with accuracy.
    Stage methods are fixed by caller: this function selects no winner or model.
    """
    _require(original_name in stages, f"Original stage missing: {original_name}")
    loaded = {str(name): _load(path) for name, path in stages.items()}
    original = loaded[original_name]
    comparisons, agreements, coverage = [], [], []
    for name, stage in loaded.items():
        scope = _validate_pair(stage, original, name)
        rows, questions = stage["rows"], stage["questions"]
        coverage.append({"stage": name, "stage_scope": scope, "score_directory": str(stage["path"]),
                         "cases": len(rows), "participants": rows.pid.nunique(),
                         "question_ids": rows.qid.nunique(), "scalar_items": rows.item_id.nunique(),
                         "valid_predictions": int(rows.prediction_valid.sum()),
                         "invalid_predictions": int((~rows.prediction_valid).sum()),
                         "prediction_valid_rate": float(rows.prediction_valid.mean()),
                         "same_case_keys_as_original": set(rows.index) == set(original["rows"].index)})
        groups = []
        for family, subset in rows.groupby("family", sort=True):
            if family != "unbounded_numeric":
                groups.append(("family:" + family, family, subset))
        for label, ids in (("pricing_40", PRICING_QIDS), ("original_9_binary", OLD_NINE_QIDS)):
            subset = rows[rows.qid.isin(ids)]
            if len(subset):
                _require(set(subset.family) == {"mc"}, f"{label} must contain only multiple-choice items")
                groups.append((label, "mc", subset))
        for qid, subset in rows.groupby("qid", sort=True):
            _require(subset.family.nunique() == 1, "A question has multiple scoring families")
            groups.append(("qid:" + qid, str(subset.family.iloc[0]), subset))
        for ref in ("C", "B"):
            for group, family, subset in groups:
                comparisons.append(_metric_row(name, scope, group, family, ref, subset,
                                                questions, original["rows"]))
        agreement_groups = [("all_stage_cases", rows),
                            ("categorical_cases", rows[rows.family.isin(CATEGORICAL)]),
                            ("pricing_40", rows[rows.qid.isin(PRICING_QIDS)]),
                            ("original_9_binary", rows[rows.qid.isin(OLD_NINE_QIDS)])]
        for group, subset in agreement_groups:
            if len(subset):
                agreements.append(_agreement(name, scope, subset, original["rows"], group))
    out = Path(rootout)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"comparison": out / "comparison.csv", "agreement": out / "prediction_agreement.csv",
             "coverage": out / "comparison_coverage.csv", "report": out / "report.txt"}
    comparison = pd.DataFrame(comparisons)
    agreement = pd.DataFrame(agreements)
    coverage = pd.DataFrame(coverage)
    comparison.to_csv(paths["comparison"], index=False)
    agreement.to_csv(paths["agreement"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    lines = ["Fixed epoch-1 validation method comparison", "",
             "C is primary; B is secondary. All methods use the retained epoch-1 adapter where an adapter is required.",
             "Scores below retain each question family's declared metric. Accuracy is higher-better; errors are lower-better.",
             "Positive gains mean improvement. Accuracy gains in _pp columns are percentage points; error gains retain their error units.",
             "No combined full-B accuracy/error score is constructed.", "", "Coverage:",
             coverage.drop(columns="score_directory").to_string(index=False), ""]
    display = ["stage", "stage_scope", "group", "metric", "better", "questions", "score",
               "training_B_baseline", "gain_over_training_B_baseline",
               "gain_over_original_on_comparison_mask", "invalid_predictions"]
    summary_mask = ~comparison.group.str.startswith("qid:")
    for ref, title in (("C", "Primary comparison against C"), ("B", "Secondary comparison against B")):
        lines.extend([title + ":", comparison[summary_mask & (comparison.reference == ref)][display]
                      .round(6).to_string(index=False), ""])
    raw_numeric = comparison[(comparison.family == "unbounded_numeric") & (comparison.reference == "C")]
    if len(raw_numeric):
        lines.extend(["Unbounded numeric C errors, by question and in its own units:",
                      raw_numeric[display + ["original_comparison_cases"]].round(6).to_string(index=False), ""])
    lines.extend(["Parsed prediction agreement with " + original_name + ":",
                  agreement.drop(columns="note").round(6).to_string(index=False), "", "Interpretation:",
                  "- pricing_40 and original_9_binary are separate subsets of MC; neither is the complete full-B task.",
                  "- Price-only methods are compared on exactly the same pricing cases as the original adapter.",
                  "- Product/price-conditioned baselines use training B. The native training_B_baseline column remains the original unconditioned train-B mode/median.",
                  "- Pricing gains over an unconditioned mode can reflect price sensitivity; they do not establish useful participant A information.",
                  "- A removed or shuffled changes the input. A paired loss with target stimuli fixed indicates useful A association on these validation cases, not causal understanding.",
                  "- No-A inputs can differ from the training distribution. A shuffled result is also descriptive and depends on the frozen shuffle.",
                  "- For unbounded numeric questions, published MAE is conditional on valid outputs. Original-versus-stage differences use the common valid mask; inspect coverage.",
                  "- Human B/C agreement is a paired reference, not a strict ceiling; its mask can differ when labels are missing.",
                  "- Question macros weight QIDs equally. Scalar answers from the same participant are correlated.",
                  "- No significance tests or confidence intervals are claimed here. If adding uncertainty, resample participants jointly across methods, not scalar rows.",
                  "- This is a development comparison on previously used validation participants. It does not replace a frozen final test evaluation.",
                  "- All per-question C and B primary metrics are saved in comparison.csv; native score folders retain secondary metrics and matrix exact matches."])
    if stage_metadata is not None:
        paths["stage_metadata"] = out / "stage_metadata.json"
        paths["stage_metadata"].write_text(json.dumps(stage_metadata, indent=2, ensure_ascii=False,
                                                    allow_nan=False) + "\n", encoding="utf-8")
        lines.extend(["", "Method metadata / retrieval coverage is stored separately in stage_metadata.json."])
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


make_comparison_report = report
