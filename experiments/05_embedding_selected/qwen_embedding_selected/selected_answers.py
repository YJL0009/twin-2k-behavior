"""Loss-aware, A-only selected-answer text for frozen profile embeddings.

This module maps the explicit SelectedByPosition/SelectedText schema found in
Twin-2K-500, rather than interpreting arbitrary answer digits as option indices.
Question prose and preceding instructions are preserved. Ordinary unselected
choice lists are omitted; numerical scale anchors and bipolar comparison
contexts remain because they define what the selected answer means.

Some source matrices contain a shortened answer list without row identifiers.
Those observations are retained explicitly as an unaligned sequence. They are
never zipped to the first rows, imputed, or silently discarded.
"""
from __future__ import annotations

import json
import math
import re

PROTOCOL = "selected-answer-a-v1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _strings(value, name, *, nonempty=True):
    _require(isinstance(value, list) and (bool(value) or not nonempty)
             and all(isinstance(item, str) for item in value),
             f"{name} must be an ordered list of strings")
    return value


def _scalar(value, name):
    _require(isinstance(value, (str, int, float)) and not isinstance(value, bool),
             f"{name} must be text or a finite number")
    _require(not isinstance(value, float) or math.isfinite(value),
             f"{name} contains a non-finite number")
    return _json(value)


def _selected(answer, choices, *, multiple):
    _require(isinstance(answer, dict)
             and set(answer) == {"SelectedByPosition", "SelectedText"},
             "Expected explicit SelectedByPosition and SelectedText only")
    positions, texts = answer["SelectedByPosition"], answer["SelectedText"]
    if multiple:
        _require(isinstance(positions, list) and isinstance(texts, list),
                 "Expected position and selected-text lists")
    else:
        _require(not isinstance(positions, list) and not isinstance(texts, list),
                 "Expected one position and one selected text")
        positions, texts = [positions], [texts]
    _require(len(positions) == len(texts) and bool(positions),
             "Selected position/text lengths differ or are empty")
    for position, text in zip(positions, texts):
        _require(isinstance(position, int) and not isinstance(position, bool)
                 and 1 <= position <= len(choices),
                 "SelectedByPosition is not a valid one-based schema position")
        _require(isinstance(text, str) and choices[position - 1] == text,
                 "SelectedText contradicts the option at SelectedByPosition")
    return positions, texts


def _scale_context(choices):
    """Retain labeled anchors only for an explicit consecutive numeric scale.

    E.g. ['Not at all 0','1','2','Severely 3'] is a scale. A household-size
    list ending in 'More than 4', or Wason cards ['A','F','3','7'], is not.
    """
    parsed = [re.search(r"(?:^|\s)(-?\d+)\s*$", choice) for choice in choices]
    if not all(parsed):
        return [], set()
    numbers = [int(match.group(1)) for match in parsed]
    if numbers[0] not in (0, 1) or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        return [], set()
    anchor_indices = {i for i, choice in enumerate(choices)
                      if not re.fullmatch(r"\s*-?\d+\s*", choice)}
    if not anchor_indices:
        return [], set()
    anchors = "; ".join(_json(choices[i]) for i in sorted(anchor_indices))
    return [f"Scale context: {numbers[0]} to {numbers[-1]}; labeled anchors: {anchors}"], anchor_indices


def _context(unit, question):
    lines = []
    block = unit.get("BlockName")
    _require(block is None or isinstance(block, str), "BlockName must be text")
    if block:
        lines.append("Section: " + block)
    instructions = unit.get("PrecedingDisplayInstructions", [])
    _require(isinstance(instructions, list), "PrecedingDisplayInstructions must be a list")
    for instruction in instructions:
        _require(isinstance(instruction, dict)
                 and isinstance(instruction.get("QuestionText"), str),
                 "Display instruction must have QuestionText")
        _require(not any(key in instruction for key in ("Answers", "Answer", "SelectedText", "SelectedByPosition")),
                 "Display instruction unexpectedly contains an answer")
        lines.append("Instructions: " + instruction["QuestionText"])
        for key in ("Options", "Rows", "Columns", "Statements", "Range"):
            if key in instruction:
                lines.append("Instruction " + key + ": " + _json(instruction[key]))
    lines.append("Question: " + question["QuestionText"])
    if "Range" in question:
        lines.append("Range: " + _json(question["Range"]))
    return lines


def _convert(unit):
    _require(isinstance(unit, dict) and isinstance(unit.get("Question"), dict),
             "Expected an A survey unit with Question")
    q = unit["Question"]
    _require(isinstance(q.get("QuestionText"), str), "QuestionText must be text")
    _require("Answers" in q, "A unit lacks its own Answers")
    settings = q.get("Settings", {})
    _require(isinstance(settings, dict), "Settings must be a mapping")
    selector, subselector = settings.get("Selector"), settings.get("SubSelector")
    question_type, answer = q.get("QuestionType"), q["Answers"]
    lines = _context(unit, q)
    info = {"kind": "", "response_count": 0, "declared_rows": 0,
            "row_alignment_ambiguous": False, "alignment": "not_applicable",
            "selected_text_verified": False, "scale_anchors_retained": 0,
            "bipolar_context_retained": False, "unselected_options_omitted": 0}
    if question_type == "MC":
        choices = _strings(q.get("Options"), "Options")
        _require(selector in (None, "SAVR", "SAHR", "MAVR", "MAHR"),
                 "Unsupported MC selector")
        _require(subselector in (None, "TX"), "Unsupported MC subselector")
        _require(not any(key in q for key in ("Rows", "Columns", "Statements")),
                 "MC contains unsupported row/column/statement structure")
        _require(isinstance(answer, dict), "MC Answers must be a mapping")
        multiple = selector in ("MAVR", "MAHR") if selector is not None else isinstance(answer.get("SelectedByPosition"), list)
        positions, texts = _selected(answer, choices, multiple=multiple)
        _require(len(set(positions)) == len(positions), "Repeated MC selection positions")
        scale_lines, anchors = _scale_context(choices)
        lines.extend(scale_lines)
        lines.append(("Selected answers: " + _json(texts)) if multiple
                     else ("Selected answer: " + _json(texts[0])))
        info.update(kind="mc_multiple" if multiple else "mc_single",
                    response_count=len(positions), selected_text_verified=True,
                    scale_anchors_retained=len(anchors),
                    unselected_options_omitted=len(set(range(len(choices)))
                                                    - {p - 1 for p in positions} - anchors))
    elif question_type == "Matrix":
        rows = _strings(q.get("Rows"), "Rows")
        columns = _strings(q.get("Columns"), "Columns")
        _require(selector in ("Likert", "Bipolar"), "Unsupported Matrix selector")
        _require((selector == "Likert" and subselector in (None, "SingleAnswer"))
                 or (selector == "Bipolar" and subselector in (None, "")),
                 "Unsupported Matrix subselector")
        positions, texts = _selected(answer, columns, multiple=True)
        _require(len(positions) <= len(rows), "Matrix has more answers than declared rows")
        if "RowsID" in q:
            _require(isinstance(q["RowsID"], list) and len(q["RowsID"]) == len(rows),
                     "RowsID does not describe all matrix rows")
        if "ColumnsID" in q:
            _require(isinstance(q["ColumnsID"], list) and len(q["ColumnsID"]) == len(columns),
                     "ColumnsID does not describe all matrix columns")
        scale_lines, anchors = _scale_context(columns)
        lines.extend(scale_lines)
        ambiguous = len(positions) != len(rows)
        info.update(kind="matrix_bipolar" if selector == "Bipolar" else "matrix_likert",
                    response_count=len(positions), declared_rows=len(rows),
                    row_alignment_ambiguous=ambiguous,
                    alignment="unknown_short_response_list" if ambiguous else "complete",
                    selected_text_verified=True, scale_anchors_retained=len(anchors),
                    bipolar_context_retained=selector == "Bipolar",
                    unselected_options_omitted=len(set(range(len(columns)))
                                                    - {p - 1 for p in positions} - anchors))
        if selector == "Bipolar":
            _require(columns == ["1", "2"],
                     "Bipolar mapping requires explicit ordered columns ['1','2']")
            _require(all(row.count(":") == 1 and all(part.strip() for part in row.split(":"))
                         for row in rows),
                     "Bipolar mapping requires exactly two colon-separated row outcomes")
        if ambiguous:
            lines.append("Row alignment unavailable: the source provides fewer selected responses than rows "
                         "and no answer row identifiers. Do not assign these responses to individual rows.")
            for number, row in enumerate(rows, 1):
                lines.append(f"Declared row {number}: " + _json(row))
            if selector == "Bipolar":
                lines.append("Bipolar coding: selected column 1 means the left outcome before the colon; "
                             "selected column 2 means the right outcome after the colon.")
            lines.append("Observed selected-answer sequence (row alignment unknown): " + _json(texts))
        else:
            for number, (row, position, text) in enumerate(zip(rows, positions, texts), 1):
                if selector == "Bipolar":
                    outcomes = row.split(":")
                    lines.append(f"Choice {number}: left {_json(outcomes[0])}; right {_json(outcomes[1])}; "
                                 f"selected {'left' if position == 1 else 'right'} outcome: {_json(outcomes[position - 1])}")
                else:
                    lines.append("Row: " + _json(row) + " | Selected answer: " + _json(text))
    elif question_type == "TE":
        _require(isinstance(answer, dict) and set(answer) == {"Text"},
                 "TE Answers must contain Text only")
        value = answer["Text"]
        if selector == "FORM":
            rows = _strings(q.get("Rows"), "Rows")
            _require(len(set(rows)) == len(rows), "Form row labels are not unique")
            _require(isinstance(value, list) and bool(value), "Form Text must be a nonempty labeled list")
            observed = []
            for field in value:
                _require(isinstance(field, dict) and len(field) == 1,
                         "Each form response must have exactly one row label")
                label, response = next(iter(field.items()))
                _require(label in rows and label not in observed, "Unknown or repeated form response row label")
                observed.append(label)
                lines.append("Field: " + _json(label) + " | Answer: " + _scalar(response, "Form response"))
            info.update(kind="text_form", response_count=len(observed), declared_rows=len(rows),
                        alignment="explicit_form_labels")
        else:
            _require(selector in (None, "SL", "ML"), "Unsupported TE selector")
            _require(not any(key in q for key in ("Rows", "Columns", "Options", "Statements")),
                     "Scalar TE contains unsupported row/column/option structure")
            lines.append("Answer: " + _scalar(value, "Text response"))
            info.update(kind="text_scalar", response_count=1)
    else:
        raise ValueError("Unsupported A question type")
    return "\n".join(lines), info


def _checked_convert(unit):
    try:
        return _convert(unit)
    except (ValueError, TypeError, KeyError) as exc:
        q = unit.get("Question", {}) if isinstance(unit, dict) else {}
        if not isinstance(q, dict):
            q = {}
        raise ValueError(f"Selected-answer representation failed for A question "
                         f"{q.get('QuestionID', '<unknown>')} ({q.get('QuestionType', '<unknown>')}): {exc}") from exc


def render_selected_a_unit(unit):
    """Return selected-answer embedding text, without IDs or B/C information."""
    return _checked_convert(unit)[0]


def describe_selected_a_unit(unit):
    """Validate a unit and return aggregate-safe conversion diagnostics only."""
    return _checked_convert(unit)[1]
