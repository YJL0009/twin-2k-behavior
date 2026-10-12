#!/usr/bin/env python3
"""Answer-free Twin-2K-500 B item schemas and strict scalar output parsing.

The schema is derived only from published survey metadata, never B/C answers.
Each Matrix row and Slider statement is evaluated as a separate scalar item.
Question text always comes from that participant's record: catalog example text
must not replace randomized prices or experimental conditions.
"""
import json
import math
import re

SCHEMA_PROTOCOL = "twin-full-b-scalar-schema-v1"
TRAINED_BINARY_QIDS = frozenset(("QID163", "QID165", "QID167", "QID169", "QID183",
                                "QID184", "QID192", "QID193", "QID196"))
PRICING_QIDS = frozenset(f"QID9_{i}" for i in range(1, 41))
ORDINAL_MC_QIDS = frozenset(("QID157", "QID158", "QID161", "QID162", "QID171",
    "QID172", "QID173", "QID174", "QID175", "QID176", "QID177", "QID178",
    "QID179", "QID189", "QID190", "QID191", "QID194", "QID195", "QID291"))
MC_QIDS = TRAINED_BINARY_QIDS | PRICING_QIDS | ORDINAL_MC_QIDS
MATRIX_NOMINAL_QIDS = frozenset(("QID198", "QID203"))
MATRIX_ORDINAL_QIDS = frozenset(("QID159", "QID160", "QID287", "QID288", "QID289"))
SLIDER_QIDS = frozenset(("QID154", "QID156", "QID290"))
TE_QIDS = frozenset(("QID164", "QID166", "QID168", "QID170", "QID181", "QID182"))
ANCHOR_FOR = {"QID164": "QID163", "QID166": "QID165",
              "QID168": "QID167", "QID170": "QID169"}
ANSWER_KEYS = frozenset(("Answers", "Answer", "SelectedByPosition", "SelectedText"))
NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
INTEGER_PATTERN = re.compile(r"[1-9]\d*", re.ASCII)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _has_answers(value):
    if isinstance(value, dict):
        return any(k in ANSWER_KEYS or _has_answers(v) for k, v in value.items())
    return isinstance(value, list) and any(_has_answers(v) for v in value)


def _header_lines(unit):
    lines = []
    if unit.get("BlockName"):
        lines.append("Section: " + str(unit["BlockName"]))
    for instruction in unit.get("PrecedingDisplayInstructions", []):
        require(isinstance(instruction.get("QuestionText"), str), "Display instruction lacks text")
        lines.append("Instructions:\n" + instruction["QuestionText"])
        for key in ("Options", "Rows", "Columns", "Statements", "Range"):
            if key in instruction:
                lines.append(key + ": " + dumps(instruction[key]))
    return lines


def _number_options(lines, options):
    lines.append("Options:")
    for number, option in enumerate(options, 1):
        lines.append(f"{number}. " + (option if isinstance(option, str) else dumps(option)))


def _plain_mc(unit):
    """Deliberately byte-identical to diagnose_liquid_outputs.plain_target for MC."""
    lines = _header_lines(unit)
    question = unit["Question"]
    require(isinstance(question.get("QuestionText"), str), "Target lacks question text")
    lines.append(question["QuestionText"])
    for key in ("Rows", "Columns", "Statements", "Range"):
        if key in question:
            lines.append(key + ": " + dumps(question[key]))
    _number_options(lines, question["Options"])
    return "\n\n".join(lines)


def _ids(question, ids_key, count):
    values = question.get(ids_key)
    if values is None:
        return [str(i) for i in range(1, count + 1)]
    require(isinstance(values, list) and len(values) == count, f"{ids_key} length mismatch")
    result = [str(v) for v in values]
    require(len(set(result)) == count and all(re.fullmatch(r"\d+", v) for v in result),
            f"Invalid or duplicate {ids_key}")
    return result


def _base_item(unit, family, item_id, sub_id=None, options=None, lower=None, upper=None):
    question = unit["Question"]
    return {"item_id": item_id, "qid": str(question["QuestionID"]),
            "question_type": question["QuestionType"], "family": family,
            "block_name": str(unit.get("BlockName", "")), "sub_id": sub_id,
            "options": options, "lower": lower, "upper": upper,
            "trained_binary": str(question["QuestionID"]) in TRAINED_BINARY_QIDS,
            "pricing": str(question["QuestionID"]) in PRICING_QIDS,
            "schema_protocol": SCHEMA_PROTOCOL,
            "schema_source": "participant B metadata; answers stripped before expansion"}


def expand_units(units, catalog=None):
    """Expand answer-free ``(qid, survey_unit)`` pairs into scalar item dictionaries.

    Units are supplied by the existing survey_units(..., include_answers=False).
    Catalog, when supplied, verifies item IDs and option/row metadata only. Its
    example question text is not substituted for participant-specific wording.
    """
    units = list(units)
    by_qid = {}
    for qid, unit in units:
        require(not _has_answers(unit), f"{qid}: answer payload reached prompt schema")
        question = unit.get("Question", {})
        require(str(question.get("QuestionID")) == str(qid), "Unit QID mismatch")
        require(str(qid) not in by_qid, f"Duplicate B QID: {qid}")
        require(isinstance(question.get("QuestionText"), str), f"{qid}: missing QuestionText")
        by_qid[str(qid)] = unit
    output = []
    for qid, unit in units:
        qid = str(qid)
        q = unit["Question"]
        kind = q["QuestionType"]
        settings = q.get("Settings") or {}
        if kind == "DB":
            require(qid == "QID8", f"Unexpected B display QID: {qid}")
            continue
        if kind == "MC":
            require(qid in MC_QIDS and settings.get("Selector") in ("SAVR", "SAHR"),
                    f"Unsupported B MC schema: {qid}/{settings}")
            options = q.get("Options")
            require(isinstance(options, list) and len(options) >= 2, f"{qid}: missing MC options")
            if qid in TRAINED_BINARY_QIDS | PRICING_QIDS:
                require(len(options) == 2, f"{qid}: binary question option count changed")
            family = "ordinal" if qid in ORDINAL_MC_QIDS else "mc"
            item = _base_item(unit, family, qid, options=list(options), lower=1, upper=len(options))
            item["target_text"] = _plain_mc(unit)
            output.append(item)
        elif kind == "Matrix":
            require(qid in MATRIX_NOMINAL_QIDS | MATRIX_ORDINAL_QIDS
                    and settings.get("Selector") == "Likert"
                    and settings.get("SubSelector") == "SingleAnswer", f"Unsupported B Matrix: {qid}")
            rows, options = q.get("Rows"), q.get("Columns")
            require(isinstance(rows, list) and rows and isinstance(options, list) and len(options) >= 2,
                    f"{qid}: incomplete matrix metadata")
            row_ids = _ids(q, "RowsID", len(rows))
            family = "matrix_nominal" if qid in MATRIX_NOMINAL_QIDS else "matrix_ordinal"
            for row_id, row in zip(row_ids, rows):
                item = _base_item(unit, family, f"{qid}_{row_id}", row_id,
                                  list(options), 1, len(options))
                lines = _header_lines(unit) + [q["QuestionText"], "Answer for this row only:\n" + str(row)]
                _number_options(lines, options)
                item["target_text"] = "\n\n".join(lines)
                output.append(item)
        elif kind == "Slider":
            require(qid in SLIDER_QIDS and settings.get("Selector") == "HSLIDER", f"Unsupported B Slider: {qid}")
            statements = q.get("Statements")
            require(isinstance(statements, list) and statements, f"{qid}: missing slider statements")
            statement_ids = _ids(q, "StatementsID", len(statements))
            bounds = q.get("Range") or {}
            require(bounds.get("Min") == 0 and bounds.get("Max") == 100,
                    f"{qid}: published slider bounds changed")
            for sid, statement in zip(statement_ids, statements):
                multi = len(statements) > 1
                item = _base_item(unit, "bounded_numeric", f"{qid}_{sid}" if multi else qid,
                                  sid if multi else None, lower=0, upper=100)
                lines = _header_lines(unit) + [q["QuestionText"]]
                if statement:
                    lines.append("Answer for this statement only:\n" + str(statement))
                lines.append("Allowed numeric range: 0 to 100 (inclusive).")
                item["target_text"] = "\n\n".join(lines)
                output.append(item)
        elif kind == "TE":
            require(qid in TE_QIDS and settings.get("Selector") == "SL", f"Unsupported B text-entry: {qid}")
            # survey_units' metadata whitelist removes ContentType. The public
            # catalog establishes that these six QIDs require numeric answers.
            require(settings.get("ContentType", "ValidNumber") == "ValidNumber", f"{qid}: nonnumeric TE")
            bounded = qid in ("QID181", "QID182")
            item = _base_item(unit, "bounded_numeric" if bounded else "unbounded_numeric",
                              f"{qid}_TEXT", lower=0, upper=20 if bounded else None)
            lines = _header_lines(unit)
            if qid in ANCHOR_FOR:
                anchor_qid = ANCHOR_FOR[qid]
                anchor = by_qid.get(anchor_qid)
                require(anchor is not None, f"{qid}: missing preceding anchoring question {anchor_qid}")
                require(str(anchor.get("BlockName", "")).strip() == str(unit.get("BlockName", "")).strip(),
                        f"{qid}: anchor belongs to a different block")
                lines.append("Earlier question in this section (no participant answer supplied):\n"
                             + _plain_mc(anchor))
                item["anchor_question_id"] = anchor_qid
            lines.append(q["QuestionText"])
            if bounded:
                lines.append("Allowed numeric range: 0 to 20 (inclusive).")
            else:
                lines.append("Return a nonnegative numeric estimate; no fixed upper limit is imposed.")
            item["target_text"] = "\n\n".join(lines)
            output.append(item)
        else:
            raise ValueError(f"Unsupported B QuestionType: {qid}/{kind}")
    require(len({item["item_id"] for item in output}) == len(output), "Expanded B item ID collision")
    if catalog is not None:
        expected = {item["item_id"]: item for item in catalog_items(catalog)}
        for item in output:
            require(item["item_id"] in expected, f"Item absent from official catalog: {item['item_id']}")
            spec = expected[item["item_id"]]
            for key in ("qid", "question_type", "family", "sub_id", "options", "lower", "upper"):
                require(item[key] == spec[key], f"Catalog schema mismatch: {item['item_id']}/{key}")
    return output


def catalog_items(catalog):
    """Return all 126 possible scalar schemas across every B condition."""
    require(isinstance(catalog, list), "Catalog must be a list")
    b = [q for q in catalog if q.get("source") == "wave4_Q_wave1_3_A"]
    require(len(b) == 85, f"Expected 85 B catalog QIDs, found {len(b)}")
    units = [(q["QuestionID"], {"BlockName": q.get("BlockName", ""),
              "PrecedingDisplayInstructions": [],
              "Question": {k: v for k, v in q.items() if k not in
                           ("BlockName", "source", "csv_columns", "is_descriptive")}}) for q in b]
    result = expand_units(units)
    expected_columns = [col for q in b for col in q["csv_columns"]]
    require(len(result) == 126 and len(expected_columns) == 126
            and {i["item_id"] for i in result} == set(expected_columns),
            "Expanded item IDs do not match the 126 published B CSV columns")
    return result


def parse_prediction(raw, item):
    """Strict complete-output parser. No extraction, rounding, clipping, or retries."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if item["family"] in ("mc", "ordinal", "matrix_nominal", "matrix_ordinal"):
        if not INTEGER_PATTERN.fullmatch(value):
            return None
        # Limit digit count before int conversion: out-of-range integer strings
        # are invalid even when they exceed Python's security conversion limit.
        if len(value) > len(str(len(item["options"]))):
            return None
        number = int(value)
        return number if 1 <= number <= len(item["options"]) else None
    if item["family"] not in ("bounded_numeric", "unbounded_numeric"):
        raise ValueError(f"Unknown response family: {item['family']}")
    if not NUMBER_PATTERN.fullmatch(value):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    if item.get("lower") is not None and number < item["lower"]:
        return None
    if item.get("upper") is not None and number > item["upper"]:
        return None
    return number


def response_instruction(item):
    """Describe this item's answer representation without reference to labels."""
    if item["family"] in ("mc", "ordinal", "matrix_nominal", "matrix_ordinal"):
        return (f"Predict the participant's answer. Return exactly one option number from 1 to "
                f"{len(item['options'])}. Return only the integer, with no explanation.")
    bounds = ""
    if item.get("upper") is not None:
        bounds = f" between {item['lower']} and {item['upper']} inclusive"
    return ("Predict the participant's answer. Return exactly one nonnegative number" + bounds
            + ". Return only the number, with no units, commas, percent sign, or explanation.")
