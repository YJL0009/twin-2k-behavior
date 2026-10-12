"""Validate every selected-answer A unit before embedding, without loading models.

The original audit helper was omitted from the public source package. This
replacement uses the unchanged selected-answer converter and preserves the
aggregate fields found in the archived ``selected_answer_audit.json``. It does
not change profile text, choose neighbors, infer missing matrix rows, or read
the person's target metadata/labels. Its 15 report fields were checked against
the archived report using all 1,749 training/validation A profiles and matched
exactly. The GPU retrieval/generation pipeline was not rerun for this repair.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from selected_answers import (
    PROTOCOL,
    describe_selected_a_unit,
    render_selected_a_unit,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_people(people):
    """Check ``{pid: {"a_units": [(qid, unit), ...], ...}}`` and return totals.

    This is the structure returned by ``support.data_io.load_people``. Reused A
    question IDs across sections are allowed; every unit is checked separately.
    The existing converter checks selected positions against selected text,
    validates typed responses and preserves explicitly ambiguous matrix rows.
    A malformed unit raises before an ``all_units_validated`` report is returned.
    Successful reports contain neither participant IDs nor response values.
    """
    _require(isinstance(people, Mapping) and bool(people),
             "Selected-answer audit requires a nonempty participant mapping")
    kinds, alignments, selectors = Counter(), Counter(), Counter()
    totals = {
        "a_units": 0,
        "response_count": 0,
        "ambiguous_matrix_alignment_units": 0,
        "bipolar_context_retained_units": 0,
        "retained_scale_anchors": 0,
        "selected_text_verified_units": 0,
        "unselected_options_omitted": 0,
    }
    fields = {
        "response_count": "response_count",
        "ambiguous_matrix_alignment_units": "row_alignment_ambiguous",
        "bipolar_context_retained_units": "bipolar_context_retained",
        "retained_scale_anchors": "scale_anchors_retained",
        "selected_text_verified_units": "selected_text_verified",
        "unselected_options_omitted": "unselected_options_omitted",
    }
    for person in people.values():
        _require(isinstance(person, Mapping), "Participant payload must be a mapping")
        units = person.get("a_units")
        _require(isinstance(units, (list, tuple)) and bool(units),
                 "Cannot audit an empty or malformed A profile")
        for pair in units:
            _require(isinstance(pair, (list, tuple)) and len(pair) == 2,
                     "A units must be (question identifier, unit payload) pairs")
            unit = pair[1]
            # Use the same converter as embedding_index.render_a_profile. The
            # diagnostics validate the source schema, while this second public
            # call checks that the actual embedding text remains nonempty.
            info = describe_selected_a_unit(unit)
            text = render_selected_a_unit(unit)
            _require(isinstance(text, str) and bool(text.strip()),
                     "A unit rendered to empty selected-answer text")
            question = unit["Question"]
            selector = question.get("Settings", {}).get("Selector")
            selectors[f"{question['QuestionType']}/{selector}"] += 1
            kinds[info["kind"]] += 1
            alignments[info["alignment"]] += 1
            totals["a_units"] += 1
            for output, diagnostic in fields.items():
                totals[output] += int(info[diagnostic])

    return {
        "representation_protocol": PROTOCOL,
        "participants": len(people),
        **totals,
        "kind_counts": dict(sorted(kinds.items())),
        "alignment_counts": dict(sorted(alignments.items())),
        "question_type_selector_counts": dict(sorted(selectors.items())),
        "ambiguous_matrix_policy": (
            "Keep all row labels and the selected response sequence; "
            "do not infer missing row locations."
        ),
        "all_units_validated": True,
        "raw_answers_or_participant_ids_in_report": False,
    }
