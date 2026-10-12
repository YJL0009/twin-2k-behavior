"""Lossless question-once / answer-per-person rendering for selected A units.

The complete answer-free metadata is the schema identity. Repeated occurrences
of that exact schema within a person's full A profile remain separate slots;
selection never changes their occurrence numbers. Raw ``Question.Answers``
values are kept as JSON rather than converted to guessed option positions.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

if __package__:
    from .support import prompts as native_prompts
else:
    from support import prompts as native_prompts

PROTOCOL = "shared-A-exact-schema-occurrence-plain-definitions-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def _has_answer_key(value):
    if isinstance(value, dict):
        return "Answers" in value or any(_has_answer_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_answer_key(child) for child in value)
    return False


def profile_entries(a_units):
    """Return stable schema/occurrence/answer entries in original profile order.

    All metadata (including block, instructions, IDs, settings, options, row and
    column mappings, and ranges) participates in exact identity. Dict ordering
    is immaterial; list ordering and numeric/string distinctions are preserved.
    The source profile is never mutated. Empty/zero/false answer JSON is kept.
    """
    seen = Counter()
    output = []
    for index, source in enumerate(a_units):
        require(isinstance(source, (tuple, list)) and len(source) == 2,
                "A units must be (question_id, metadata_and_answer) pairs")
        unit = source[1]
        require(isinstance(unit, dict) and isinstance(unit.get("Question"), dict),
                "A unit lacks a Question object")
        require("Answers" in unit["Question"], "A unit lacks raw Question.Answers")
        schema = deepcopy(unit)
        answer = schema["Question"].pop("Answers")
        require(not _has_answer_key(schema), "Unexpected answer payload in A schema metadata")
        schema_json = dumps(schema)
        seen[schema_json] += 1
        occurrence = seen[schema_json]
        # Use the already established readable question/context/options format
        # in the prompt. Identity still includes every metadata key, including
        # IDs/settings that need not be repeated as model-facing JSON.
        definition_unit = deepcopy(schema)
        definition_unit["Question"]["Answers"] = None
        definition_text, marker, dummy_answer = native_prompts.compact_a_unit(definition_unit).rpartition("\nResponse: ")
        require(marker and dummy_answer == "null", "Could not separate question definition from dummy answer")
        output.append({
            "index": index, "schema": schema, "schema_json": schema_json,
            "schema_hash": hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
            "definition_text": definition_text,
            "occurrence": occurrence, "answer": answer, "answer_json": dumps(answer),
            "semantic_hash": digest({"schema": schema, "occurrence": occurrence, "answer": answer}),
        })
    return tuple(output)


def _semantics(rows):
    return [{"schema": entry["schema"], "occurrence": entry["occurrence"],
             "answer": entry["answer"]} for entry in rows]


def build_table(own_entries, donor_entries):
    """Build shared definitions and answer mappings, assigning current A first.

    ``donor_entries`` is a list of selected-entry lists in retrieval rank order.
    Full metadata plus occurrence is matched exactly across people. Own IDs
    therefore remain identical when adding/removing the reference sections.
    """
    own_entries = list(own_entries)
    donor_entries = [list(rows) for rows in donor_entries]
    definitions, lookup = [], {}

    def map_rows(entries):
        result, seen = [], set()
        for entry in entries:
            index = entry["index"]
            require(type(index) is int and index >= 0 and index not in seen,
                    "Selected A indices must be distinct nonnegative integers for each profile")
            seen.add(index)
            key = (entry["schema_json"], entry["occurrence"])
            if key not in lookup:
                schema_id = f"A{len(definitions) + 1}"
                lookup[key] = schema_id
                definitions.append({"id": schema_id, "schema": entry["schema"],
                    "schema_json": entry["schema_json"], "schema_hash": entry["schema_hash"],
                    "definition_text": entry["definition_text"],
                    "occurrence": entry["occurrence"]})
            result.append({"id": lookup[key], "index": index, "answer": entry["answer"],
                "answer_json": entry["answer_json"], "schema_hash": entry["schema_hash"],
                "occurrence": entry["occurrence"], "semantic_hash": entry["semantic_hash"]})
        require(len({row["id"] for row in result}) == len(result),
                "A occurrences collapsed into one schema ID within a participant")
        return result

    own = map_rows(own_entries)
    donors = [map_rows(rows) for rows in donor_entries]
    response_count = len(own) + sum(map(len, donors))
    return {"protocol": PROTOCOL, "definitions": definitions, "own": own, "donors": donors,
            "response_count": response_count,
            "definition_reuse_count": response_count - len(definitions),
            "own_semantic_hash": digest(_semantics(own_entries)),
            "mapping_semantic_hash": digest({"own": _semantics(own_entries),
                                              "donors": [_semantics(rows) for rows in donor_entries]})}


def render_definitions(table):
    """Render question/context/options once in the prior readable A format."""
    lines = ["SHARED A QUESTION DEFINITIONS:",
             "A identifiers below are local lookup labels, not answer choices. Each participant's response "
             "is its original Answers JSON, interpreted using the question and options below. "
             "Occurrence distinguishes repeated appearances of an identical question in a profile."]
    if not table["definitions"]:
        lines.append("(No complete A question was included in these excerpts.)")
    for definition in table["definitions"]:
        lines.append(f"{definition['id']} [occurrence {definition['occurrence']}]:\n{definition['definition_text']}")
    return "\n".join(lines)


def render_answers(rows, all_definition_ids=None):
    """Only answer JSON is repeated per person, with explicit absence semantics.

    The optional definition-ID list validates references without expanding a
    dense missing-value table. An absent ID means not selected for this prompt,
    which must not be confused with an actual unanswered survey question.
    """
    if all_definition_ids is not None:
        require({row["id"] for row in rows} <= set(all_definition_ids),
                "A answer references an undefined shared question")
    lines = [f"{row['id']} = {row['answer_json']}" for row in rows]
    if not lines:
        lines.append("(No complete A response fits this snippet budget.)")
    lines.append("Unlisted definitions: NOT_INCLUDED_IN_EXCERPT (not a survey answer).")
    return "\n".join(lines)
