"""CPU-only checks for the restored selected-answer representation audit."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

SELECTED = (Path(__file__).resolve().parents[1] / "experiments"
            / "05_embedding_selected" / "qwen_embedding_selected")
sys.path.insert(0, str(SELECTED))
from audit_selected_answers import audit_people
from selected_answers import render_selected_a_unit


def unit(qid, question_type, selector, answers, **metadata):
    return (qid, {"BlockName": "Synthetic section", "PrecedingDisplayInstructions": [],
                  "Question": {"QuestionID": qid, "QuestionText": "Synthetic question?",
                               "QuestionType": question_type,
                               "Settings": {"Selector": selector}, "Answers": answers,
                               **metadata}})


def chosen(positions, texts):
    return {"SelectedByPosition": positions, "SelectedText": texts}


def fixture():
    return {"synthetic-participant-private-id": {"a_units": [
        unit("Q1", "MC", "SAVR", chosen(2, "1"),
             Options=["Never 0", "1", "2", "Always 3"]),
        unit("Q2", "MC", "MAVR", chosen([1, 3], ["Red", "Blue"]),
             Options=["Red", "Green", "Blue"]),
        unit("Q3", "Matrix", "Likert", chosen([1, 2], ["No", "Yes"]),
             Rows=["First", "Second"], Columns=["No", "Yes"]),
        unit("Q4", "Matrix", "Likert", chosen([2], ["Yes"]),
             Rows=["First", "Second"], Columns=["No", "Yes"]),
        unit("Q5", "Matrix", "Bipolar", chosen([2], ["2"]),
             Rows=["Left outcome:Right outcome"], Columns=["1", "2"]),
        unit("Q6", "TE", "FORM", {"Text": [{"Age": 30}]},
             Rows=["Age", "Household size"]),
        unit("Q7", "TE", "SL", {"Text": "synthetic-private-free-text-answer"}),
    ], "items": {}, "targets": {}}}


class SelectedAnswerAuditTests(unittest.TestCase):
    def test_all_supported_families_have_correct_aggregate_counts(self):
        report = audit_people(fixture())
        self.assertEqual(report, {
            "representation_protocol": "selected-answer-a-v1", "participants": 1,
            "a_units": 7, "response_count": 9,
            "ambiguous_matrix_alignment_units": 1, "bipolar_context_retained_units": 1,
            "retained_scale_anchors": 2, "selected_text_verified_units": 5,
            "unselected_options_omitted": 4,
            "kind_counts": {"matrix_bipolar": 1, "matrix_likert": 2, "mc_multiple": 1,
                            "mc_single": 1, "text_form": 1, "text_scalar": 1},
            "alignment_counts": {"complete": 2, "explicit_form_labels": 1,
                                 "not_applicable": 3, "unknown_short_response_list": 1},
            "question_type_selector_counts": {"MC/MAVR": 1, "MC/SAVR": 1,
                                               "Matrix/Bipolar": 1, "Matrix/Likert": 2,
                                               "TE/FORM": 1, "TE/SL": 1},
            "ambiguous_matrix_policy": "Keep all row labels and the selected response sequence; "
                                       "do not infer missing row locations.",
            "all_units_validated": True, "raw_answers_or_participant_ids_in_report": False,
        })

    def test_does_not_mutate_input_or_include_private_values_in_report(self):
        people = fixture()
        original = deepcopy(people)
        report = audit_people(people)
        self.assertEqual(people, original)
        serialized = json.dumps(report, allow_nan=False)
        self.assertNotIn("synthetic-participant-private-id", serialized)
        self.assertNotIn("synthetic-private-free-text-answer", serialized)
        self.assertNotIn("Synthetic question?", serialized)

    def test_only_a_is_read_and_reused_qids_are_allowed(self):
        people = fixture()
        person = next(iter(people.values()))
        person["items"] = object()
        person["targets"] = object()
        person["a_units"].append(deepcopy(person["a_units"][0]))
        self.assertEqual(audit_people(people)["a_units"], 8)

    def test_short_matrix_keeps_sequence_without_inventing_row_alignment(self):
        pair = next(iter(fixture().values()))["a_units"][3]
        text = render_selected_a_unit(pair[1])
        self.assertIn('Declared row 1: "First"', text)
        self.assertIn('Declared row 2: "Second"', text)
        self.assertIn('Observed selected-answer sequence (row alignment unknown): ["Yes"]', text)
        self.assertNotIn('Row: "First" | Selected answer:', text)

    def test_contradictory_selected_text_is_rejected(self):
        people = fixture()
        next(iter(people.values()))["a_units"][0][1]["Question"]["Answers"]["SelectedText"] = "Never 0"
        with self.assertRaisesRegex(ValueError, "SelectedText contradicts"):
            audit_people(people)

    def test_out_of_range_choice_is_rejected(self):
        people = fixture()
        next(iter(people.values()))["a_units"][0][1]["Question"]["Answers"]["SelectedByPosition"] = 0
        with self.assertRaisesRegex(ValueError, "one-based schema position"):
            audit_people(people)

    def test_nonfinite_scalar_is_rejected(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=invalid):
                people = {"p": {"a_units": [unit("Q", "TE", "SL", {"Text": invalid})]}}
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    audit_people(people)

    def test_unknown_form_row_is_rejected(self):
        people = {"p": {"a_units": [unit("Q", "TE", "FORM", {"Text": [{"Unknown": 5}]},
                                             Rows=["Known"])]}}
        with self.assertRaisesRegex(ValueError, "Unknown or repeated form"):
            audit_people(people)

    def test_malformed_profile_containers_are_rejected(self):
        for people in ({}, [], {"p": None}, {"p": {"a_units": []}},
                       {"p": {"a_units": ["not-a-pair"]}}):
            with self.subTest(people=people), self.assertRaises(ValueError):
                audit_people(people)


if __name__ == "__main__":
    unittest.main()
