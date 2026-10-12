"""Portable demo tests: prompt parity, leakage rejection and no-model execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("infer_adapter", ROOT / "scripts/infer_adapter.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def example(name):
    return json.loads((ROOT / "examples" / name).read_text())


class PortableInferenceTests(unittest.TestCase):
    def test_binary_matches_original_plain_target_conversion(self):
        payload = example("full_b_binary.json")
        actual, item, cap = demo.build_messages(payload)
        engine = demo.native_modules()
        context = "\n\n".join(engine.base.dumps(unit) for unit in payload["a_units"])
        target = {"Question": {"QuestionID": "SYNTHETIC_TARGET", "QuestionType": "MC",
                  "QuestionText": "Which charity would you choose to support?",
                  "Options": ["A local food bank", "A wildlife rescue organization"]}}
        original = {"messages": engine.base.messages_for(context, target), "pid": "synthetic",
                    "qid": "SYNTHETIC_TARGET", "input_tokens": 0}
        original["prompt_hash"] = engine.base.digest_text(engine.base.dumps(original["messages"]))
        expected = engine.inputs.diag.make_case("synthetic", original, plain=True)["messages"]
        self.assertEqual(actual, expected)
        self.assertEqual(cap, 8)
        self.assertEqual(engine.schema.parse_prediction("1", item), 1)
        self.assertIsNone(engine.schema.parse_prediction("Yes", item))

    def test_scalar_uses_native_general_system_and_bounds(self):
        payload = example("full_b_numeric.json")
        messages, item, cap = demo.build_messages(payload)
        engine = demo.native_modules()
        context = "\n\n".join(engine.base.dumps(unit) for unit in payload["a_units"])
        self.assertEqual(messages, engine.inputs.messages_for_item(context, item))
        self.assertEqual(messages[0]["content"], engine.inputs.GENERAL_SYSTEM)
        self.assertIn("between 0 and 100 inclusive", messages[1]["content"])
        self.assertEqual(cap, 16)
        self.assertEqual(engine.schema.parse_prediction("50", item), 50)
        self.assertIsNone(engine.schema.parse_prediction("101", item))
        self.assertIsNone(engine.schema.parse_prediction("50%", item))

    def test_target_answer_is_rejected(self):
        payload = example("full_b_binary.json")
        for key in ("Answers", "label_b", "label_c", "prediction"):
            contaminated = deepcopy(payload)
            contaminated["target_item"]["nested"] = {key: 1}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "answer-free"):
                demo.build_messages(contaminated)

    def test_unanswered_a_units_are_rejected_but_zero_is_preserved(self):
        payload = example("full_b_numeric.json")
        for answer in (None, {}, [], "", " \n\t"):
            unanswered = deepcopy(payload)
            unanswered["a_units"][0]["Question"]["Answers"] = answer
            with self.subTest(answer=answer), self.assertRaisesRegex(ValueError, "nonempty earlier answer"):
                demo.build_messages(unanswered)
        for answer in (0, {"Text": 0}, {"SelectedByPosition": 1, "SelectedText": "Tea"}):
            answered = deepcopy(payload)
            answered["a_units"][0]["Question"]["Answers"] = answer
            with self.subTest(answer=answer):
                messages, _, _ = demo.build_messages(answered)
                self.assertIn(demo.native_modules().base.dumps(answer), messages[1]["content"])

    def test_invalid_response_schema_is_rejected(self):
        payload = example("full_b_numeric.json")
        payload["target_item"]["upper"] = -1
        with self.assertRaisesRegex(ValueError, "upper bound"):
            demo.build_messages(payload)
        payload = example("full_b_binary.json")
        payload["target_item"]["options"].append("Maybe")
        with self.assertRaisesRegex(ValueError, "exactly two"):
            demo.build_messages(payload)

    def test_dry_run_without_site_packages_from_another_directory(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/infer_adapter.py"),
            "--input", str(ROOT / "examples/full_b_binary.json"), "--dry-run"],
            cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertFalse(output["model_loaded"])
        self.assertFalse(output["token_count_checked"])
        self.assertEqual(len(output["messages"]), 2)

    def test_base_model_must_be_explicit(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/infer_adapter.py"),
            "--input", str(ROOT / "examples/full_b_binary.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires --base-model", result.stderr)


if __name__ == "__main__":
    unittest.main()
