#!/usr/bin/env python3
"""Run the published Full-B adapter on one supplied A excerpt and scalar target.

Prompt rendering uses the experiment's existing builders. This portable demo
does not reconstruct historical A selection, run provenance, or validation
scores. --dry-run requires only Python's standard library and loads no weights.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = REPO_ROOT / "adapters/full-b-lora/epoch-1"
CATEGORICAL = {"mc", "ordinal", "matrix_nominal", "matrix_ordinal"}
NUMERIC = {"bounded_numeric", "unbounded_numeric"}
ANSWER_KEYS = {"Answers", "Answer", "SelectedByPosition", "SelectedText",
               "label_b", "label_c", "answer_text", "prediction", "labels"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=1)
def native_modules():
    """Load pure-Python prompt/schema modules; no training/evaluator imports."""
    engine = REPO_ROOT / "experiments/02_full_b_lora/qwen_full_b_lora_scripts"
    sys.path.insert(0, str(engine))
    names = ("run_liquid_baseline", "full_b_data", "full_b_schema")
    loaded = [importlib.import_module(name) for name in names]
    require(all(Path(module.__file__).resolve().parent == engine for module in loaded),
            "An unrelated experiment module was imported; use a fresh Python process")
    prompt_path = (REPO_ROOT / "experiments/03_epoch1_comparisons/epoch1_comparisons/prompts.py")
    spec = importlib.util.spec_from_file_location("adapter_demo_prompt_rules", prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(base=loaded[0], inputs=loaded[1], schema=loaded[2], prompts=module)


def has_target_answer(value):
    if isinstance(value, dict):
        return any(key in ANSWER_KEYS or has_target_answer(child) for key, child in value.items())
    return isinstance(value, list) and any(has_target_answer(child) for child in value)


def validate_input(payload):
    require(isinstance(payload, dict), "Input must be a JSON object")
    require(set(payload) <= {"description", "a_units", "target_item"},
            "Use only description, a_units and target_item; do not supply B/C labels")
    units = payload.get("a_units")
    require(isinstance(units, list) and bool(units), "a_units must contain at least one complete A unit")
    for index, unit in enumerate(units):
        require(isinstance(unit, dict) and isinstance(unit.get("Question"), dict),
                f"A unit {index} must contain a Question object")
        question = unit["Question"]
        require(isinstance(question.get("QuestionText"), str) and bool(question["QuestionText"].strip())
                and "Answers" in question, f"A unit {index} needs QuestionText and its own earlier Answers")
        answer = question["Answers"]
        require(answer is not None and answer != {} and answer != []
                and not (isinstance(answer, str) and not answer.strip()),
                f"A unit {index} must have a nonempty earlier answer")
    item = payload.get("target_item")
    require(isinstance(item, dict) and not has_target_answer(item),
            "target_item must contain answer-free metadata only")
    require(isinstance(item.get("target_text"), str) and bool(item["target_text"].strip()),
            "target_item.target_text must contain the full question and any options/conditions")
    require(type(item.get("trained_binary")) is bool,
            "target_item.trained_binary must explicitly be true or false")
    family = item.get("family")
    require(isinstance(family, str) and family in CATEGORICAL | NUMERIC,
            "Unsupported scalar response family")
    if family in CATEGORICAL:
        options = item.get("options")
        require(isinstance(options, list) and len(options) >= 2 and
                all(isinstance(option, str) and bool(option.strip()) for option in options),
                "Categorical items require the original ordered options list")
        require(not item["trained_binary"] or (family == "mc" and len(options) == 2),
                "The original binary format requires family=mc and exactly two options")
    else:
        require(not item["trained_binary"], "Numeric items cannot use the original binary format")
        lower, upper = item.get("lower"), item.get("upper")
        require(type(lower) in (int, float) and lower == 0,
                "Full-B numeric items require lower=0")
        if family == "bounded_numeric":
            require(type(upper) in (int, float) and math.isfinite(upper) and upper > lower,
                    "Bounded numeric items need a finite upper bound greater than zero")
        else:
            require(upper is None, "Unbounded numeric items must have upper=null or omit it")
    return units, item


def build_messages(payload):
    units, item = validate_input(payload)
    engine = native_modules()
    # The Full-B training/evaluation prompts serialize whole A units this way.
    context = "\n\n".join(engine.base.dumps(unit) for unit in units)
    messages = engine.prompts.messages_for_context(engine, context, item)
    cap = 8 if item["trained_binary"] else 16
    return messages, item, cap


def device_and_dtype(torch, requested_device, requested_dtype):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    require(device != "cuda" or torch.cuda.is_available(), "CUDA was requested but is unavailable")
    dtype_name = requested_dtype
    if dtype_name == "auto":
        dtype_name = "float32" if device == "cpu" else (
            "bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    require(not (device == "cpu" and dtype_name != "float32"),
            "Use float32 for this CPU demo (or choose --device cuda)")
    require(not (device == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported()),
            "This GPU does not support bfloat16; use float16 or float32")
    return device, getattr(torch, dtype_name), dtype_name


def generate(args, messages, item, cap):
    require(bool(args.base_model),
            "Inference requires --base-model: a local Qwen2.5-0.5B-Instruct directory or "
            "Qwen/Qwen2.5-0.5B-Instruct (downloads/caches external base weights)")
    adapter = args.adapter.resolve()
    for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json",
                     "tokenizer.json", "chat_template.jinja"):
        require((adapter / filename).is_file(), f"Missing adapter asset: {adapter / filename}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ValueError("Install the inference dependencies in docs/INFERENCE.md") from error
    device, dtype, dtype_name = device_and_dtype(torch, args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True,
                                              trust_remote_code=False, padding_side="left")
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", truncation=False, padding=False)
    input_tokens = int(encoded["input_ids"].shape[-1])
    require(input_tokens <= args.max_input_tokens and input_tokens + cap <= 32768,
            f"Prompt has {input_tokens} tokens; supply a shorter whole-unit A excerpt. "
            "No input was silently truncated")
    # Load the public base explicitly: adapter_config.json retains an original
    # server path which is provenance, not a portable base-model location.
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype,
        local_files_only=args.local_files_only, trust_remote_code=False, attn_implementation="sdpa")
    require(base.config.model_type == "qwen2" and
            sum(parameter.numel() for parameter in base.parameters()) == 494_032_768,
            "Use the unmodified Qwen2.5-0.5B-Instruct base model")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, local_files_only=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=cap,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, input_tokens:], skip_special_tokens=True).strip()
    parsed = native_modules().schema.parse_prediction(raw, item)
    return {"prediction_text": raw, "parsed_prediction": parsed, "valid_output": parsed is not None,
            "family": item["family"], "input_tokens": input_tokens, "max_new_tokens": cap,
            "base_model": args.base_model, "adapter": str(adapter), "device": device,
            "base_dtype": dtype_name, "evaluation_performed": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path, help="JSON with a_units and one target_item")
    result.add_argument("--dry-run", action="store_true", help="Print exact messages; load no model or tokenizer")
    result.add_argument("--base-model", help="Local Qwen base directory or Qwen/Qwen2.5-0.5B-Instruct")
    result.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    result.add_argument("--local-files-only", action="store_true", help="Forbid base-model downloads")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    result.add_argument("--max-input-tokens", type=int, default=32752,
                        help="Reject longer prompts; never truncate supplied A units")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require(1 <= args.max_input_tokens <= 32752, "max-input-tokens must be between 1 and 32752")
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        messages, item, cap = build_messages(payload)
        if args.dry_run:
            result = {"messages": messages, "family": item["family"], "max_new_tokens": cap,
                      "model_loaded": False, "token_count_checked": False,
                      "note": "All supplied A units are included; historical A selection is not reconstructed."}
        else:
            result = generate(args, messages, item, cap)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("valid_output") is False else 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Adapter inference stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
