#!/usr/bin/env python3
"""Small, label-free diagnostics for Liquid's saved identifier-shaped answers.

Run with the Liquid Transformers 5.2 environment. Reads only prepared prompts
and raw predictions; never reads participant answer labels or computes accuracy.
Plain targets retain substantive text, display instructions and ordered options.
No saved baseline files are changed. Each run creates a new diagnostic folder.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PREFIX = "EARLIER SURVEY RESPONSES (A):\n"
MARKER = "\n\nTARGET QUESTION:\n"
SUFFIX = ("\n\nThe Options list is ordered: first option = 1, second option = 2. "
          "Predict the participant's answer. Return only 1 or 2.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def has_answers(value):
    if isinstance(value, dict):
        return any(k in {"Answers", "Answer", "SelectedByPosition", "SelectedText"}
                   or has_answers(v) for k, v in value.items())
    return isinstance(value, list) and any(has_answers(v) for v in value)


def split_prompt(record):
    messages = record["messages"]
    require([m["role"] for m in messages] == ["system", "user"], "Unexpected message roles")
    text = messages[1]["content"]
    require(text.startswith(PREFIX) and text.endswith(SUFFIX), "Unexpected saved prompt format")
    body = text[len(PREFIX):-len(SUFFIX)]
    require(body.count(MARKER) == 1, "Ambiguous target boundary")
    context, target_json = body.split(MARKER)
    target = json.loads(target_json)
    require(not has_answers(target), "Target contains an answer field")
    require(target["Question"]["QuestionID"] == record["qid"], "Target QID mismatch")
    require(isinstance(target["Question"]["Options"], list)
            and len(target["Question"]["Options"]) == 2, "Expected two ordered options")
    units = context.split("\n\n") if context else []
    for unit in units:
        require(isinstance(json.loads(unit).get("Question"), dict), "Invalid complete A unit")
    require(PREFIX + "\n\n".join(units) + MARKER + dumps(target) + SUFFIX == text,
            "Parsing did not reproduce the original prompt exactly")
    require(sha(dumps(messages)) == record["prompt_hash"], "Saved prompt hash mismatch")
    return messages[0]["content"], units, target


def plain_target(target):
    lines = []
    if target.get("BlockName"):
        lines.append("Section: " + str(target["BlockName"]))
    for instruction in target.get("PrecedingDisplayInstructions", []):
        require(isinstance(instruction.get("QuestionText"), str), "Display instruction lacks text")
        lines.append("Instructions:\n" + instruction["QuestionText"])
        for key in ("Options", "Rows", "Columns", "Statements", "Range"):
            if key in instruction:
                lines.append(key + ": " + dumps(instruction[key]))
    question = target["Question"]
    require(isinstance(question["QuestionText"], str), "Target lacks question text")
    lines.append(question["QuestionText"])
    for key in ("Rows", "Columns", "Statements", "Range"):
        if key in question:
            lines.append(key + ": " + dumps(question[key]))
    lines.append("Options:")
    for number, option in enumerate(question["Options"], 1):
        lines.append(f"{number}. " + (option if isinstance(option, str) else dumps(option)))
    return "\n\n".join(lines)


def make_case(name, record, context="full", plain=False, max_new=64, cache=True):
    system, units, target = split_prompt(record)
    chosen = units if context == "full" else units[:3] if context == "three" else []
    target_text = plain_target(target) if plain else dumps(target)
    messages = [{"role": "system", "content": system}, {"role": "user", "content":
                PREFIX + "\n\n".join(chosen) + MARKER + target_text + SUFFIX}]
    return {"name": name, "messages": messages, "source_pid": record["pid"],
            "source_qid": record["qid"], "source_prompt_hash": record["prompt_hash"],
            "a_units": len(chosen), "max_new_tokens": max_new, "use_cache": cache,
            "source_input_tokens": record["input_tokens"] if context == "full" and not plain else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/twin_qwen"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/LFM2.5-350M"))
    parser.add_argument("--run", type=Path, default=Path("outputs/liquid_baseline_tf52_v2"))
    args = parser.parse_args()
    root, model_path = args.root.resolve(), args.model.resolve()
    run = args.run if args.run.is_absolute() else root / args.run
    prepared = read_jsonl(run / "prepared_prompts.jsonl")
    raw = read_jsonl(run / "raw_predictions.jsonl")
    selected = {}
    for qid in ("QID196", "QID163"):
        selected[qid] = next((p for p in prepared if p["qid"] == qid), None)
        require(selected[qid] is not None, f"No saved {qid} prompt")
        split_prompt(selected[qid])
    cases = []
    for digit, prompt in (("1", "Return exactly the digit 1. Nothing else."),
                          ("2", "Return exactly the digit 2. Nothing else."),
                          ("4", "What is 2 + 2? Return only the number.")):
        cases.append({"name": "control_" + digit, "messages": [
            {"role": "system", "content": "Follow the user's instructions."},
            {"role": "user", "content": prompt}], "max_new_tokens": 64, "use_cache": True,
            "expected_control": digit, "a_units": 0})
    p196, p163 = selected["QID196"], selected["QID163"]
    cases.extend([make_case("q196_full_json", p196),
                  make_case("q196_no_a_json", p196, context="none"),
                  make_case("q196_full_a_plain_target", p196, plain=True),
                  make_case("q196_three_a_plain_target", p196, context="three", plain=True),
                  make_case("q196_no_a_plain_target", p196, context="none", plain=True),
                  make_case("q163_full_json", p163),
                  make_case("q163_full_a_plain_target", p163, plain=True),
                  make_case("q196_full_json_no_cache", p196, max_new=8, cache=False)])
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    require(torch.cuda.is_available(), "CUDA unavailable")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    require(tokenizer.eos_token_id is not None, "Missing EOS token")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "outputs" / ("liquid_diagnostics_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    config = {"model": str(model_path), "source_run": str(run), "python": sys.version,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "tokenizers")},
              "model_type": model.config.model_type, "model_class": type(model).__name__,
              "dtype": str(model.dtype),
              "parameters": sum(p.numel() for p in model.parameters()),
              "tokenizer_class": type(tokenizer).__name__,
              "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id,
              "label_access": False, "greedy": True,
              "download_revision": (model_path / "download_revision.txt").read_text().strip()
                  if (model_path / "download_revision.txt").exists() else None,
              "cases": [{k: v for k, v in case.items() if k != "messages"} for case in cases],
              "source_examples": [{"pid": p["pid"], "qid": p["qid"], "prompt_hash": p["prompt_hash"],
                  "saved_raw_output": next((r["raw_output"] for r in raw
                       if (r["pid"], r["qid"], r["prompt_hash"]) ==
                          (p["pid"], p["qid"], p["prompt_hash"])), None)} for p in selected.values()]}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: config[key] for key in
          ("versions", "model_class", "model_type", "parameters", "dtype", "eos", "pad")},
          indent=2), flush=True)
    print("Cases:", len(cases), "| Saved examples:",
          dumps(config["source_examples"]), flush=True)
    results = {}
    with (out / "results.jsonl").open("x", encoding="utf-8") as handle:
        for case in cases:
            inputs = tokenizer.apply_chat_template(case["messages"], tokenize=True,
                     add_generation_prompt=True, return_dict=True, return_tensors="pt", truncation=False).to("cuda")
            width = inputs["input_ids"].shape[1]
            require(width == int(inputs["attention_mask"].sum()), "Unexpected padding in single example")
            if case.get("source_input_tokens") is not None:
                require(width == case["source_input_tokens"], "Saved/input token count mismatch")
            require(width + case["max_new_tokens"] <= model.config.max_position_embeddings,
                    "Diagnostic exceeds model context window")
            generation = GenerationConfig(do_sample=False, max_new_tokens=case["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id, use_cache=case["use_cache"],
                return_dict_in_generate=True, output_scores=True)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation)
            ids = generated.sequences[0, width:].tolist()
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            first = generated.scores[0][0].float()
            finite = bool(torch.isfinite(first).all())
            top = []
            if finite:
                probabilities, indices = first.softmax(dim=-1).topk(5)
                top = [{"token_id": int(i), "token": tokenizer.decode([int(i)], skip_special_tokens=False),
                        "probability": float(p)} for i, p in zip(indices.tolist(), probabilities.tolist())]
            eos = bool(ids and ids[-1] == tokenizer.eos_token_id)
            result = {k: v for k, v in case.items() if k != "messages"}
            result.update(prompt_sha256=sha(dumps(case["messages"])), input_tokens=width,
                generated_ids=ids, generated_tokens=len(ids), raw_output=answer,
                raw_repr=repr(answer), special_repr=repr(tokenizer.decode(ids, skip_special_tokens=False)),
                ended_with_eos=eos, hit_token_cap=len(ids) == case["max_new_tokens"] and not eos,
                strict_valid=bool(re.fullmatch(r"\s*[12]\s*", answer)), first_scores_finite=finite,
                first_token_top5=top)
            if "expected_control" in case:
                result["control_pass"] = answer.strip() == case["expected_control"]
            handle.write(dumps(result) + "\n")
            handle.flush()
            results[case["name"]] = result
            print(f"\n{case['name']} | input={width} | generated={len(ids)} "
                  f"| EOS={eos} | cap={result['hit_token_cap']} "
                  f"| valid_1_or_2={result['strict_valid']}"
                  + (f" | control_pass={result['control_pass']}"
                     if "control_pass" in result else ""), flush=True)
            print("  raw:", result["raw_repr"], flush=True)
            print("  with_special_tokens:", result["special_repr"], flush=True)
            print("  generated_ids:", ids, flush=True)
            print("  finite_first_scores:", finite, "| first_token_top5:",
                  dumps(top), flush=True)
            del generated, inputs
    cached = results["q196_full_json"]["generated_ids"][:8]
    uncached = results["q196_full_json_no_cache"]["generated_ids"]
    comparison = {"cached_first_up_to_8_ids": cached, "uncached_up_to_8_ids": uncached,
                  "identical": cached == uncached,
                  "note": "A mismatch warrants investigation; this alone does not establish a cache defect."}
    (out / "cache_comparison.json").write_text(dumps(comparison) + "\n", encoding="utf-8")
    print("Cache comparison:", dumps(comparison), flush=True)
    print("Saved diagnostics:", out, flush=True)


if __name__ == "__main__":
    main()
