"""Label-free ablations and train-only retrieved demonstrations for fixed epoch 1.

All target text, options, product prices and original system/output instructions
are preserved. No training runs here. Retrieval is few-shot prompting with the
unadapted base model; its paired no-demo condition uses the same compact own A.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

PROTOCOL = "epoch1-personalization-prompts-v1"
DEFAULT_SHUFFLE_SEED = 20261012


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def target_signature(item):
    """Exact scalar question/condition/options; price remains part of target_text."""
    return digest({key: item.get(key) for key in (
        "item_id", "qid", "family", "target_text", "options", "lower", "upper", "sub_id")})


def messages_for_context(engine, context, item):
    diag = engine.inputs.diag
    require(not diag.has_answers(item), "Target metadata contains an answer payload")
    if item["trained_binary"]:
        return [{"role": "system", "content": engine.base.SYSTEM},
                {"role": "user", "content": diag.PREFIX + context + diag.MARKER
                 + item["target_text"] + diag.SUFFIX}]
    return engine.inputs.messages_for_item(context, item)


def _encode(tokenizer, messages, item, max_input_tokens):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_dict=False, truncation=False)
    require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids),
            "Expected flat token IDs from native chat template")
    cap = 8 if item.get("trained_binary") else 16
    require(len(ids) <= max_input_tokens and len(ids) + cap <= 32768,
            "Target and whole A units exceed the token budget; no truncation is permitted")
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _record(engine, tokenizer, original, messages, *, method, indices, max_input_tokens, **extra):
    encoded = _encode(tokenizer, messages, original, max_input_tokens)
    record = dict(original)
    # These refer to the old selection-tokenizer run and are not meaningful for
    # compact/no-A interventions. The original provenance remains explicit.
    for field in ("selection_prompt_hash", "selection_input_tokens"):
        record.pop(field, None)
    record.update(comparison_protocol=PROTOCOL, comparison_method=method,
                  original_prompt_hash=original["prompt_hash"],
                  prompt_hash=engine.base.digest_text(engine.base.dumps(messages)),
                  input_tokens=len(encoded["input_ids"]), a_unit_indices=list(indices),
                  a_units_selected=len(indices), target_signature=target_signature(original),
                  **extra)
    engine.data.validate_prefix(encoded, record, max_input_tokens=max_input_tokens)
    return record, encoded


def original_messages(engine, record, people):
    pid = str(record["pid"])
    person = people[pid]
    item = person["items"][record["item_id"]]
    require(target_signature(item) == target_signature(record),
            f"Participant target metadata changed: {pid}/{record['item_id']}")
    units = person["a_units"]
    indices = record["a_unit_indices"]
    require(len(indices) == len(set(indices)) and all(type(i) is int and 0 <= i < len(units) for i in indices),
            "Invalid original A-unit indices")
    context = "\n\n".join(engine.base.dumps(units[i][1]) for i in indices)
    return messages_for_context(engine, context, item)


def verify_originals(engine, tokenizer, dataset, manifest, people):
    """Check every saved original hash and every token ID before interventions."""
    require(len(dataset) == len(manifest) and bool(manifest), "Original prefix/manifest count mismatch")
    seen, pids = set(), set()
    for index, record in enumerate(manifest):
        key = (str(record["pid"]), record["item_id"])
        require(key not in seen, f"Duplicate validation scalar: {key}")
        seen.add(key)
        pids.add(key[0])
        messages = original_messages(engine, record, people)
        require(engine.base.digest_text(engine.base.dumps(messages)) == record["prompt_hash"],
                f"Original prompt reconstruction hash differs: {key}")
        encoded = _encode(tokenizer, messages, record, 32760)
        saved = dataset[index]
        require(encoded == {"input_ids": list(saved["input_ids"]),
                            "attention_mask": list(saved["attention_mask"])},
                f"Original prompt reconstruction token IDs differ: {key}")
        engine.data.validate_prefix(encoded, record)
    return {"protocol": PROTOCOL, "verified_examples": len(manifest),
            "verified_participants": len(pids), "all_original_hashes_and_token_ids_match": True}


def participant_derangement(pids, seed=DEFAULT_SHUFFLE_SEED):
    """Uniform permutation conditional on having no fixed points; one donor/person."""
    pids = sorted(set(map(str, pids)))
    require(len(pids) >= 2, "Profile shuffling needs at least two participants")
    rng = random.Random(seed)
    while True:
        donors = list(pids)
        rng.shuffle(donors)
        if all(recipient != donor for recipient, donor in zip(pids, donors)):
            return dict(zip(pids, donors))


def iter_ablation(engine, tokenizer, selection_tokenizer, manifest, people, *,
                  mode, seed=DEFAULT_SHUFFLE_SEED, selection_budget=8192,
                  max_input_tokens=32760):
    """Yield native-generation (manifest, prefix) rows, never retaining full prompts."""
    require(mode in {"no_a", "shuffled_a"}, "Unknown A ablation")
    require(selection_budget == 8192 and max_input_tokens == 32760,
            "Keep the frozen native full-A context policy")
    donor_map = participant_derangement((r["pid"] for r in manifest), seed) if mode == "shuffled_a" else {}
    rendered_cache = {}
    for original in manifest:
        pid, item_id = str(original["pid"]), original["item_id"]
        person, item = people[pid], people[pid]["items"][item_id]
        require(target_signature(original) == target_signature(item), "Recipient target changed")
        if mode == "no_a":
            messages = messages_for_context(engine, "", item)
            yield _record(engine, tokenizer, original, messages, method=mode, indices=[],
                          max_input_tokens=max_input_tokens, context_policy="original-system-and-target-empty-A")
            continue
        donor = donor_map[pid]
        units = people[donor]["a_units"]
        if donor not in rendered_cache:
            rendered_cache[donor] = engine.base.render_a_units(selection_tokenizer, units)
        new, encoded = engine.inputs.prepare_scalar_prompt(
            selection_tokenizer, tokenizer, pid, units, item, person["targets"][item["qid"]],
            selection_budget=selection_budget, max_input_tokens=max_input_tokens,
            rendered_units=rendered_cache[donor])
        require(target_signature(new) == target_signature(original), "Shuffling changed recipient target")
        engine.data.validate_prefix(encoded, new, max_input_tokens=max_input_tokens)
        new.update(comparison_protocol=PROTOCOL, comparison_method=mode,
                   original_prompt_hash=original["prompt_hash"], profile_donor_pid=donor,
                   shuffle_seed=seed, shuffle_scope="global-validation-participant-derangement",
                   target_signature=target_signature(item))
        yield new, encoded


_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
_STOP = frozenset("a an and are as at be by for from how i in is it of on or that the their this to was what when which with would you your please answer question participant section options".split())


def _words(text):
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP}


def compact_a_unit(unit):
    """Complete A response with question, context and option semantics preserved."""
    q = unit["Question"]
    require("Answers" in q, "A profile unit lacks its own earlier answer")
    lines = []
    if unit.get("BlockName"):
        lines.append("Section: " + str(unit["BlockName"]))
    for instruction in unit.get("PrecedingDisplayInstructions", []):
        lines.append("Instructions: " + str(instruction["QuestionText"]))
        for key in ("Options", "Rows", "Columns", "Statements", "Range"):
            if key in instruction:
                lines.append(key + ": " + dumps(instruction[key]))
    lines.append("Question: " + str(q["QuestionText"]))
    for key in ("Options", "Rows", "RowsID", "Columns", "ColumnsID", "Statements", "StatementsID", "Range"):
        if key in q:
            lines.append(key + ": " + dumps(q[key]))
    lines.append("Response: " + dumps(q["Answers"]))
    return "\n".join(lines)


def _render_compact_profile(tokenizer, a_units):
    result = []
    for index, (_, unit) in enumerate(a_units):
        text = compact_a_unit(unit)
        result.append((index, text, len(tokenizer.encode("\n\n" + text, add_special_tokens=False)),
                       _words(unit["Question"]["QuestionText"])))
    return result


def _select_compact(rendered, target, budget):
    words = _words(target)
    ranked = sorted(rendered, key=lambda row: (-len(words & row[3]), row[0]))
    selected, remaining = [], budget
    for index, text, tokens, _ in ranked:
        if tokens <= remaining:
            selected.append((index, text))
            remaining -= tokens
    return selected


def _fewshot_messages(engine, item, own, donors):
    context = "\n\n".join(text for _, text in own)
    messages = messages_for_context(engine, context, item)
    if donors:
        pieces = ["REFERENCE EXAMPLES FROM SIMILAR PARTICIPANTS:\n"
                  "Each reference answered the exact target question shown below, with the same options and conditions. "
                  "Their responses are observations about those participants. Predict the current participant's response."]
        for number, (_, selected, answer, _) in enumerate(donors, 1):
            pieces.append(f"Reference {number} earlier responses (A):\n" +
                          "\n\n".join(text for _, text in selected) +
                          f"\nObserved target response (B): {answer}")
        messages[1]["content"] = "\n\n".join(pieces) + "\n\nCURRENT PARTICIPANT:\n" + messages[1]["content"]
    return messages


def iter_retrieval_pairs(engine, tokenizer, manifest, people, donors, train_labels, train_pids, *,
                         budget=4096, k=3, own_budget=1800, donor_budget=450):
    """Yield ((no-demo manifest,prefix), (few-shot manifest,prefix)) for base Qwen.

    Donor ranking is supplied by an A-only retrieval index. Donor labels must be
    training B. Fixed lexical A snippet selection uses no answer labels. Own A is
    selected once per pair, then any budget-driven shrinkage is shared exactly.
    ``budget`` includes generation reserve, and no target or A unit is truncated.
    """
    from baselines import target_signature as retrieval_signature

    train_pids = set(map(str, train_pids))
    validation_pids = {str(row["pid"]) for row in manifest}
    require(not train_pids & validation_pids, "Training/validation participants overlap")
    require(type(k) is int and 1 <= k <= 3 and 256 <= budget <= 32768,
            "Invalid few-shot settings")
    require(own_budget > 0 and donor_budget > 0, "A snippet budgets must be positive")
    rendered_cache = {}

    def rendered(pid):
        if pid not in rendered_cache:
            rendered_cache[pid] = _render_compact_profile(tokenizer, people[pid]["a_units"])
        return rendered_cache[pid]

    for original in manifest:
        pid, item_id = str(original["pid"]), original["item_id"]
        item = people[pid]["items"][item_id]
        require(target_signature(item) == target_signature(original), "Retrieval changed recipient target")
        # Pricing item IDs are randomized survey positions, not product IDs.
        # A training donor may have the exact product/price at another position.
        donor_entries = [dict(value) if isinstance(value, dict) else {"pid": str(value), "item_id": item_id}
                         for value in donors.get((pid, item_id), [])][:k]
        donor_ids = [str(value["pid"]) for value in donor_entries]
        require(len(donor_ids) == len(set(donor_ids)), "Duplicate retrieved training participant")
        own = _select_compact(rendered(pid), item["target_text"], own_budget)
        require(own, f"No complete own A unit fits compact budget: {pid}/{item_id}")
        chosen = []
        for donor_entry in donor_entries:
            donor = str(donor_entry["pid"])
            source_item_id = str(donor_entry["item_id"])
            require(donor in train_pids and donor != pid, "Retrieved demonstration is not training-only")
            donor_item = people[donor]["items"].get(source_item_id)
            require(donor_item is not None and retrieval_signature(donor_item) == retrieval_signature(item),
                    "Retrieved demonstration has a different target, price, condition or option mapping")
            label = train_labels.get((donor, source_item_id))
            require(engine.data.label_state(label, donor_item) == "valid", "Retrieved B label is invalid")
            if "label" in donor_entry:
                require(donor_entry["label"] == label, "Cached retrieved label differs from source training B")
            selected = _select_compact(rendered(donor), item["target_text"], donor_budget)
            require(selected, f"No complete donor A unit fits compact budget: {donor}/{item_id}")
            chosen.append((donor, selected, engine.data.format_target(label, donor_item), source_item_id))
        cap = engine.data.response_cap(item)
        max_input = min(32760, budget - cap)
        # Adjust only whole snippets. Never silently truncate the target/format.
        while True:
            demo_messages = _fewshot_messages(engine, item, own, chosen)
            length = engine.base.token_count(tokenizer, demo_messages)
            if length <= max_input:
                break
            candidates = [(len(selected), index) for index, (_, selected, _, _) in enumerate(chosen)
                          if len(selected) > 1]
            if candidates:
                _, index = max(candidates)
                chosen[index][1].pop()
            elif len(own) > 1:
                own.pop()
            elif chosen:
                # Rare long target case: retain highest-ranked complete examples.
                chosen.pop()
            else:
                raise ValueError(f"Target plus one complete own A unit exceeds {budget} tokens: {pid}/{item_id}")
        no_messages = _fewshot_messages(engine, item, own, [])
        common = dict(context_policy="compact-whole-A-units-lexical-target-relevance",
                      retrieval_k_requested=k, retrieval_k_available=len(donor_ids),
                      paired_demo_k_used=len(chosen), paired_demo_donor_pids=[entry[0] for entry in chosen],
                      paired_demo_source_items=[{"pid": entry[0], "item_id": entry[3]} for entry in chosen],
                      paired_demo_donor_a_indices={entry[0]: [index for index, _ in entry[1]] for entry in chosen},
                      own_a_budget=own_budget, donor_a_budget=donor_budget,
                      total_context_budget=budget, base_model_without_adapter=True,
                      matched_own_a_prompt_hash=digest([text for _, text in own]),
                      own_a_selection_uses_labels=False)
        indices = [index for index, _ in own]
        no_record, no_row = _record(engine, tokenizer, original, no_messages,
            method="base_compact_no_demo", indices=indices, max_input_tokens=max_input,
            demonstrations_in_prompt=0, retrieval_k_used=0, retrieval_donor_pids=[],
            retrieval_source_items=[], **common)
        demo_record, demo_row = _record(engine, tokenizer, original, demo_messages,
            method="base_retrieved_demo", indices=indices, max_input_tokens=max_input,
            demonstrations_in_prompt=len(chosen), retrieval_k_used=len(chosen),
            retrieval_donor_pids=[entry[0] for entry in chosen],
            retrieval_source_items=[{"pid": entry[0], "item_id": entry[3]} for entry in chosen], **common)
        yield (no_record, no_row), (demo_record, demo_row)
