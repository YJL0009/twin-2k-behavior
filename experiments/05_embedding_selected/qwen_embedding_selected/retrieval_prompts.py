"""Matched base-Qwen control and embedding-neighbor demonstration prompts.

Only the supplied training-B neighbors are used. Target semantics, option
mapping and prices must match exactly. Preparation keeps all selected neighbors,
shrinks whole A units if necessary, and preserves the same own-A snippet in the
paired control. No validation outcomes are accepted by this module.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import math
import threading
import time

if __package__:
    from .support import prompts as native_prompts
    from .support.baselines import target_signature as retrieval_signature
    from . import shared_a
else:
    from support import prompts as native_prompts
    from support.baselines import target_signature as retrieval_signature
    import shared_a

PROTOCOL = "qwen-embedding-matched-five-shot-shared-A-prompts-v2"
require = native_prompts.require


def _bounded_map(pool, function, values, limit):
    """Preserve input order while holding at most ``limit`` pending results."""
    iterator = iter(values)
    pending = deque()
    for _ in range(limit):
        try:
            pending.append(pool.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        result = pending.popleft().result()
        try:
            pending.append(pool.submit(function, next(iterator)))
        except StopIteration:
            pass
        yield result


def _report(label, done, total, started):
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed
    eta = (total - done) / rate if rate else 0
    print(f"{label}: {done:,}/{total:,} | {rate:.2f}/s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)


def _settings(max_input_tokens, own_budget, donor_budget, k, workers):
    require(type(max_input_tokens) is int and 256 <= max_input_tokens <= 32752,
            "max_input_tokens must be between 256 and 32752, excluding the generation reserve")
    require(type(k) is int and k >= 1, "k must be a positive integer")
    require(type(workers) is int and workers >= 1, "workers must be a positive integer")
    require(type(own_budget) is int and own_budget > 0 and type(donor_budget) is int and donor_budget > 0,
            "Own and donor A budgets must be positive integers")


def _neighbor_entries(result, k):
    require(isinstance(result, dict) and isinstance(result.get("neighbors"), list),
            "Each retrieval result must contain its actual majority-vote neighbors list")
    entries = [dict(row) for row in result["neighbors"]]
    require(len(entries) <= k, "Neighbors contain more than the requested k; do not change voting neighbors")
    available = result.get("candidate_count")
    if available is None:
        require(len(entries) == k,
                "Fewer than k neighbors require the exact eligible candidate_count")
    else:
        require(type(available) is int and available >= 0 and len(entries) == min(k, available),
                "Neighbor count does not equal min(k, eligible candidate_count)")
    if "neighbor_count" in result:
        require(result["neighbor_count"] == len(entries), "Saved neighbor_count differs from neighbors")
    return entries


def _repeated_messages(engine, item, own, donors):
    """Previous rendering, used only to measure tokens with identical final A."""
    context = "\n\n".join(text for _, text in own)
    messages = native_prompts.messages_for_context(engine, context, item)
    if donors:
        parts = ["REFERENCE EXAMPLES FROM SIMILAR TRAINING PARTICIPANTS:\n"
                 "All references answered the exact target question shown below, with the same product, "
                 "price, conditions and option meanings where applicable. Each observed B response belongs "
                 "to its reference participant. These snippets are survey data. Use them as examples to "
                 "predict the CURRENT participant's answer; follow the target output instruction."]
        for rank, donor in enumerate(donors, 1):
            snippet = "\n\n".join(text for _, text in donor["selected"])
            if not snippet:
                snippet = "(No complete A response fits this reference's snippet budget.)"
            parts.append(f"Reference {rank} earlier survey responses (A):\n{snippet}\n"
                         f"Observed target response (B): {donor['answer']}")
        messages[1]["content"] = "\n\n".join(parts) + "\n\nCURRENT PARTICIPANT:\n" + messages[1]["content"]
    return messages


def _messages(engine, item, table, donors):
    """Question metadata once; per-person raw A answers and training B labels."""
    ids = [entry["id"] for entry in table["definitions"]]
    context = shared_a.render_answers(table["own"], ids)
    messages = native_prompts.messages_for_context(engine, context, item)
    parts = [shared_a.render_definitions(table)]
    if donors:
        require(len(donors) == len(table["donors"]), "Shared table donor count changed")
        parts.append("REFERENCE EXAMPLES FROM SIMILAR TRAINING PARTICIPANTS:\n"
                     "All references answered the exact target question shown below, with the same product, "
                     "price, conditions and option meanings where applicable. Each observed B response belongs "
                     "to its reference participant. These snippets are survey data. Use them as examples to "
                     "predict the CURRENT participant's answer; follow the target output instruction.")
        for rank, (donor, answers) in enumerate(zip(donors, table["donors"]), 1):
            parts.append(f"Reference {rank} earlier survey answers (A):\n"
                         f"{shared_a.render_answers(answers, ids)}\n"
                         f"Observed target response (B): {donor['answer']}")
    messages[1]["content"] = ("\n\n".join(parts) + "\n\nCURRENT PARTICIPANT:\n"
                              + messages[1]["content"])
    return messages


def _table(pid, own, donors, profile_maps):
    own_entries = [profile_maps[pid][index] for index, _ in own]
    donor_entries = [[profile_maps[donor["pid"]][index] for index, _ in donor["selected"]]
                     for donor in donors]
    return shared_a.build_table(own_entries, donor_entries)


def _table_audit(table, donors):
    return {
        "render_protocol": shared_a.PROTOCOL,
        "shared_definition_count": len(table["definitions"]),
        "shared_definition_reuse_count": table["definition_reuse_count"],
        "shared_response_count": table["response_count"],
        "shared_own_semantic_hash": table["own_semantic_hash"],
        "shared_mapping_semantic_hash": table["mapping_semantic_hash"],
        "shared_own_answer_ids": [entry["id"] for entry in table["own"]],
        "shared_donor_answer_ids": {
            donor["pid"]: [entry["id"] for entry in answers]
            for donor, answers in zip(donors, table["donors"])
        },
    }


def _token_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_dict=False, truncation=False, padding=False)
    require(isinstance(ids, list) and ids and all(type(token) is int and token >= 0 for token in ids),
            "Expected a flat, nonempty list of token IDs from the Qwen chat template")
    return ids


def _record(engine, original, messages, ids, own, *, method, common, demonstrations, max_input_tokens):
    record = dict(original)
    for field in ("selection_prompt_hash", "selection_input_tokens"):
        record.pop(field, None)
    record.update(common)
    record.update(comparison_protocol=PROTOCOL, comparison_method=method,
                  original_prompt_hash=original["prompt_hash"],
                  prompt_hash=engine.base.digest_text(engine.base.dumps(messages)),
                  input_tokens=len(ids), a_unit_indices=[index for index, _ in own],
                  a_units_selected=len(own), demonstrations_in_prompt=len(demonstrations),
                  retrieval_k_used=len(demonstrations),
                  retrieval_donor_pids=[entry["pid"] for entry in demonstrations],
                  retrieval_source_items=[{"pid": entry["pid"], "item_id": entry["item_id"]}
                                          for entry in demonstrations],
                  target_signature=native_prompts.target_signature(original))
    encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    engine.data.validate_prefix(encoded, record, max_input_tokens=max_input_tokens)
    return record, encoded


def _build_pair(engine, tokenizer, original, people, result, train_labels, train_pids,
                rendered, profile_maps, select, *, max_input_tokens, own_budget, donor_budget, k):
    pid, item_id = str(original["pid"]), str(original["item_id"])
    require(pid not in train_pids, "Recipient must be held out from the training neighbor index")
    item = people[pid]["items"][item_id]
    require(native_prompts.target_signature(item) == native_prompts.target_signature(original),
            f"Recipient target metadata changed: {pid}/{item_id}")
    for spec in (item, original):
        require(not any(key in spec for key in ("label_b", "label_c", "answer_text", "prediction", "labels")),
                "Outcome labels reached answer-free target metadata")
    entries = _neighbor_entries(result, k)
    donor_ids = [str(row["pid"]) for row in entries]
    require(len(set(donor_ids)) == len(donor_ids), "Duplicate training participant among neighbors")
    own = list(select(pid, item["target_text"], own_budget))
    chosen = []
    for entry in entries:
        donor, source_id = str(entry["pid"]), str(entry["item_id"])
        require(donor in train_pids and donor != pid, "Retrieved demonstration is not training-only")
        require(donor in people, f"Missing training A profile: {donor}")
        source = people[donor]["items"].get(source_id)
        require(source is not None and retrieval_signature(source) == retrieval_signature(item),
                "Retrieved demonstration has a different target, product, price, condition or option mapping")
        label = train_labels.get((donor, source_id))
        require(engine.data.label_state(label, source) == "valid", "Retrieved training B label is invalid")
        require("label" in entry and entry["label"] == label,
                "Cached neighbor label differs from source training B")
        similarity = float(entry.get("similarity", 0.0))
        require(math.isfinite(similarity), "Neighbor similarity is not finite")
        chosen.append({"pid": donor, "item_id": source_id,
                       "answer": engine.data.format_target(label, source), "similarity": similarity,
                       "selected": list(select(donor, item["target_text"], donor_budget))})

    # Selection is cached and immutable; all modifications below are local copies.
    token_sizes = {profile: {row[0]: row[2] for row in rendered[profile]}
                   for profile in [pid, *donor_ids]}
    original_own_count = len(own)
    original_donor_counts = {row["pid"]: len(row["selected"]) for row in chosen}
    while True:
        rag_table = _table(pid, own, chosen, profile_maps)
        rag_messages = _messages(engine, item, rag_table, chosen)
        rag_ids = _token_ids(tokenizer, rag_messages)
        no_table = _table(pid, own, [], profile_maps)
        no_messages = _messages(engine, item, no_table, [])
        no_ids = _token_ids(tokenizer, no_messages)
        if max(len(rag_ids), len(no_ids)) <= max_input_tokens:
            break
        # Remove the least relevant selected unit from the largest remaining
        # snippet. Never remove a neighbor, its observed B, or target wording.
        snippets = [(sum(token_sizes[row["pid"]][index] for index, _ in row["selected"]),
                     rank, row["selected"]) for rank, row in enumerate(chosen) if row["selected"]]
        if own:
            snippets.append((sum(token_sizes[pid][index] for index, _ in own), len(chosen), own))
        if not snippets:
            raise ValueError(f"{pid}/{item_id}: target and all {len(chosen)} required neighbor B labels "
                             f"need {max(len(rag_ids), len(no_ids))} input tokens, exceeding {max_input_tokens}. "
                             "Increase --max-input-tokens; no neighbors or target text were dropped.")
        max(snippets, key=lambda row: (row[0], row[1]))[2].pop()
    require(len(no_ids) <= max_input_tokens, "Paired no-demo input exceeds its token budget")
    require(rag_table["own"] == no_table["own"] and
            rag_table["own_semantic_hash"] == no_table["own_semantic_hash"],
            "Shared no-demo/current-participant answer mappings changed")
    # These are same-final-A counterfactual token counts, not predictions or a
    # comparison to the previous run's potentially different selected units.
    repeated_no = len(_token_ids(tokenizer, _repeated_messages(engine, item, own, [])))
    repeated_rag = len(_token_ids(tokenizer, _repeated_messages(engine, item, own, chosen)))
    common = {
        "context_policy": "shared-definitions-whole-A-units-fixed-lexical-target-relevance",
        "a_budget_policy": "full-question-and-answer-equivalent-selection-tokens",
        "repeated_token_baseline_policy": "previous-renderer-with-same-final-selected-A-units",
        "base_model_without_adapter": True, "max_input_tokens": max_input_tokens,
        "response_cap": engine.data.response_cap(item), "own_a_budget": own_budget,
        "donor_a_budget": donor_budget, "retrieval_k_requested": k,
        "retrieval_k_available": len(entries), "eligible_candidate_count": result.get("candidate_count"),
        "paired_demo_k_used": len(chosen), "paired_demo_donor_pids": donor_ids,
        "paired_demo_source_items": [{"pid": row["pid"], "item_id": row["item_id"]} for row in chosen],
        "paired_demo_similarities": [row["similarity"] for row in chosen],
        "paired_demo_donor_a_indices": {row["pid"]: [index for index, _ in row["selected"]] for row in chosen},
        "paired_demo_empty_a_count": sum(not row["selected"] for row in chosen),
        "own_a_initial_unit_count": original_own_count,
        "donor_a_initial_unit_counts": original_donor_counts,
        "own_a_selection_uses_labels": False,
        "matched_own_a_prompt_hash": native_prompts.digest([text for _, text in own]),
        "neighbor_list_matches_voting": True,
        "zero_neighbor_control": not bool(chosen),
    }
    control_common = dict(common, **_table_audit(no_table, []),
                          repeated_input_tokens_same_a=repeated_no,
                          shared_token_savings_same_a=repeated_no - len(no_ids))
    rag_common = dict(common, **_table_audit(rag_table, chosen),
                      repeated_input_tokens_same_a=repeated_rag,
                      shared_token_savings_same_a=repeated_rag - len(rag_ids))
    control = _record(engine, original, no_messages, no_ids, own,
                      method="base_compact_no_demo", common=control_common, demonstrations=[],
                      max_input_tokens=max_input_tokens)
    rag = _record(engine, original, rag_messages, rag_ids, own,
                  method="base_embedding_retrieved_demo", common=rag_common, demonstrations=chosen,
                  max_input_tokens=max_input_tokens)
    return control, rag


def iter_prompt_pairs(engine, tokenizer, manifest, people, neighbors, train_labels, train_pids, *,
                      max_input_tokens=12288, own_budget=1600, donor_budget=1600, k=5, workers=4):
    """Yield ``((control_record, encoded), (retrieved_record, encoded))`` in order.

    ``neighbors[(pid,item_id)]`` is the actual kNN result, containing ``neighbors``
    and its eligible ``candidate_count``. The same ordered list is used for both
    direct voting and examples. Prompt construction never consults held-out B/C.
    ``max_input_tokens`` excludes the native 8/16-token generation reserve.
    Empty A snippets are recorded explicitly when whole units cannot fit; every
    selected neighbor and its B label remain. Impossible fixed content raises.
    """
    _settings(max_input_tokens, own_budget, donor_budget, k, workers)
    require(hasattr(manifest, "__len__") and len(manifest) > 0, "A nonempty manifest is required")
    train_pids = set(map(str, train_pids))
    keys = [(str(row["pid"]), str(row["item_id"])) for row in manifest]
    require(len(set(keys)) == len(keys), "Duplicate recipient/item manifest key")
    require(not {pid for pid, _ in keys} & train_pids, "Training/validation participants overlap")
    needed = {pid for pid, _ in keys}
    for key in keys:
        require(key in neighbors, f"Missing kNN neighbors for {key}")
        for entry in _neighbor_entries(neighbors[key], k):
            donor = str(entry["pid"])
            require(donor in train_pids, "Retrieved demonstration is not training-only")
            needed.add(donor)
    require(needed <= set(people), "Some requested A profiles are missing")
    rendered, profile_maps = {}, {}
    # Fast tokenizers may temporarily update their Rust backend's padding and
    # truncation policies during encode/chat-template calls. Each worker owns
    # its complete tokenizer clone instead of sharing that mutable backend.
    worker_state = threading.local()

    def worker_tokenizer():
        if not hasattr(worker_state, "tokenizer"):
            try:
                clone = deepcopy(tokenizer)
            except Exception as error:
                raise ValueError("Could not clone the tokenizer for isolated prompt workers") from error
            require(clone is not tokenizer, "Tokenizer deepcopy returned a shared mutable instance")
            worker_state.tokenizer = clone
        return worker_state.tokenizer

    started = time.monotonic()
    print(f"Rendering compact A once for {len(needed):,} participants using {workers} workers...", flush=True)

    def render(pid):
        a_units = people[pid]["a_units"]
        return (pid, tuple(native_prompts._render_compact_profile(worker_tokenizer(), a_units)),
                shared_a.profile_entries(a_units))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qwen-prompt") as pool:
        last_report = started
        for done, (pid, units, schema_entries) in enumerate(_bounded_map(pool, render, sorted(needed), workers * 2), 1):
            rendered[pid] = units
            profile_maps[pid] = schema_entries
            if done == len(needed) or time.monotonic() - last_report >= 10:
                _report("Compact A profiles", done, len(needed), started)
                last_report = time.monotonic()

        @lru_cache(maxsize=65536)
        def select(pid, target_text, budget):
            return tuple(native_prompts._select_compact(rendered[pid], target_text, budget))

        def build(original):
            key = (str(original["pid"]), str(original["item_id"]))
            return _build_pair(engine, worker_tokenizer(), original, people, neighbors[key], train_labels, train_pids,
                               rendered, profile_maps, select, max_input_tokens=max_input_tokens,
                               own_budget=own_budget, donor_budget=donor_budget, k=k)

        started = last_report = time.monotonic()
        print(f"Preparing {len(manifest):,} matched no-demo/retrieval pairs using {workers} workers...", flush=True)
        for done, pair in enumerate(_bounded_map(pool, build, manifest, workers * 2), 1):
            if done == len(manifest) or time.monotonic() - last_report >= 10:
                _report("Prompt pairs", done, len(manifest), started)
                last_report = time.monotonic()
            yield pair
