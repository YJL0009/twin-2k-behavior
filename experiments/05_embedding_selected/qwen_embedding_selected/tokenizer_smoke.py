"""CPU-only checks of the actual local tokenizers before model weights load."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from embedding_index import QUERY_PREFIX, _require, _special_token_layout, prepare_profile_texts


def check_tokenizers(embedding_model, generator_model, *, embedding_max_tokens=2048):
    from transformers import AutoTokenizer

    embedding = AutoTokenizer.from_pretrained(str(embedding_model), local_files_only=True,
                                             trust_remote_code=False, padding_side="left")
    _require(embedding.pad_token_id is not None, "Embedding tokenizer needs its downloaded pad token")
    layout = _special_token_layout(embedding)
    clone = deepcopy(embedding)
    _require(_special_token_layout(clone) == layout, "Worker tokenizer clone changed special-token behavior")
    texts = ["Question: Which activity? Options: reading, running. Response: reading.",
             "Question: Explain your preference. Response: " + "A longer multilingual survey answer. 问卷回答。 " * 400]
    payload = embedding(["\n\n" + text for text in texts], add_special_tokens=False,
                        padding=False, truncation=False, return_attention_mask=False)["input_ids"]
    expected = [token for unit in payload for token in unit]
    caps = sorted({min(256, embedding_max_tokens), embedding_max_tokens})
    cases = []
    for cap in caps:
        for is_query in (False, True):
            prepared = prepare_profile_texts(clone, texts, is_query=is_query, max_length=cap,
                                             special_token_layout=layout)
            front = len(layout["prefix_ids"]) + prepared["instruction_tokens"]
            recovered = [token for chunk, weight in zip(prepared["chunks"], prepared["weights"])
                         for token in chunk[front:front + weight]]
            _require(recovered == expected, "Tokenizer smoke check lost or changed A content")
            _require(sum(prepared["weights"]) == len(expected), "Special/instruction tokens entered A pooling weights")
            for chunk in prepared["chunks"]:
                _require(chunk[:len(layout["prefix_ids"])] == layout["prefix_ids"], "Native prefix lost")
                if layout["suffix_ids"]:
                    _require(chunk[-len(layout["suffix_ids"]):] == layout["suffix_ids"], "Native suffix lost")
            selected = [prepared["chunks"][0], prepared["chunks"][-1]]
            batch = embedding.pad({"input_ids": selected, "attention_mask": [[1] * len(row) for row in selected]},
                                  padding=True, return_tensors="np")
            for row, mask, original in zip(batch["input_ids"], batch["attention_mask"], selected):
                _require(row[mask.astype(bool)].tolist() == original, "Padding changed embedding tokens")
            cases.append({"role": "query" if is_query else "document", "cap": cap,
                          "chunks": prepared["chunk_count"], "content_tokens": prepared["content_tokens"],
                          "maximum_input_tokens": prepared["maximum_input_tokens"]})
    generator = AutoTokenizer.from_pretrained(str(generator_model), local_files_only=True, padding_side="left")
    messages = [{"role": "system", "content": "Follow the user's instructions."},
                {"role": "user", "content": "Return only the digit 1."}]
    ids = generator.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=False, padding=False, truncation=False)
    _require(isinstance(ids, list) and ids and all(type(token) is int for token in ids),
             "Generator chat template did not produce flat token IDs")
    _require(generator.eos_token_id is not None, "Generator tokenizer has no EOS token")
    if generator.pad_token_id is None:
        generator.pad_token = generator.eos_token
    generated_batch = generator.pad({"input_ids": [ids, ids[:-1]],
                                     "attention_mask": [[1] * len(ids), [1] * (len(ids) - 1)]},
                                    padding=True, return_tensors="np")
    _require(generated_batch["input_ids"].shape == (2, len(ids)), "Generator padding has unexpected shape")
    for row, mask, original in zip(generated_batch["input_ids"], generated_batch["attention_mask"], [ids, ids[:-1]]):
        _require(row[mask.astype(bool)].tolist() == original, "Generator padding changed chat tokens")
    return {"embedding_tokenizer_class": type(embedding).__name__, "special_token_layout": layout,
            "embedding_cases": cases, "all_A_content_preserved": True,
            "generator_tokenizer_class": type(generator).__name__, "generator_chat_tokens": len(ids),
            "worker_clone_verified": True, "padding_verified": True, "model_weights_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", type=Path, default=Path("/root/autodl-tmp/models/Qwen3-Embedding-0.6B"))
    parser.add_argument("--generator-model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"))
    args = parser.parse_args()
    print(json.dumps(check_tokenizers(args.embedding_model, args.generator_model), indent=2))


if __name__ == "__main__":
    main()
