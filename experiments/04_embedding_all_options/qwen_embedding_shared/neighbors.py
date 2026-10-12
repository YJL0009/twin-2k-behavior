"""Exact-condition, training-only nearest-neighbor predictions.

Retrieval is independent of held-out B/C outcomes. The embedding producer owns
A-only profile construction; this module accepts its frozen, normalized vectors
and checks their participant alignment. Target metadata restricts the eligible
training pool *before* cosine ranking. Pricing QIDs are randomized positions,
so matching uses the actual product stimulus and price instead of the QID.
"""
from collections import Counter, defaultdict
import hashlib
from collections.abc import Mapping

import numpy as np

try:  # Package import and directly executed runner are both supported.
    from .support.baselines import (
        CATEGORICAL, PRICING_ID_RE, _aggregate, _valid, price_and_template,
        require, target_signature,
    )
except ImportError:
    from support.baselines import (
        CATEGORICAL, PRICING_ID_RE, _aggregate, _valid, price_and_template,
        require, target_signature,
    )


PROTOCOL = "qwen-embedding-training-only-exact-target-knn-v1"


def _participant_ids(values, name):
    require(not isinstance(values, (str, bytes)), f"{name} must be a collection of participant IDs")
    pids = list(values)
    require(pids and all(isinstance(pid, str) and pid for pid in pids),
            f"{name} must contain nonempty string participant IDs")
    require(len(pids) == len(set(pids)), f"Duplicate participant ID in {name}")
    return pids


def _vectors(result, group, expected):
    key = f"{group}_pids"
    require(key in result and f"{group}_vectors" in result,
            f"Embedding result is missing {group} IDs or vectors")
    # A set has no usable row order: accepting one would hide a cache bug.
    require(isinstance(result[key], (tuple, list)), f"Embedding {key} must have an explicit row order")
    pids = _participant_ids(result[key], f"embedding {key}")
    require(set(pids) == set(expected),
            f"Embedding {key} coverage differs from the requested split")
    array = np.asarray(result[f"{group}_vectors"])
    require(array.ndim == 2 and array.shape[0] == len(pids) and array.shape[1] > 0,
            f"Embedding {group} vector shape does not match ordered participant IDs")
    require(np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.complexfloating),
            f"Embedding {group} vectors must be real numeric values")
    array = array.astype(np.float64, copy=False)
    require(np.isfinite(array).all(), f"Embedding {group} vectors contain nonfinite values")
    norms = np.linalg.norm(array, axis=1)
    require(np.allclose(norms, 1.0, rtol=0, atol=1e-4),
            f"Embedding {group} vectors must be nonzero and L2-normalized")
    return pids, array


def build_neighbors(people, train_labels, train_pids, query_pids, embedding_result, *, k=5):
    """Build compatible kNN and condition-only predictions from training B.

    ``embedding_result`` must contain ordered ``train_pids``, ``query_pids``
    lists and normalized 2-D ``train_vectors``, ``query_vectors`` arrays. The
    order may differ from the supplied split lists; vector rows are always
    associated with the accompanying ordered IDs, never with a sorted copy.

    ``people`` contains only requested training/query A and answer-free target
    metadata. Labels are a ``(train_pid, source_item_id) -> B`` mapping. Other
    participant labels, unknown nonmissing label keys, duplicate donors and
    missing exact target support raise errors. Empty CSV cells for unassigned
    experimental branches are ignored. Invalid or missing training B values do not
    enter the eligible pool. Having fewer than k eligible donors is allowed
    and reported; having zero fails instead of using a positional-QID fallback.

    Returns ``knn`` and ``condition_baseline`` tuple-key maps plus ``report``.
    Every kNN record has the SAME top-k list under ``neighbors`` and ``donors``:
    use this list for both voting and the subsequent LLM demonstrations.
    """
    train_pids = _participant_ids(train_pids, "train_pids")
    query_pids = _participant_ids(query_pids, "query_pids")
    train_set, query_set = set(train_pids), set(query_pids)
    require(not train_set & query_set, "Training and query participants must be disjoint; no self-neighbors")
    require(set(people) == train_set | query_set,
            "Participant metadata must cover exactly the train/query split; no test or extra participants")
    require(isinstance(k, int) and not isinstance(k, bool) and k > 0, "k must be a positive integer")
    require(isinstance(train_labels, Mapping), "Training labels must be a mapping")
    structural_missing_labels = 0
    for key in train_labels:
        require(isinstance(key, tuple) and len(key) == 2, "Malformed training label key")
        pid, item_id = key
        require(pid in train_set, "Held-out or nontraining participant answer passed as a training label")
        # Native CSV loading requests the union of columns across experimental
        # branches. A person not assigned that branch legitimately has a None
        # cell despite having no corresponding target in their questionnaire.
        if item_id not in people[pid]["items"]:
            require(train_labels[key] is None,
                    f"Training label has unknown source item with a nonmissing answer: {pid}/{item_id}")
            structural_missing_labels += 1
    require(isinstance(embedding_result, Mapping), "Embedding result must be a mapping")
    train_order, train_vectors = _vectors(embedding_result, "train", train_pids)
    query_order, query_vectors = _vectors(embedding_result, "query", query_pids)
    require(train_vectors.shape[1] == query_vectors.shape[1], "Training/query embedding dimensions differ")
    train_index = {pid: index for index, pid in enumerate(train_order)}
    query_index = {pid: index for index, pid in enumerate(query_order)}

    groups = defaultdict(list)
    signatures = {}
    invalid_labels = missing_labels = 0
    pricing_products = set()
    for pid in sorted(train_set | query_set):
        require(isinstance(people[pid].get("items"), Mapping) and bool(people[pid]["items"]),
                f"Missing target items for participant {pid}")
        seen_products = set()
        for item_id, spec in people[pid]["items"].items():
            require(item_id == spec.get("item_id"), "Item dictionary key mismatch")
            signature = target_signature(spec)  # Also rejects answer payloads.
            signatures[(pid, item_id)] = signature
            if PRICING_ID_RE.fullmatch(str(spec["qid"])):
                _, template = price_and_template(spec)
                product = hashlib.sha256(template.encode()).hexdigest()
                require(product not in seen_products, f"Product occurs twice for participant {pid}")
                seen_products.add(product)
                pricing_products.add(product)
            if pid not in train_set:
                continue
            if (pid, item_id) not in train_labels:
                missing_labels += 1
                continue
            value = train_labels[(pid, item_id)]
            if not _valid(value, spec):
                invalid_labels += 1
                continue
            label = int(float(value)) if spec["family"] in CATEGORICAL else float(value)
            groups[signature].append((pid, item_id, label))

    for pool in groups.values():
        require(len(pool) == len({row[0] for row in pool}),
                "Duplicate participant in exact-target candidate pool")

    knn, condition_baseline = {}, {}
    for pid in sorted(query_set):
        # A single participant cosine vector is reused for every target item.
        scores = np.clip(train_vectors @ query_vectors[query_index[pid]], -1.0, 1.0)
        for item_id, spec in people[pid]["items"].items():
            candidates = groups.get(signatures[(pid, item_id)], [])
            require(bool(candidates),
                    f"No valid training-B donor under the exact target conditions: {pid}/{item_id}. "
                    "No positional-QID or different-price fallback is permitted.")
            ranked = sorted(candidates,
                            key=lambda row: (-float(scores[train_index[row[0]]]), row[0], row[1]))
            neighbors = [{"pid": donor, "item_id": source_item, "label": label,
                          "similarity": float(scores[train_index[donor]])}
                         for donor, source_item, label in ranked[:k]]
            key = (pid, item_id)
            knn[key] = {
                "prediction": _aggregate([neighbor["label"] for neighbor in neighbors], spec),
                "neighbors": neighbors, "donors": [dict(neighbor) for neighbor in neighbors],
                "candidate_count": len(candidates), "neighbor_count": len(neighbors),
                "train_support": len(candidates), "fallback": False, "fallback_reason": None,
                "method": "train_A_qwen_embedding_exact_target_knn",
            }
            condition_baseline[key] = {
                "prediction": _aggregate([label for _, _, label in candidates], spec),
                "candidate_count": len(candidates), "train_support": len(candidates),
                "fallback": False, "fallback_reason": None,
                "method": "train_B_exact_target_mode_or_median",
            }

    records = list(knn.values())
    report = {
        "protocol": PROTOCOL,
        "train_participants": len(train_set), "query_participants": len(query_set),
        "examples": len(knn), "k": k, "demonstrations": k,
        "embedding_dimension": train_vectors.shape[1],
        "similarity": "Cosine of frozen L2-normalized A-profile embeddings; training participants only",
        "ranking": "Eligible exact-target training donors first; descending cosine, then lexical PID/source-item tie break",
        "retrieval_target_match": "Exact wording/options/schema; pricing ignores randomized QID position and matches actual product plus exact price",
        "nominal_aggregation": "Unweighted mode; lowest numeric option wins ties",
        "ordinal_aggregation": "Lower sample median, preserving integer option validity",
        "numeric_aggregation": "Sample median",
        "condition_baseline": "Same exact-target eligible pool; unweighted training-B mode/median across all donors",
        "fallback_examples": 0,
        "fewer_than_k_neighbors": sum(row["neighbor_count"] < k for row in records),
        "candidate_count_min": min(row["candidate_count"] for row in records),
        "candidate_count_max": max(row["candidate_count"] for row in records),
        "candidate_count_histogram": dict(sorted(Counter(row["candidate_count"] for row in records).items())),
        "neighbor_count_histogram": dict(sorted(Counter(row["neighbor_count"] for row in records).items())),
        "best_similarity_nonpositive_examples": sum(row["neighbors"][0]["similarity"] <= 0 for row in records),
        "invalid_train_labels_excluded": invalid_labels,
        "missing_train_labels_excluded": missing_labels,
        "unassigned_branch_empty_label_cells_ignored": structural_missing_labels,
        "pricing_unique_products": len(pricing_products),
        "same_neighbors_for_vote_and_LLM": True,
        "heldout_labels_read": False, "C_labels_read": False,
    }
    return {"knn": knn, "condition_baseline": condition_baseline, "report": report}
