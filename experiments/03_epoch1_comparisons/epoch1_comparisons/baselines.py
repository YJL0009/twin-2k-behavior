"""Training-only A-neighbor and product/price baselines (NumPy only).

No held-out outcomes are accepted. Similarity uses A answer payloads only;
question wording is hashed only to disambiguate question identity, never
tokenized as a similarity feature. B/C labels never become retrieval features.
Target metadata is used solely to restrict donors to exactly the same item,
wording, treatment, price, option order and response schema. Pricing QIDs are
randomized display positions: those donors match the product and price while
retaining the donor's own source item ID for reading its training-B label.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re

import numpy as np


PROTOCOL = "epoch1-training-only-baselines-v1"
CATEGORICAL = {"mc", "ordinal", "matrix_nominal", "matrix_ordinal"}
ORDERED = {"ordinal", "matrix_ordinal"}
PRICE_RE = re.compile(r"(The product is priced at:\s*\$)([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)")
PRICING_ID_RE = re.compile(r"QID9_(?:[1-9]|[1-3][0-9]|40)")
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
ANSWER_KEYS = {"Answers", "Answer", "SelectedByPosition", "SelectedText"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _has_answers(value):
    if isinstance(value, dict):
        return any(k in ANSWER_KEYS or _has_answers(v) for k, v in value.items())
    return isinstance(value, (tuple, list)) and any(_has_answers(v) for v in value)


def target_signature(spec):
    require(not _has_answers(spec), "Answer payload reached target schema")
    require(isinstance(spec.get("target_text"), str) and spec["target_text"], "Missing target wording")
    keys = ("item_id", "qid", "target_text", "family", "sub_id", "options", "lower", "upper")
    if PRICING_ID_RE.fullmatch(str(spec["qid"])):
        # QID9_n is display position, not a stable product identity.
        keys = tuple(key for key in keys if key not in {"item_id", "qid"})
    return _json({key: spec.get(key) for key in keys})


def _valid(value, spec):
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return False
    if not math.isfinite(number):
        return False
    if spec["family"] in CATEGORICAL:
        return number.is_integer() and 1 <= number <= len(spec["options"])
    return (spec.get("lower") is None or number >= float(spec["lower"])) and (
        spec.get("upper") is None or number <= float(spec["upper"]))


def _aggregate(values, spec):
    require(bool(values), f"No valid training-B support for {spec['item_id']}")
    if spec["family"] in {"mc", "matrix_nominal"}:
        counts = Counter(map(int, values))
        return min(counts, key=lambda answer: (-counts[answer], answer))
    if spec["family"] in ORDERED:
        # Lower sample median remains a valid option when support is even.
        return int(sorted(values)[(len(values) - 1) // 2])
    return float(np.median(values))


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from _leaves(child, path + (str(i + 1),))
    elif value is not None:
        yield path, value


def answer_identity(qid, unit):
    """A QIDs are reused across sections, so qid alone is not an identity."""
    question = unit.get("Question", {})
    identity = {"qid": str(qid), "block": unit.get("BlockName", ""),
                "instructions": unit.get("PrecedingDisplayInstructions", []),
                "question": {key: value for key, value in question.items() if key not in ANSWER_KEYS}}
    return str(qid) + "@" + hashlib.sha256(_json(identity).encode()).hexdigest()[:24]


def answer_features(a_units, forbidden_qids):
    """Features grouped by A QID, with matrix rows kept distinct.

    Positional answers are categorical; free-text answers supply Unicode word
    unigrams keyed by question/subfield. SelectedText is not duplicated when
    positions exist. Each question gets equal L2 weight later, so a long matrix
    or essay cannot dominate solely through its number of rows or words.
    """
    result = {}
    for qid, unit in a_units:
        qid = str(qid)
        require(qid not in forbidden_qids, f"B/C target {qid} leaked into A retrieval features")
        question = unit.get("Question", {})
        require(str(question.get("QuestionID")) == qid, "A question ID mismatch")
        identity = answer_identity(qid, unit)
        require(identity not in result, f"Duplicate full A question identity {qid}")
        answers = question.get("Answers")
        require(isinstance(answers, dict), f"Unsupported A answer payload for {qid}")
        counter = Counter()
        if "SelectedByPosition" in answers:
            position = answers["SelectedByPosition"]
            selector = (question.get("Settings") or {}).get("Selector", "")
            # Answer positions are comparable only under identical option/row
            # mappings. This metadata disambiguates a value; it is not itself
            # a feature on which two profiles can agree.
            mapping = {key: question.get(key) for key in
                       ("QuestionType", "Options", "Rows", "RowsID", "Columns", "ColumnsID")}
            mapping["Selector"] = selector
            mapping_hash = hashlib.sha256(_json(mapping).encode()).hexdigest()[:24]
            if question.get("QuestionType") == "MC" and selector in {"MAVR", "MAHR"}:
                require(isinstance(position, list), f"Invalid multi-select A positions: {qid}")
                for value in position:
                    counter[(identity, "selection/" + mapping_hash, _json(value))] = 1
            else:
                for path, value in _leaves(position):
                    counter[(identity, "position/" + mapping_hash + "/" + "/".join(path), _json(value))] += 1
        else:
            for path, value in _leaves(answers):
                if isinstance(value, str):
                    for word in WORD_RE.findall(value.casefold()):
                        counter[(identity, "text/" + "/".join(path), word)] += 1
                else:
                    counter[(identity, "value/" + "/".join(path), _json(value))] += 1
        result[identity] = counter
    return result


class AOnlyIndex:
    """Train-fitted TF-IDF cosine, using NumPy posting lists rather than SciPy."""

    def __init__(self, people, train_pids, forbidden_qids):
        self.train_pids = sorted(map(str, train_pids))
        self.forbidden_qids = set(forbidden_qids)
        require(self.train_pids, "Empty training pool")
        self.pid_index = {pid: i for i, pid in enumerate(self.train_pids)}
        features = {pid: answer_features(people[pid]["a_units"], self.forbidden_qids)
                    for pid in self.train_pids}
        df = Counter()
        for groups in features.values():
            df.update(set(feature for group in groups.values() for feature in group))
        # Unique free-text words are omitted; rare categorical answers remain.
        self.idf = {feature: math.log((1 + len(self.train_pids)) / (1 + count)) + 1
                    for feature, count in df.items()
                    if count >= 2 or not feature[1].startswith("text/")}
        posting_ids, posting_weights = defaultdict(list), defaultdict(list)
        for pid, groups in features.items():
            for feature, weight in self._vector(groups).items():
                posting_ids[feature].append(self.pid_index[pid])
                posting_weights[feature].append(weight)
        self.postings = {feature: (np.asarray(ids, dtype=np.int32),
                                  np.asarray(posting_weights[feature], dtype=np.float64))
                         for feature, ids in posting_ids.items()}
        self.vocabulary_hash = hashlib.sha256(_json(sorted(self.idf.items())).encode()).hexdigest()

    def _vector(self, groups):
        vector = {}
        for counter in groups.values():
            values = {feature: (1 + math.log(count)) * self.idf[feature]
                      for feature, count in counter.items() if feature in self.idf and count > 0}
            norm = math.sqrt(sum(value * value for value in values.values()))
            if norm:
                vector.update({feature: value / norm for feature, value in values.items()})
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {feature: value / norm for feature, value in vector.items()} if norm else {}

    def scores(self, a_units):
        vector = self._vector(answer_features(a_units, self.forbidden_qids))
        scores = np.zeros(len(self.train_pids), dtype=np.float64)
        for feature, weight in vector.items():
            if feature in self.postings:
                indices, values = self.postings[feature]
                scores[indices] += values * weight
        return np.clip(scores, 0, 1), len(vector)


def price_and_template(spec):
    """Parse exactly the price slot, retaining every other stimulus byte."""
    require(PRICING_ID_RE.fullmatch(str(spec["qid"])), "Not a pricing item")
    require(spec["item_id"] == spec["qid"] and spec["family"] == "mc", "Unexpected pricing schema")
    require(spec.get("options") == ["Yes, I would purchase the product", "No, I would not purchase the product"],
            f"Pricing option mapping changed: {spec['item_id']}")
    matches = list(PRICE_RE.finditer(spec["target_text"]))
    require(len(matches) == 1, f"Expected exactly one explicit product price: {spec['item_id']}")
    match = matches[0]
    price = float(match.group(2).replace(",", ""))
    require(math.isfinite(price) and price >= 0, "Product price must be nonnegative and finite")
    template = spec["target_text"][:match.start(2)] + "<PRICE>" + spec["target_text"][match.end(2):]
    return price, template


def fit_decreasing_isotonic(prices, purchased):
    """Weighted PAV on empirical price groups; no regularization is tuned."""
    x = np.asarray(prices, dtype=float)
    y = np.asarray(purchased, dtype=float)
    require(len(x) == len(y) and len(x) > 0, "Invalid isotonic training arrays")
    require(np.isfinite(x).all() and np.isfinite(y).all() and set(y.tolist()) <= {0.0, 1.0},
            "Isotonic labels must be binary")
    unique, inverse = np.unique(x, return_inverse=True)
    weights = np.bincount(inverse).astype(float)
    successes = np.bincount(inverse, weights=y)
    blocks = []
    for i, (weight, total) in enumerate(zip(weights, successes)):
        blocks.append([i, i, weight, total])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] < blocks[-1][3] / blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    probabilities = np.empty(len(unique), dtype=float)
    for first, last, weight, total in blocks:
        probabilities[first:last + 1] = total / weight
    return {"prices": unique.tolist(), "purchase_probabilities": probabilities.tolist(),
            "price_group_support": weights.astype(int).tolist(), "train_support": len(x),
            "isotonic_blocks": len(blocks)}


def build_baselines(people, train_labels, train_pids, val_pids, k=5, demonstrations=3):
    """Return ``{'knn': tuple-key map, 'price': tuple-key map, 'report': dict}``.

    people[pid] has a_units and item_id->answer-free scalar specs. train_labels
    maps (pid,item_id)->B value; any held-out PID in this mapping is rejected.
    kNN records provide top-k neighbors, top-demonstrations donors, predictions
    and explicit fallback/support metadata. The price map covers only
    QID9_1..40 with per-product isotonic purchase curves. Labels never fit
    the A index.
    """
    train_pids, val_pids = set(map(str, train_pids)), set(map(str, val_pids))
    require(train_pids and val_pids and not train_pids & val_pids, "Training and validation IDs must be nonempty and disjoint")
    require(train_pids | val_pids <= set(people), "Requested participant metadata missing")
    require(all(str(pid) in train_pids for pid, _ in train_labels), "Held-out answer passed as a training label")
    require(isinstance(k, int) and k > 0 and isinstance(demonstrations, int) and demonstrations > 0,
            "k and demonstrations must be positive integers")
    forbidden = {str(spec["qid"]) for pid in train_pids | val_pids for spec in people[pid]["items"].values()}
    index = AOnlyIndex(people, train_pids, forbidden)
    groups, item_values, price_rows, templates = defaultdict(list), defaultdict(list), defaultdict(list), {}
    all_specs = {}
    product_counts, product_positions = {}, defaultdict(set)
    for pid in sorted(train_pids | val_pids):
        seen_products = set()
        for item_id, spec in people[pid]["items"].items():
            require(item_id == spec["item_id"], "Item dictionary key mismatch")
            signature = target_signature(spec)
            all_specs[item_id] = spec
            pricing = bool(PRICING_ID_RE.fullmatch(str(spec["qid"])))
            if pricing:
                price, template = price_and_template(spec)
                product = hashlib.sha256(template.encode()).hexdigest()
                require(product not in seen_products, f"Product occurs twice for participant {pid}")
                seen_products.add(product)
                templates[product] = template
                product_positions[product].add(item_id)
            if pid not in train_pids:
                continue
            label = train_labels.get((pid, item_id))
            if not _valid(label, spec):
                continue
            label = int(float(label)) if spec["family"] in CATEGORICAL else float(label)
            groups[signature].append((pid, item_id, label))
            item_values[item_id].append(label)
            if pricing:
                price_rows[product].append((price, int(label == 1)))
        product_counts[pid] = len(seen_products)
    fallbacks = {item_id: _aggregate(item_values[item_id], spec) for item_id, spec in all_specs.items()}
    curves = {product: dict(fit_decreasing_isotonic(*zip(*rows)), product_template=templates[product])
              for product, rows in price_rows.items()}
    knn, price_output = {}, {}
    empty_vectors = 0
    for pid in sorted(val_pids):
        scores, feature_n = index.scores(people[pid]["a_units"])
        empty_vectors += int(feature_n == 0)
        for item_id, spec in people[pid]["items"].items():
            candidates = groups.get(target_signature(spec), [])
            require(len({row[0] for row in candidates}) == len(candidates), "Duplicate participant in exact-target candidate pool")
            ranked = sorted(candidates, key=lambda row: (-float(scores[index.pid_index[row[0]]]), row[0], row[1]))
            selected = [{"pid": donor, "item_id": source_item, "label": label,
                         "similarity": float(scores[index.pid_index[donor]])}
                        for donor, source_item, label in ranked[:max(k, demonstrations)]]
            fallback = not bool(selected)
            knn[(pid, item_id)] = {
                "prediction": fallbacks[item_id] if fallback else _aggregate([v["label"] for v in selected[:k]], spec),
                "neighbors": selected[:k], "donors": selected[:demonstrations],
                "candidate_count": len(candidates), "neighbor_count": min(k, len(candidates)),
                "fallback": fallback, "fallback_reason": "no_exact_target_train_donor" if fallback else None,
                "train_support": len(item_values[item_id]), "query_feature_count": feature_n,
                "method": "train_A_only_exact_target_knn" if not fallback else "train_item_fallback"}
            if PRICING_ID_RE.fullmatch(str(spec["qid"])):
                price, template = price_and_template(spec)
                product = hashlib.sha256(template.encode()).hexdigest()
                require(product in curves, f"No price curve training support for product: {item_id}/{product}")
                curve = curves[product]
                probability = float(np.interp(price, curve["prices"], curve["purchase_probabilities"]))
                position = next((i for i, p in enumerate(curve["prices"]) if p == price), None)
                price_output[(pid, item_id)] = {
                    "prediction": 1 if probability >= 0.5 else 2, "probability_purchase": probability,
                    "product_template_sha256": product,
                    "price": price, "train_support": curve["train_support"],
                    "same_price_train_support": 0 if position is None else curve["price_group_support"][position],
                    "outside_train_price_range": price < curve["prices"][0] or price > curve["prices"][-1],
                    "method": "train_product_price_decreasing_isotonic"}
    records = list(knn.values())
    report = {"protocol": PROTOCOL, "train_participants": len(train_pids), "validation_participants": len(val_pids),
        "examples": len(knn), "k": k, "demonstrations": demonstrations,
        "similarity": "A-only train-fitted TF-IDF; equal A-question L2 weights; global cosine; no B/C labels",
        "vocabulary_features": len(index.idf), "vocabulary_sha256": index.vocabulary_hash,
        "query_empty_vectors": empty_vectors, "retrieval_target_match": "exact item/schema/options/target_text; pricing ignores randomized QID position and matches exact product+price",
        "fallback_examples": sum(v["fallback"] for v in records),
        "fewer_than_k_neighbors": sum(v["neighbor_count"] < k for v in records),
        "fewer_than_requested_demonstrations": sum(len(v["donors"]) < demonstrations for v in records),
        "zero_best_similarity_examples": sum(bool(v["neighbors"]) and v["neighbors"][0]["similarity"] == 0 for v in records),
        "nominal_aggregation": "unweighted mode; lowest numeric option wins ties",
        "ordinal_aggregation": "lower sample median, preserving integer option validity",
        "numeric_aggregation": "sample median", "price_curves": curves,
        "pricing_identity": "sha256 of actual product stimulus with only the price slot replaced; pool training B across all randomized QID9 display positions",
        "pricing_unique_products": len(templates),
        "pricing_products_per_train_participant": dict(Counter(product_counts[pid] for pid in train_pids)),
        "pricing_products_per_validation_participant": dict(Counter(product_counts[pid] for pid in val_pids)),
        "pricing_distinct_display_positions_per_product": {product: len(positions) for product, positions in product_positions.items()},
        "price_rule": "weighted decreasing isotonic P(purchase|product,price), linear interpolation, constant tails, purchase at probability>=0.5",
        "heldout_labels_read": False, "C_labels_read": False}
    return {"knn": knn, "price": price_output, "report": report}
