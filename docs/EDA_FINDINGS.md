# Extended EDA findings

This audit separates **training-only profile and response analysis** from **previously scored validation human retest results**. The reserved test set is not evaluated.

## Human B/C consistency on validation

B and C are the same person's earlier and later answers to repeated questions. The table uses original aggregate files bundled under `docs/eda/source/` from the selected-answer retrieval run; it does not read new C answers. Scores first average within each question ID, then across IDs.

| Question group | Question IDs | Paired scalar responses | Human B/C result | Measure |
|---|---:|---:|---:|---|
| Pricing | 40 | 12,360 | 84.00% | Agreement ↑ |
| Pilot binary subset | 9 | 1,545 | 76.73% | Agreement ↑ |
| Nominal matrix | 2 | 2,498 | 75.28% | Agreement ↑ |
| Ordinal | 19 | 2,472 | 0.1557 | Normalized absolute difference ↓ |
| Ordinal matrix | 5 | 6,489 | 0.1526 | Normalized absolute difference ↓ |
| Bounded numeric | 5 | 3,708 | 0.1810 | Normalized absolute difference ↓ |

All groups cover the same **309 validation participants**. Pair counts count scalar responses, not independent people. The nine pilot question IDs encode alternative branches: each participant contributes five binary scenarios. Ordered differences are divided by `K − 1`; bounded numeric differences use the allowed range. Lower difference means better retest consistency.

The four **unbounded numeric targets** are reported separately in their original units; they have no shared normalization range and must not be averaged into the table above.

| Question ID | Target and preceding anchor | Unit | Paired responses / participants | Mean absolute B/C difference ↓ |
|---|---|---|---:|---:|
| QID164 | Number of African countries in the UN; preceding anchor: 12 countries | countries | 149 | 10.3557 |
| QID166 | Number of African countries in the UN; preceding anchor: 65 countries | countries | 160 | 46.3312 |
| QID168 | Height of the tallest redwood; preceding anchor: 85 feet | feet | 161 | 66.0720 |
| QID170 | Height of the tallest redwood; preceding anchor: 1,000 feet | feet | 148 | 432.7074 |

The anchors are numbers shown in the preceding survey question, not correct answers or bounds. Both country-count targets use countries; both tree-height targets use feet. The experiment's numeric schema treats these as nonnegative values without a fixed upper limit.

These are **retest agreement/difference measures**, not psychometric reliability coefficients and not a strict prediction ceiling. Real preferences can change between waves; a model may also benefit from aggregate regularities. The validation split has been reused for development, so these figures are descriptive, not independent test evidence.

## Training demographics

Computed from A for **1,440 training participants** at pinned dataset revision `f883165a3026fde855dfd448e0cd16443ab257b6`. Percentages use all training participants as the denominator. Categories smaller than five are pooled, with another category if necessary. Question definitions and the released categories are in `eda/training_demographics.csv`.

| Variable | Recorded category | Count | % of training participants |
|---|---|---:|---:|
| US region | South (TX, OK, AR, LA, KY, TN, MS, AL, WV, DC, MD, DE, VA, NC, SC, GA, FL) | 565 | 39.24% |
| US region | West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM) | 349 | 24.24% |
| US region | Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH) | 276 | 19.17% |
| US region | Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME) | 238 | 16.53% |
| US region | Pacific (HI, AK) | 12 | 0.83% |
| Sex assigned at birth | Male | 729 | 50.62% |
| Sex assigned at birth | Female | 711 | 49.38% |
| Age | 30-49 | 515 | 35.76% |
| Age | 50-64 | 470 | 32.64% |
| Age | 18-29 | 263 | 18.26% |
| Age | 65+ | 192 | 13.33% |
| Education | College graduate/some postgrad | 512 | 35.56% |
| Education | Some college, no degree | 323 | 22.43% |
| Education | Postgraduate | 222 | 15.42% |
| Education | High school graduate | 191 | 13.26% |
| Education | Associate's degree | 182 | 12.64% |
| Education | Less than high school | 10 | 0.69% |
| Race or origin | White | 968 | 67.22% |
| Race or origin | Black | 167 | 11.60% |
| Race or origin | Hispanic | 139 | 9.65% |
| Race or origin | Asian | 92 | 6.39% |
| Race or origin | Other | 74 | 5.14% |
| US citizenship | Other categories (pooled to avoid counts below 5) | 1,440 | 100.00% |
| Family income before tax | $100,000 or more | 372 | 25.83% |
| Family income before tax | $50,000-$75,000 | 303 | 21.04% |
| Family income before tax | $30,000-$50,000 | 290 | 20.14% |
| Family income before tax | Less than $30,000 | 248 | 17.22% |
| Family income before tax | $75,000-$100,000 | 227 | 15.76% |
| Employment | Full-time employment | 623 | 43.26% |
| Employment | Self-employed | 196 | 13.61% |
| Employment | Part-time employment | 180 | 12.50% |
| Employment | Unemployed | 178 | 12.36% |
| Employment | Retired | 167 | 11.60% |
| Employment | Student | 51 | 3.54% |
| Employment | Home-maker | 45 | 3.12% |

## Training missingness and coverage

A contains **227,491 stored question records**, with **0 empty answer objects among present records** and **29 absent question records** relative to the catalog's question definitions. These are question-record counts, not scalar-answer counts. **52 A matrix records have shortened answer sequences**, totaling **97 fewer selections than declared rows**; these are retained without assigning entries to guessed rows.

| Stored A question records per person | Training participants |
|---|---:|
| 158 | 1,415 |
| 157 | 21 |
| 156 | 4 |

| B family | Global participant × item cells | Not represented in person's B metadata | Administered cells | Valid answers | Missing answers | Invalid answers |
|---|---:|---:|---:|---:|---:|---:|
| bounded_numeric | 20,160 | 2,880 | 17,280 | 17,280 | 0 | 0 |
| matrix_nominal | 23,040 | 11,496 | 11,544 | 11,544 | 0 | 0 |
| matrix_ordinal | 34,560 | 4,320 | 30,240 | 30,240 | 0 | 0 |
| mc | 70,560 | 5,760 | 64,800 | 64,800 | 0 | 0 |
| ordinal | 27,360 | 15,840 | 11,520 | 11,520 | 0 | 0 |
| unbounded_numeric | 5,760 | 2,880 | 2,880 | 2,880 | 0 | 0 |
| **Total** | 181,440 | 43,176 | 138,264 | 138,264 | 0 | 0 |

The 126 global B scalar columns combine alternative survey branches. A blank in a target absent from the person's B metadata is separated from a blank in an administered target. Metadata absence is not proof of refusal or dropout; this audit does not infer its cause per cell. A-record absence likewise must not be relabeled as nonresponse without survey routing evidence.

## Representativeness and limits

- A participant split prevents identity overlap, but it does not make this survey sample representative of a country, age group, or deployment population.
- Training demographic frequencies describe this split only. No population benchmark, sampling weights, subgroup performance estimates, or causal claims are supplied by this audit.
- Survey routing, incomplete matrix sequences, and repeated measures create unequal observation counts. The 126 global target columns must not be used as a per-person nonresponse denominator.
- Repeated answers are correlated within a person and question. Pair counts are not independent sample sizes; uncertainty estimates would need participant-aware resampling and a declared question aggregation.
- Test C remains reserved. This extension selects no C parquet column and never opens the wave-4 response CSV. Dataset hashing reads raw bytes for integrity only.

## Reproduce

From a fresh checkout, the original aggregate `comparison.csv` and `run_config.json` are already included under `docs/eda/source/`. They contain aggregate metrics, historical paths/settings, and source hashes, with no participant records. Their original bytes and SHA256 hashes are preserved in the audit provenance. `--results` defaults to this bundled folder; it can select another compatible aggregate folder explicitly.

```bash
# Rebuild archived validation retest tables using only bundled inputs (standard library)
python scripts/extend_eda.py --output-dir ../twin_work/eda_archive_review

# Add training A/B demographic and missingness analysis (CPU; pandas/pyarrow)
python scripts/extend_eda.py \
  --dataset-dir /path/to/Twin-2K-500 \
  --output-dir ../twin_work/eda_full_review
```

The manifest is optional: without it the script reconstructs the established seed-42 split with participant 574 reserved for training. If supplied, membership must match exactly. The default demographics use exact question IDs in the pinned catalog's Demographics block. `--demographic-map` accepts a JSON mapping from display names to exact `block` and `qid` selectors; ambiguous matches are omitted, never guessed.

The default output directory is `../twin_work/eda_review` when viewed from the repository root. Each run writes `EDA_FINDINGS.md`, aggregate CSV tables under `eda/`, and `eda/provenance.json` beneath its output directory. Ordinary review commands leave the fuller published report untouched. Updating the published report requires an explicit full-data run with `--output-dir docs`. Raw participant records are not written.
