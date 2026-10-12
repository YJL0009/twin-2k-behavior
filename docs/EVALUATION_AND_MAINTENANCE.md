# Evaluation acceptance and long-run maintenance

**Status:** proposed next-stage work, not implemented monitoring or achieved acceptance. Reported model scores are development-validation results; the reserved 309-person test set remains unscored. These numerical thresholds are new engineering decisions made after development analysis, not preregistered rules for historical experiments.

## Freeze the evaluation before using test outcomes

Freeze one candidate, participant split, model/tokenizer revisions, prompt and A-selection policy, decoding, target matching, metrics and acceptance rules. Pricing is the proposed primary endpoint for aggregate purchase-response research. Other families remain secondary; do not change the primary endpoint after seeing test results.

Use only A and answer-free target metadata at inference. Exclude the recipient's B/C answers and the merged full persona. Fit baselines and retrieval indexes using training participants only; pricing matches product and exact price. Freeze predictions before reading held-out labels. Repeated validation informs development, not a final generalization claim.

Compare model and question-only baseline on identical cases. Resample participants, keeping their responses together, for **2,000 paired bootstrap replicates** (seed `20261013`). Recompute within-QID then across-QID aggregation and report percentile 95% intervals for differences. Secondary comparisons are exploratory; searching multiple models or endpoints requires a new confirmatory sample or prespecified multiplicity adjustment.

## Proposed acceptance criteria

These gates would support only a limited research pilot.

| Gate | Proposed rule |
|---|---|
| Primary improvement | Pricing accuracy exceeds the matched question-only baseline by **at least 1.0 percentage point**, with the paired 95% interval for the improvement entirely above zero. |
| Other question families | No prespecified family loses more than **1.0 percentage point** of accuracy or increases normalized error by more than **0.01** versus its baseline. Report every family, including failures. |
| Output validity | At least **99% valid outputs in each family**; publish invalid-output counts and rates with denominators. |
| Subgroup stability | For prespecified age groups and the dataset's recorded sex categories with **at least 50 distinct participants with valid references**, no accuracy loss above **2.0 percentage points** or normalized-error increase above **0.02** versus that group's baseline. Publish group counts and uncertainty. Smaller groups remain descriptive; inadequate coverage prevents a claim of subgroup reliability. |

Keep unbounded numeric MAE and median absolute error separate by question and original unit. For any proposed numeric use, set a domain-specific tolerance before testing; the pricing gate does not validate numeric predictions. Report valid-only numeric performance with the baseline on the identical mask, plus all-case performance of the declared training-median fallback. Do not hide invalid outputs through filtering.

Report human B/C agreement or error on valid paired responses, with participant and pair counts; compare model and baseline on that same subset. Human repeatability is a reference, not a guaranteed ceiling or acceptance threshold. Current evidence establishes neither these gates nor reliable personalization gains.

## Proposed maintenance and ownership

| Responsibility | Monitoring and action |
|---|---|
| ML engineer | During an authorized pilot, check schema failures, output validity, missing A evidence, context coverage and retrieval eligibility daily. Invalid-output rate above **1%** or a schema/leakage failure pauses affected predictions pending investigation. |
| Research lead | Review cohort composition and response distributions monthly. A **10-percentage-point** change in a monitored demographic share or missingness rate triggers investigation, not automatic retraining. Refresh consented human follow-up labels quarterly and reassess baseline gaps, subgroup errors and B/C repeatability. |
| Data steward | Approve collection purpose, consent, access and retention; review deletion requests and each new population or use. Keep participant-level logs private and publish aggregates with small-cell suppression. |

Sustained failure of an acceptance gate on two labeled monitoring cohorts, or a material questionnaire/population change, triggers a retraining proposal. Investigate data and parsing first. New training must use eligible, consented data and pass a fresh temporal holdout; the old test set becomes a regression check after its first use.

Version the dataset snapshot, split, code, base model, adapter, tokenizer, prompts, retrieval index and evaluation outputs together. Record approvals and known limitations in a model card. The research lead approves promotion; the ML engineer retains the prior approved release for rollback. If no approved model is available, pause use or provide the validated baseline with its limitations. Never treat simulated responses as real customer evidence or use this POC for individual high-stakes decisions.
