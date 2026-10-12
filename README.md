# Behavior prediction on Twin-2K-500

Predict a person's later survey answers from their earlier profile. This project compares question-only baselines, small language model fine-tuning, and retrieval of similar participants.

**Main finding:** the question-only baseline remains competitive. Qwen LoRA slightly improves pricing accuracy (73.49% vs 73.41%), but no statistically significant improvement has been established. Adding neighbor examples improves the original Qwen across all six groups below, but both retrieval methods still trail the question-only baseline.

**All results are development-validation scores. The 309-person test set remains unscored.**

[Experiment slides](Twin_2K_500_Experiments.pptx) · [EDA notebook](Twin_2K_500_EDA.ipynb)

## Data

The dataset has **2,058 participants**, split by person into **1,440 train / 309 validation / 309 test**.

| Name | Meaning | Role |
|---|---|---|
| **A: profile** | Earlier questions and answers that are not repeated in wave 4 | Model input and participant similarity |
| **B: earlier target answers** | Earlier answers to questions repeated in wave 4 | Training labels and neighbors' example answers |
| **C: later target answers** | Wave-4 answers to those repeated questions | Evaluation reference |

Inputs use wave-split A and an answer-free target question. The merged full-persona field can contain C answers and must not be used as model input. Baselines and retrieval examples use training participants only.

## Methods: what each results column means

| Results column | How the prediction is made |
|---|---|
| **Question-only baseline** | Use all training participants who answered the same target condition. Predict their most common B answer for nominal choices, or their median B answer for ordered/numeric questions. No profile or LLM is used. |
| **Qwen LoRA (epoch 1)** | Fine-tune Qwen2.5-0.5B-Instruct on A + target question → B. Use the epoch-1 adapter to predict validation C from A + target question. |
| **5-neighbor vote/median** | Retrieve the five most similar eligible training participants, then directly combine their B answers using the same mode/median rule. No LLM generates the answer. |
| **Qwen without neighbor examples** | Give the original Qwen2.5-0.5B-Instruct only the current person's A snippet and target question. Use the same A snippet as the neighbor-example run, with no adapter. |
| **Qwen + 5 neighbor examples** | Give the original Qwen2.5-0.5B-Instruct the current person's A, the target question, and those same neighbors' A snippets + B answers. Qwen generates the prediction without a fine-tuned adapter. |

The two retrieval columns were previously named **“Selected-answer top-5”** and **“Qwen + selected neighbors.”** They share the same neighbors; they differ in how the final answer is produced.

### How the five neighbors are selected

1. Represent each person's A profile with the question text and their actual selected/entered answers. Remove ordinary unchosen options; retain context needed to interpret answers, such as scale anchors. This is what **selected-answer representation** means.
2. Encode A in chunks with frozen **Qwen3-Embedding-0.6B** and combine the chunk vectors into one profile vector.
3. Keep only training participants with a valid B answer to the exact target condition. For pricing questions, match the **product and exact price**, not just the question ID.
4. Rank eligible participants by cosine similarity and take the top five. Both retrieval methods use this same list; if fewer than five are eligible, use all available matches.

**Top-5 means five people, not five candidate answers.** For example, if their binary B answers are `Yes, Yes, No, Yes, No`, the vote predicts `Yes`. The Qwen method instead receives their examples and generates its own answer.

The embedding model finds similar people; the answering model predicts a response. **No reranker is used.** The separate all-options experiment retains unchosen options in the embedding text; the results below use selected-answer embeddings.

## Validation results

| Question group / metric | Question-only baseline | Qwen LoRA (epoch 1) | 5-neighbor vote/median | Qwen without neighbor examples | Qwen + 5 neighbor examples |
|---|---:|---:|---:|---:|---:|
| Pricing accuracy ↑ | 73.41% | 73.49% | 69.13% | 44.81% | 47.01% |
| Pilot subset accuracy ↑ | 72.52% | 67.09% | 66.79% | 44.76% | 44.97% |
| Nominal matrix accuracy ↑ | 74.60% | 74.60% | 74.51% | 73.29% | 74.54% |
| Ordinal normalized error ↓ | 0.2041 | 0.2204 | 0.2321 | 0.3011 | 0.2818 |
| Ordinal matrix normalized error ↓ | 0.1903 | 0.1923 | 0.2172 | 0.3431 | 0.3322 |
| Bounded numeric normalized error ↓ | 0.1959 | 0.2100 | 0.2258 | 0.3224 | 0.3066 |

**Compare the last two columns to measure the effect of adding neighbor examples:** they use the same original Qwen model, target questions and current-person A snippets. The no-neighbor scores come from the matched `qwen_no_demo` run. Pricing accuracy rises from **44.81% to 47.01%** with examples; the question-only baseline remains stronger at **73.41%**.

Full-B LoRA uses **138,264 training targets** and **29,690 validation cases**. Configuration: rank 8, alpha 16, dropout 0.05, learning rate `1e-4`, batch 16 × accumulation 2. The reported checkpoint is **epoch 1, step 4,321, from a three-epoch schedule**. A is selected within a prompt token budget; prompt and padding tokens are masked from training loss.

The **pilot subset** contains five binary scenarios per person. Its separate training experiments produced:

| Model | Zero-shot | LoRA | Full fine-tuning |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 44.76% | 65.74% | 72.52% |
| Liquid LFM2.5-350M | 51.00% | 66.42% | 72.52% |

Pilot baseline: **72.52%**. Human B/C consistency: **76.73%**, a reference rather than a strict ceiling. Early zero-shot and fine-tuning prompts differ, so these scores do not isolate the effect of training alone.

Scores average within each question ID, then across IDs. Accuracy is higher-is-better; normalized absolute error is lower-is-better (divided by `K − 1` for ordered choices or the allowed numeric range). Unbounded numeric errors are reported separately in the slides. Repeated validation use limits generalization claims.

## Repository contents

| Path | Purpose |
|---|---|
| `Twin_2K_500_Experiments.pptx` | Experimental setup and detailed results |
| `Twin_2K_500_EDA.ipynb` | Data structure, question types and leakage example |
| `experiments/01_pilot/` | Pilot inference, LoRA, full fine-tuning and cross-validation |
| `experiments/02_full_b_lora/` | Full-B LoRA training and evaluation |
| `experiments/03_epoch1_comparisons/` | Epoch-1 inference, profile ablations and lexical baselines |
| `experiments/04_embedding_all_options/` | Retrieval with all-options profile text |
| `experiments/05_embedding_selected/` | Retrieval with selected-answer profile text |
| `experiments/setup/` | Workspace preparation |

Raw data, model weights, checkpoints and run outputs are not included.

## Getting started

Use Linux, Python 3.12 and CUDA-enabled PyTorch 2.8.0. Historical fine-tuning used a 48 GB NVIDIA GPU. Run from the repository root with a private workspace outside the repository:

```bash
python -m pip install -r experiments/requirements.txt
export RUN_ROOT="$PWD/../twin_work"
export DATASET="$RUN_ROOT/data/Twin-2K-500"
export QWEN=/path/to/Qwen2.5-0.5B-Instruct
export LIQUID=/path/to/LFM2.5-350M
export EMBEDDING=/path/to/Qwen3-Embedding-0.6B

python experiments/01_pilot/download_data.py --destination "$DATASET"
python experiments/01_pilot/prepare_data.py --root "$RUN_ROOT" --dataset-dir "$DATASET"
```

Replace model paths with your local snapshots. Preparation checks dataset revision `f883165a3026fde855dfd448e0cd16443ab257b6` and the participant split. To reuse an existing compatible split, pass `--split-manifest /path/to/participant_split.csv` to preparation.

Preview the pilot pipeline; replace `plan` with `run` to execute:

```bash
python experiments/01_pilot/run_experiments.py plan \
  --experiments qwen_full --workspace "$RUN_ROOT" --dataset-dir "$DATASET" \
  --qwen-model "$QWEN" --liquid-model "$LIQUID"
```

Other groups: `qwen_lora`, `liquid_lora`, `liquid_full`, `cv`. Dependencies run first. Full-B training requires the genuine Qwen full-tuning and Liquid LoRA metadata produced by this chain; retain the three-epoch schedule when reproducing the epoch-1 result. Use `experiments/setup/prepare_workspace.py --workspace "$RUN_ROOT"` to stage the Full-B scripts.

Use fresh output directories. Historical runs require their original frozen source, model snapshots and environment; this README update does not rerun training or inference.

## References

- [Twin-2K-500 paper](https://arxiv.org/abs/2505.17479) · [Dataset](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Liquid LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) · [LoRA](https://arxiv.org/abs/2106.09685)

This project uses its own experimental protocol; it does not reproduce the paper's split, models or published score. Upstream dataset and model licenses apply.
