# Reproduce the experiment pipeline

The portable [adapter demo](INFERENCE.md) needs only the base model and included adapter. Reproducing the reported validation scores requires the historical data, split, prompt selection and evaluation protocol described below. The reserved test set remains unscored.

## Environment and data

Use Linux and Python 3.12. Install a PyTorch 2.8.0 build compatible with your device before the remaining dependencies. Historical fine-tuning used a 48 GB NVIDIA GPU. Run these Bash commands from the repository root, with data and outputs outside the public repository:

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

Replace model paths with local snapshots. Preparation verifies raw-file hashes for dataset revision `f883165a3026fde855dfd448e0cd16443ab257b6`, reconstructs the fixed 1,440/309/309 participant split and creates the training-B majority baseline for validation C. Test answer values are not analyzed. To reuse a compatible split, pass `--split-manifest /path/to/participant_split.csv` to preparation.

## Pilot training and evaluation

Preview the pipeline, then replace `plan` with `run` to execute it:

```bash
python experiments/01_pilot/run_experiments.py plan \
  --experiments qwen_full --workspace "$RUN_ROOT" --dataset-dir "$DATASET" \
  --qwen-model "$QWEN" --liquid-model "$LIQUID"
```

Other experiment groups are `qwen_lora`, `liquid_lora`, `liquid_full` and `cv`. The runner executes prerequisite stages first. Use fresh output directories and retain each run's configuration, source/model hashes, environment and metrics. Do not substitute new results for the historical numbers without reporting their provenance.

## Full-B and retrieval runs

Stage the Full-B scripts with:

```bash
python experiments/setup/prepare_workspace.py --workspace "$RUN_ROOT"
```

The historical Full-B chain requires the genuine Qwen full-tuning and Liquid LoRA metadata produced by the pilot chain. Its evaluators also require original run manifests, model fingerprints and path/configuration consistency. The uploaded adapter alone does not satisfy those provenance checks. Retain the three-epoch training schedule when reproducing the reported epoch-1 checkpoint.

The selected-answer retrieval package includes a reconstructed `audit_selected_answers.py` helper because its original source was missing from the public package. It calls the unchanged selected-answer converter. All 15 report fields match the archived audit on 1,749 training/validation A profiles (276,305 units), checked without reading B/C answers. This packaging repair does not regenerate historical retrieval scores. Use a fresh retrieval output directory so recorded source hashes remain consistent; do not bypass an existing run's hash checks.

Use each script's `--help` for its required paths. These archived experiment chains are more demanding than the standalone adapter demo and are not claimed to have been rerun end to end for this submission update.

## EDA

See [EDA findings](EDA_FINDINGS.md) for the reproducible training-only demographic/missingness command, aggregate outputs and archived validation human-repeatability evidence. The notebook contains the original structure and leakage checks. Point `TWIN_DATA_ROOT` at the dataset before running it. Its optional empirical full-persona leakage check additionally needs `full_persona/chunks/persona_chunk_001.parquet`; the experiment downloader intentionally downloads only wave-split data and label tables.
