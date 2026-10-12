# Run the Full-B LoRA adapter

The repository includes the epoch-1 adapter and tokenizer at
[`adapters/full-b-lora/epoch-1`](../adapters/full-b-lora/epoch-1).
The external base model is **Qwen/Qwen2.5-0.5B-Instruct**; its weights are not
included. This quick start runs one supplied A excerpt and one target item.
It needs no dataset, original AutoDL workspace, training checkpoints, or B/C labels.

## 1. Inspect the prompt without downloading a model

From the repository root, with Python 3.12:

```bash
python scripts/infer_adapter.py --input examples/full_b_binary.json --dry-run
python scripts/infer_adapter.py --input examples/full_b_numeric.json --dry-run
```

This uses only the Python standard library and prints the actual messages and
output-token limit. It does not load the tokenizer or check the token count.
Both files contain **synthetic data**, not participant records or reported results.
No expected model prediction is asserted.

## 2. Install inference dependencies

Use a virtual environment and install PyTorch for your machine. For a CPU demo:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==5.2.0 peft==0.18.0 accelerate==1.12.0 safetensors
```

For CUDA, install a compatible CUDA build of PyTorch instead of the CPU build;
then install the same remaining packages. The full experiment environment is
listed in [`experiments/requirements.txt`](../experiments/requirements.txt).

## 3. Generate one answer

The first command explicitly downloads/caches the public base model if needed:

```bash
python scripts/infer_adapter.py --input examples/full_b_binary.json --base-model Qwen/Qwen2.5-0.5B-Instruct --device cpu
```

For a base model already downloaded locally:

```bash
python scripts/infer_adapter.py --input examples/full_b_numeric.json --base-model /path/to/Qwen2.5-0.5B-Instruct --local-files-only --device auto
```

`auto` chooses CUDA when available, otherwise CPU. CPU uses float32; CUDA uses
bfloat16 when supported, otherwise float16. Use `--device cuda` to require a GPU.
The adapter path defaults to the included folder, independently of the current
working directory. `--adapter /path/to/adapter` can override it. The script loads
the base explicitly, so the original server path retained in `adapter_config.json`
is not used to locate base weights.

The output is JSON containing `prediction_text`, `parsed_prediction`,
`valid_output`, input-token count, model paths and device. The binary example
accepts only the integer `1` or `2`; the numeric example accepts one number from
`0` to `100`. Generation is greedy (`do_sample=False`), with at most 8 output
tokens for the original binary format or 16 for other scalar items. An invalid
answer is reported unchanged, with `parsed_prediction: null`, `valid_output: false`
and exit code 2; it is not repaired or silently replaced. Other errors use exit
code 1. Successful generation or prompt rendering uses exit code 0.

Both included examples were run successfully on CPU/float32 with the published
adapter, PyTorch 2.8.0+cpu, Transformers 5.2.0 and PEFT 0.18.0. The official base
snapshot was `7ae557604adf67be50417f59c2c2f167def9a775`. The binary input produced
`1` (212 input tokens); the numeric input produced `50` (230 input tokens).
These are **observed synthetic-demo outputs, not correct labels or evaluation
scores**. Selected fields from the binary JSON output were:

```json
{
  "prediction_text": "1",
  "parsed_prediction": 1,
  "valid_output": true,
  "input_tokens": 212,
  "device": "cpu",
  "evaluation_performed": false
}
```

## Input format and scope

- `a_units`: a nonempty list of complete earlier A question/answer units. Each
  has a `Question` object containing its original `QuestionText` and `Answers`.
- `target_item`: one answer-free expanded scalar item: `target_text`, `family`,
  `trained_binary`, plus original ordered `options` for categorical items or
  `lower`/`upper` numeric bounds. Keep exact question wording, option order,
  product, price, instructions and matrix row/statement where applicable.
- `description`: optional explanation, excluded from the model prompt.

Supported families are `mc`, `ordinal`, `matrix_nominal`, `matrix_ordinal`,
`bounded_numeric` and `unbounded_numeric`. `trained_binary: true` selects the
original pilot binary prompt policy; it is **not** a flag for every question with
two options. In the real schema it applies to QID163, QID165, QID167, QID169,
QID183, QID184, QID192, QID193 and QID196. Other items, including binary pricing
questions, use `trained_binary: false`. Numeric bounds start at zero; unbounded
numeric items omit `upper` or set it to `null`.

For actual dataset inputs, obtain the scalar item with the existing
[`full_b_schema.expand_units`](../experiments/02_full_b_lora/qwen_full_b_lora_scripts/full_b_schema.py)
builder from the participant's answer-free target metadata. Supply only the
person's earlier **A** evidence. Do not use the merged full-persona field or add
their B/C target answer. Keep the `target_text` and its options/bounds consistent.

The CLI reuses the experiment's
[`messages_for_context`](../experiments/03_epoch1_comparisons/epoch1_comparisons/prompts.py)
and native scalar prompt/output parsers, and serializes whole A units exactly
as the Full-B engine does. It includes **every A unit supplied in the input**;
it does not recreate historical token-budget selection with the Liquid tokenizer.
Oversized prompts fail instead of being truncated. The default maximum is 32,752
input tokens with space reserved for generation; shorter inputs use less memory.

This is a portable inference demonstration, not a reproduction of the README's
validation metrics. Those require the historical split, A selection, frozen
inputs and scoring protocol. The original evaluator also reconstructs recorded
base precision and merges the adapter; this demo uses ordinary PEFT inference,
so it does not promise bit-for-bit historical outputs. See
[`PROMPTS.md`](PROMPTS.md) for the experiment prompt families and worked examples.
