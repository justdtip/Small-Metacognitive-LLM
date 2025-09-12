Tina Introspective Reasoning — Local Training and Serving
=========================================================

Overview
--------
This repository implements an additive, introspective Chain‑of‑Thought (CoT) pipeline for small local LLMs:

- Side adapters (LoRA‑style) gated to the <think> segment only
- Metacognitive heads that predict plan, think token budget, and confidence
- Budget‑aware decoding with a soft‑cap and strict stop‑on‑tags parity across train/eval/serve
- Optional per‑layer heads with an attention aggregator for broader visibility

Requirements
------------
- Python 3.10+
- PyTorch (GPU recommended)
- transformers, peft (optional for adapters)
- Dev/testing: pytest, jsonschema, ruff, mypy

Install (editable with dev extras):

```
python -m pip install --upgrade pip
pip install -e .[dev]
```

Quickstart
----------
Serve with introspection and budgeted CoT:

```
python tina_chat_introspective.py \
  --models-root model --base Base --adapter Tina --subfolder checkpoint-2000 \
  --budget-cap 128 --temperature 0.2 --top-p 0.95 --no-stream \
  --calibration artifacts/metacog_calibration.json
```

Observe inference with structured logs:

```
python tools/observe_infer.py \
  --models-root model --base Base --adapter Tina --subfolder checkpoint-2000 \
  --prompt "Add 37 and 45" --visible-cot false \
  --calibration artifacts/metacog_calibration.json \
  --debug-per-layer --jsonl-out logs/obs.jsonl
```

Training (phases)
-----------------
- One‑cycle smoke: `python train/one_cycle.py --data data/train.jsonl`
- Full trainer (optional): set `trainer.enabled: true` in `train_config.yaml`, then run:

```
python train/runner.py --train-config train_config.yaml --steps 2000
```

The example phase schedule in `train_config.yaml` progresses from frozen base (A) to on‑policy and partial unfreeze (C). Losses include answer CE, plan/budget/conf auxiliaries, optional over‑budget penalty, think CE, and Quiet‑Star consistency.

Calibration
-----------
Fit confidence temperature and optional thresholds/clip after validation:

```
from train.eval_loop import export_calibration_from_eval
# tensors: conf_logits, conf_labels, plan_logits, plan_labels, pred_budgets, gold_budgets
export_calibration_from_eval(
  conf_logits=conf_logits, conf_labels=conf_labels,
  plan_logits=plan_logits, plan_labels=plan_labels,
  pred_budgets=pred_budgets, gold_budgets=gold_budgets,
  out_path='artifacts/metacog_calibration.json')
```

Pass the JSON to serving/observation via `--calibration` (applies confidence temperature, plan thresholds, and budget clip).

Quality‑vs‑Budget Evaluation
----------------------------
Run a sweep and plot:

```
from train.eval_loop import run_budget_sweep
results = run_budget_sweep(tokenizer, model, eval_dataset, budgets=[16,32,64], split='dev')

python tools/plot_quality_budget.py --csv artifacts/budget_sweep_dev.csv --out artifacts
```

Outputs include EM, token‑F1, avg think tokens, leakage rate, and (optionally) ECE/Brier when confidence/labels are available.

CI / Validation
---------------
Guardrails run in CI (`.github/workflows/ci.yml`):

- `tools/validate_configs.py` checks stop sequences (</answer>, </think>), tokenizer atomicity for tags, and parity
- `pytest` executes unit tests (stop rules, parity, heads/adapter behavior, budget counting, etc.)

Security & PII Redaction
------------------------
The CLI redacts logs by default (emails, digit runs). Use `--log_raw` to disable redaction for local debugging only. Hidden CoT is off by default for end users; visible mode is opt‑in.

Key References
--------------
- Heads and tokenizer utilities: `tina/metacog_heads.py`, `tina/tokenizer_utils.py`
- Serving engine: `tina/serve.py`
- Training: `train/runner.py`, `train/one_cycle.py`, `train/losses.py`
- Observation: `tools/observe_infer.py`
- Config validator: `tools/validate_configs.py`

