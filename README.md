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

Training (phases / TUI)
-----------------
- Interactive TUI (Textual):

```
python tools/run_training_tui.py
```

Use the form to set model/data paths, steps, save/eval intervals, experts, and aggregator; click Run to start. A live status panel prints training updates.

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
- Self‑Play (AZR)
-----------------
We also support an Absolute Zero Reasoner (AZR) self‑play mode that can complement supervised training:

- The proposer samples code‑based reasoning tasks (program + inputs/outputs) conditioned on one of three modes: deduction, abduction, induction.
- A safe executor validates tasks by sandboxed execution and determinism checks (no imports/attributes; subprocess with a timeout).
- The solver attempts to answer tasks; rewards combine a learnability score (MC success under the executor) and solver correctness, with a small format penalty.
- A lightweight policy update maximizes reward by increasing likelihood of correct answers under the solver prompt.

Enable modes via config:

```
trainer:
  mode: self_play   # supervised | self_play | hybrid
self_play:
  buffer_size: 4096
  num_propose: 16
  num_mc_rollouts: 4
  lambda_learnability: 1.0
  lambda_solver: 1.0
  lambda_format: 0.1
  task_types: [deduction, abduction, induction]
```

Then run:

```
python train/runner.py --train-config train_config.yaml --steps 200
```

Hybrid schedule alternates supervised and self‑play chunks (defaults shown in train_config.yaml):

```
trainer:
  mode: hybrid
  hybrid_schedule:
    supervised_steps: 1000
    self_play_iters: 200
```

Safety constraints are enforced by `train/safe_executor.py` (AST validation, restricted builtins, sandboxed subprocess, timeout, determinism).

Self‑play Training (AZR)
------------------------
The Absolute Zero Reasoner (AZR) paradigm trains via self‑play without external labels. We implement a propose→solve loop:

- Proposer: samples or composes small code‑based reasoning tasks (program + inputs/outputs). Tasks are validated by safe execution and determinism checks (no I/O, no imports).
- Solver: attempts to produce the correct answer for each task. The safe executor acts as a verifier for correctness.
- Rewards: combine a learnability signal (executor success rate under small Monte‑Carlo runs) and solver correctness, minus a format penalty for malformed programs. See `train/self_play_azr.py`.
- Modes: three reasoning modes are supported—deduction, abduction, induction—so you can probe different task structures.

Introspection guidance: the metacognitive heads (plan/budget/confidence) optionally guide task selection by boosting learnability on tasks where the model predicts “stop”, low confidence, or very small think budgets. This encourages exploration of challenging tasks.

References: see the AZR paper for the propose/solve paradigm and TRR++ variants for advantage estimation.
