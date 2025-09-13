Training Specification (Summary)
================================

Data contract (JSONL)
---------------------
- Required: `text` containing exactly one `<think>...</think>` and one `<answer>...</answer>` pair
- Optional labels: `plan_class`, `target_budget`, `correct`/`correctness`, `difficulty_bin`, `rewrite_pair_id`
- Optional style/decomposition: `style_tag`, `<plan>/<exec>/<eval>` inside `<think>` (collator emits soft masks)

Collation
---------
- `train/data.py::make_collate_fn` tokenizes, builds masks: `think_mask`, `answer_mask`, and `loss_mask`
- Emits labels, rewrite groups and pair maps for paraphrase consistency (change_19)

Losses
------
- `train/losses.py::compute_losses` computes:
  - Answer CE (masked to answer)
  - Plan CE, Budget regression (Huber), Confidence calibration (Brier/BCE)
  - Optional: Quiet‑Star, style invariance, rewrite symmetric masked‑KL (change_19)
  - Optional: think CE, over‑budget penalty, per‑layer variance regularizer

Heads and Aggregator
--------------------
- `tina/metacog_heads.py` provides per‑layer trunk/heads, attention aggregator, and final plan/budget/conf outputs
- Linked‑all‑layers mode pools last‑token hidden states from all decoder layers

On‑Policy + Phases
------------------
- Runner supports periodic `decode_with_budget` to measure think tokens on‑policy and update rewards
- Example phases (A→D): freeze base → on‑policy → partial unfreeze; weights annealed via schedules

Calibration
-----------
- `train/eval_loop.py::export_calibration_from_eval` writes `artifacts/metacog_calibration.json`
- Serve loads `conf_temp`, `plan_thresholds`, and `budget_clip` for parity

Quality vs Budget
-----------------
- `train/eval_loop.py::run_budget_sweep` writes JSON/CSV summaries per budget; `tools/plot_quality_budget.py` plots

AZR Self‑Play Integration (Summary)
-----------------------------------
- Propose→Solve Loop: The model proposes small programmatic reasoning tasks and attempts to solve them.
- Safety: Programs are sandboxed by `train/safe_executor.py` (AST validation, restricted builtins, subprocess, timeout).
- Rewards: combined learnability (executor success, MC trials) + solver correctness − format penalty. Introspection signals (plan/budget/confidence) optionally modulate learnability to prioritize challenging tasks.
- Advantage Estimation: TRR++‑style advantage can be approximated by reward‑weighted teacher forcing on solver prompts (surrogate in our minimal implementation).
