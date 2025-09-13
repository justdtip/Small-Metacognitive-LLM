Self‑Play (AZR) — Implementation Notes
======================================

Files
-----
- `train/self_play_azr.py`: proposer/solver loop and trainer
- `train/safe_executor.py`: sandboxed program execution and deterministic validation

Core classes
------------
- `ProgramTask`: container for a task (mode, program, inputs/outputs) and rewards.
- `TaskBuffer`: ring buffer per reasoning mode, with sample/append/prune.
- `SelfPlayAZRTrainer`:
  - `propose_tasks(mode, n_examples)`: builds an ICL prompt from past tasks, samples a candidate program+IO, validates via the safe executor, estimates a learnability reward (MC runs), and optionally applies introspection guidance via metacognitive heads.
  - `solve_tasks(tasks)`: verifies solver correctness using the safe executor as ground truth.
  - `update_policy(tasks)`: simple reward‑weighted LM loss on the solver prompt (surrogate for TRR++).
  - `train_loop(steps)`: runs propose→solve→update and prunes buffers.

Safety executor
---------------
`train/safe_executor.py` parses programs with `ast.parse`, disallows unsafe nodes (Import, Attribute, With, Try, Lambda, etc.),
provides a restricted builtins set, and executes code in a subprocess with a timeout. Determinism is enforced by running
twice and comparing outputs.

Extensibility
-------------
- Task types: add new `mode` keys (e.g., transformation, synthesis) and adjust prompts in `_prompt_proposer` / `_prompt_solver`.
- Rewards: incorporate new terms (e.g., complexity penalties, style constraints). Make them visible in logs/metrics.
- Introspection: change the mapping from (plan, budget, confidence) → `introspection_score` to reflect curriculum goals.

Caveats
-------
- Self‑play complements supervised training; it is not a drop‑in replacement. Use hybrid schedules to retain supervised anchors.
- The included policy update is minimal; for production, consider a full TRR++ estimator with per‑mode baselines.

