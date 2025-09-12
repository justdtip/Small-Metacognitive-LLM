# Current Plan of Action

**Status:** 10-step verification pending.

## A. Preconditions (must be true after the 10-step test)
1. `think_tokens_used` reflects *actual* tokens generated during `<think>`, not budget.
2. `stop_sequences` parity across train/eval/serve; no silent defaults in eval.
3. Decomposition tags (`<plan>/<exec>/<eval>`) registered as single tokens and stripped from hidden outputs.
4. Gate coverage is logged and resets each step.
5. Calibration blob persists `conf_temp`, `plan_thresholds`, and `budget_posthoc_clip`, and serve applies them.
6. On-policy decode + RL budget update run at the configured cadence and log RL stats.

## B. If the test passes — implementation roadmap

### Phase 1 — Heads at each MLP layer + aggregator
- **1.1** Add per-layer pooled states (last-token) for **all L layers**.
- **1.2** Implement a **shared trunk MLP** (H→D→D) applied to each layer’s pooled state (parameter tying).
- **1.3** Per-layer small heads output (plan logits, budget scalar, confidence).
- **1.4** Add a **learned attention aggregator** over per-layer embeddings to produce global (plan/budget/conf).
- **1.5** Telemetry: log per-layer vs aggregated predictions; ensure budgets clamped in [0, B_max].
- **1.6** Optional modulation hooks (behind flags): FiLM into side-adapters or small KV-prefix (no effect on `<answer>`).

> Rationale: shared trunk limits params, improves regularization, and gives ~L/3× more visibility vs 3 taps.

### Phase 2 — Reasoning features (order chosen to de-risk)
1. **change_20_cog_operation_head** — multi-label cognitive operations head (retrieve/reason/calc/simulate/verify) on the aggregator embedding; metrics per op.
2. **change_22_strategy_tags_and_alignment** — strategy tags and multi-label head; slice metrics; optional `<strategy:...>` hint (hidden).
3. **change_24_subgoal_planner_head** — next-subgoal classifier; record subgoal telemetry; optional hint token (hidden).
4. **change_28_invariant_supervision** — truth-and-violation tracking for simple invariants (off by default); violation metrics only.
5. **change_25_contradiction_entailment_probe** — small NLI probe (premise: prompt/evidence, hypothesis: current thought); contradiction penalty (off by default).
6. **change_27_analogy_structure_checker** — bijection + anchor preservation loss for analogy items (off by default).
7. **change_29_elimination_grid_schema** — candidate mask tensors for grid/elimination logic; BCE loss off by default.
8. **change_30_answer_invariance_under_strategy_hint** — enforce answer invariance test harness; zero loss by default.
9. **change_31_symbolic_constraints_head** — constraint vector head (emission only; no coupling initially).
10. **change_21_self_reflection_head** — reflection score predictor (rubric/evidence coverage) as telemetry; no coupling initially.
11. **change_32_reflection_summary_generation** — optional `<reflect>...</reflect>` generation (hidden by default), CE off by default; leakage guard.
12. **change_33_program_of_thought_head** — optional `<program>...</program>` segment (hidden by default), CE off by default; sandboxed check tool.

> Rationale: Start with **low-risk probes and telemetry** (20,22,24) that benefit from the aggregator; then add **weakly-coupled evaluators** (28,25,27,29,30,31,21). Finally, add **optional generation** features (32,33) behind strict guards and hidden by default.

### Phase 3 — Gradual coupling (optional)
- Once telemetry shows stable benefits, selectively enable small λ for the relevant auxiliaries (invariants, NLI contradictions, analogy bijection, grid BCE), monitor quality-vs-budget and ECE.

### Phase 4 — Documentation & CI
- Update DATA_SPEC.md for decomposition tags.
- Expand `tools/validate_configs.py` to check decomposition atomicity and sanitizer behaviour (string search forbidding tags in hidden outputs).
- Add CI job to run Steps 1–3 and 10 on every PR touching serve/eval/tokenizer.

---

## C. Exit criteria to move to Phase 1
- Steps 1–10 of the test pass with **no tag leakage**, **parity validated**, **calibration applied**, and **on-policy RL** active. 
- The one-cycle JSON contains: `think_tokens_used`, `think_budget`, `gate_coverage(_mean)`, `stop_sequences`, `slack_ratio`, and a non-null `parity_digest`.

*Signed: Current Plan of Action (prepared for the coder).*
