# Tina Metacognitive Integration Guide (Local Model)

> **Assumptions**: local-only development; no external model APIs. Timezone set by your OS.  
> **Citations**: Project goal summary fileciteturn24file0, Training Architect brief fileciteturn24file1, CoT‑Space theory (noise↔length, U‑curve, difficulty/capacity factors) fileciteturn24file2.

---

## 0) Why this exists (one page)

- **Primary objective**: “Transform Tina into a self‑aware system capable of examining and improving its own thought processes.” fileciteturn24file0  
- **Design pillars**: additive enhancement (don’t regress base), progressive development (curriculum), practical deployment (fast, local) fileciteturn24file0 fileciteturn24file1.  
- **Theory**: CoT‑Space shows an **optimal CoT length** L\* from a bias–variance trade‑off; **noise scale ∝ 1/L** (Theorem 3.1). Harder tasks ⇒ longer L\*; larger models ⇒ shorter L\*; algorithm choice is secondary; higher noise ⇒ shorter L\* (Fig. 3/4). fileciteturn24file2  

**Consequence for code**: we must (1) teach *hidden CoT* and (2) learn *how much to think*, not just “always long.” Heads (plan/budget/confidence), gated adapters, train/serve **parity**, and robust **metrics** are mandatory.

---

## 1) The local model & core folders

**Local base LLM**: loaded from disk (HF‑style). No remote inference—adapters + heads are attached in‑process.  
**Key folders/files** (the new coder should know these cold):

- `tina/tokenizer_utils.py` – **Adds/verifies** special tags `<think>`, `</think>`, `<answer>`, `</answer>` as **single tokens**; builds `think_mask` / `answer_mask`. *Call this once at bootstrap and any time a tokenizer is constructed.* (Guardrail: assert one‑token round‑trip.) fileciteturn24file1  
- `tina/metacog_heads.py` – **Heads** over pooled hidden states (tap layers): `plan_logits`, `budget (int cap)`, `confidence`. Expose temps/clamps; keep math for heads in fp32 for stability. fileciteturn24file1  
- `tina/side_adapters.py` – LoRA‑style residual adapters with **hard‑concrete** per‑token gates and **think‑mask gating**. Coverage telemetry = |masked delta| / |total delta| ∈ [0,1]. fileciteturn24file1  
- `train/data.py` – JSONL ingestion + **collation** (`make_collate_fn`): creates masks, attaches labels (plan/budget/correctness/difficulty). Heuristics backfill where labels missing. fileciteturn24file1  
- `train/losses.py` – `compute_losses`: `answer_ce` (+ optional `think_ce`) + auxiliaries (`plan_ce`, `budget_reg`, `confidence_cal`, `gate_reg`) + optional `quiet_star_loss`. Weighted aggregation with schedulable λ’s. fileciteturn24file1  
- `train/runner.py` – reference **training step** (forward with output_hidden_states=True → register taps → run heads → `compute_losses` → backward/step). RL glue (`rl_phase_step`) uses real hidden‑state features.  
- `train/metrics.py` – accuracy/EM/F1, ECE/Brier, aggregation (slices) + CSV/JSON export.  
- `train/eval_loop.py` – **Parity** with serve: uses `load_service_config()` → gets `visible_cot_default`, `stop_sequences`, `soft_cap_slack_ratio`; `decode_with_budget()` builds identical stop criteria.  
- `tina/serve.py` – **Production engine**: (1) **budget estimate** via dry forward + heads, (2) THINK decode: `StopOnTags('</think>')` + **`SlackStop(budget, slack_ratio)`**, (3) ANSWER decode: stop on `</answer>`, (4) strip `<think>` unless visible mode. Logs a parity digest.  
- `tools/noise_probe.py` – Temperature sweep (LIVE mode should decode per‑T; OFFLINE grouping by record `temperature` when LIVE is off).  
- `tools/reports.py` – Aggregates metrics and writes `metrics.{json,csv}` + `quality_vs_budget.{json,csv}` dashboards.  
- `config/*.json` (+ schemas) – Model/generation/adapter/service config. **Validate at startup**; fail fast if tags split or `</answer>` missing from stop sequences. fileciteturn24file1  

---

## 2) How the pieces plug together (train → eval → serve)

### Train (single step or full run)
1. **Collate**: `make_collate_fn(tokenizer)` builds `think_mask`, `answer_mask`, `loss_mask`, attaches labels.  
2. **Ensure tags**: `ensure_reasoning_tokens(tokenizer)`; assert 1‑token IDs for all special tags.  
3. **Forward**: wrap with `think_mask_context(think_mask.float())`; call model with `output_hidden_states=True`; collect `_hidden_states`.  
4. **Heads**: register taps (e.g., layers `[6,10,14]`), get `plan_logits`, `budget_pred (int)`, `confidence`.  
5. **Loss**: `compute_losses(logits, labels, gate_modules=adapters, weights=...)` with `plan_logits/targets`, `budget_pred/target_budget`, `conf_logits/labels`. (Optionally add `quiet_star_loss`.)  
6. **Backward/step**: clip heads/adapters grads; step. Base can remain frozen early (Stage A), unfreeze top layers later (Stage B). fileciteturn24file1  
7. **(Optional RL)**: `rl_phase_step` pools features and updates a GaussianBudgetPolicy with budget‑aware reward.  
8. **Log**: per‑step JSON with: loss breakdown, plan histogram, budget pred/target/abs‑err, conf mean + ECE, **gate_coverage**, **think_tokens_used**, slack_ratio & stop sequences, parity digest.

### Eval (must mirror serve)
- `decode_with_budget()` **must** use the same stop tags and the **same `soft_cap_slack_ratio`** from `service_config.json`.  Compute accuracy/F1, plan accuracy, budget error, ECE, leakage rate (should be ≈0 when hidden).  Build **quality‑vs‑budget** curves for the U‑curve view.  
- Difficulty/provenance slices should be non‑empty and the **overall** equals the **weighted average** of slices.

### Serve (local, production)
- `IntrospectiveEngine` reads `visible_cot_default`, `stop_sequences`, **`soft_cap_slack_ratio`**, and `confidence_calibration_path`; logs a parity digest.  
- THINK: `StopOnTags('</think>')` + `SlackStop`.  ANSWER: `StopOnTags('</answer>')`.  Strip `<think>` when hidden.  
- Apply **confidence temperature** (from calibration blob) to scale the confidence head probability at serve time.

---

## 3) One‑cycle recipe (no checkpoints)

1. **JSONL data** (each line):  
   ```json
   {"text":"<think>...</think><answer>...</answer>","plan_class":1,"target_budget":64,"correct":1,"difficulty_bin":2}
   ```
2. **Dataloader** with `make_collate_fn(loss_on="answer")`.  
3. **Ensure tags** via `ensure_reasoning_tokens`.  
4. **Forward** under `think_mask_context` with `output_hidden_states=True`; register taps; run heads.  
5. **Loss** via `compute_losses` (answer_ce + auxiliaries + gate_reg + optional quiet_star).  
6. **Backward/step** adapters+heads.  
7. **Mini‑eval** with `IntrospectiveEngine.generate_cot(visible_cot=False)` to verify no `<think>` leakage; record `think_tokens_used`.  
8. **Log JSON**:  
   ```json
   {"loss_terms":{"answer_ce":...,"plan_ce":...,"budget_reg":...,"conf_cal":...,"gate_reg":...,"aux_loss":...},
    "plan_hist":[...],
    "budget_pred_mean":...,"budget_target_mean":...,"budget_abs_err_mean":...,
    "conf_prob_mean":...,"ece_estimate":...,
    "think_tokens_used":...,"slack_ratio":...,"stop_sequences":["</answer>","</think>"],
    "parity_digest":{"visible_cot_default":false,"tag_ids":{"<think>":..., ...}},
    "gate_coverage":...,"leakage_detected":false}
   ```

---

## 4) Guardrails & acceptance

- **Atomic tags**: any split → fail fast.  
- **Stop sequences**: `</answer>` (required) and `</think>` (for think stop) must be present; mismatch between eval and serve is a fail.  
- **Soft‑cap parity**: eval and serve take **the same** `soft_cap_slack_ratio` from `service_config.json`; no hard‑coding.  
- **Calibration**: fitting temperature must **reduce ECE** on held‑out; serve applies 1/T scaling to conf logit.  
- **Gate coverage**: ∈[0,1], monotone with mask inclusion (zeros→0, ones→1), non‑NaN.  
- **Noise probe**: LIVE per‑T decoding or OFFLINE grouping by record `temperature`; rows must differ across T in at least one metric.  
- **Slices**: difficulty/provenance slices not empty; overall equals count‑weighted average of slices.  

---

## 5) How this supports CoT‑Space (for GPT‑5 coders)

- **Optimal L\*** emerges when we balance **underthinking** (Theorem 3.7 lower bound) and **overthinking** (Theorem 3.4 upper bound); the **U‑curve** in Fig. 4 is the operational signature.  Our pipeline enforces this through budget heads, soft‑cap decoding, token‑cost penalties, and the quality‑vs‑budget dashboard. fileciteturn24file2  
- **Noise ↔ length** (Theorem 3.1): temperature sweeps in LIVE mode and per‑T grouping reveal that higher noise leads to shorter CoTs; expect `think_tokens_used` to **decrease** as T increases.  This is why the noise probe is part of CI. fileciteturn24file2  
- **Difficulty slices** (Remark 1): harder bins should show larger budgets/think tokens; verify overall == weighted average of slice metrics.  Capacity (Remark 2) and algorithm‑agnostic convergence (Remark 3) are visible longitudinally in the dashboards. fileciteturn24file2  

---

## 6) “Plug‑here” checklist (copy for PRs)

- [ ] `ensure_reasoning_tokens` called at bootstrap; assert one‑token IDs for all tags. fileciteturn24file1  
- [ ] `make_collate_fn` attaches {think, answer, loss} masks + labels (plan/budget/correctness/difficulty).  
- [ ] Forward uses `output_hidden_states=True`; taps registered; heads run on pooled features; head weights in fp32.  
- [ ] `compute_losses` receives real head outputs/targets; non‑zero auxiliaries observed on labeled data.  
- [ ] Adapters gated by `think_mask_context`; coverage logged and within [0,1]; monotonicity checks pass.  
- [ ] `load_service_config` returns `soft_cap_slack_ratio`; `decode_with_budget` and `serve` both use it.  
- [ ] Confidence calibration fitted (ECE_before > ECE_after); serve applies 1/T scaling.  
- [ ] No leakage of `<think>` in hidden mode; stop at `</answer>` works universally.  
- [ ] Noise probe **LIVE** shows metric differences across T; OFFLINE grouping errors if temperature is absent.  
- [ ] Reports show non‑empty slices and curve length equals requested budgets; overall = weighted slice average.  

---

### Appendix A — Minimal data format (JSONL)

Each line:
```json
{"text":"<think>…</think><answer>…</answer>","plan_class":0|1|…,"target_budget":int,"correct":0|1,"difficulty_bin":int,"temperature":float}
```
