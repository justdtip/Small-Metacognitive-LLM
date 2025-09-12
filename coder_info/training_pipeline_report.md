# TRAINING PIPELINE REPORT

## Training Pipeline — Architecture & Interdependence Digest (drop into `training_pipeline_report.md`)

### Project goals (recap)

- **Additive metacognition:** keep the base LM intact; add **side adapters** (gated residual deltas) and **metacognitive heads** to (a) plan, (b) allocate a **think** budget, and (c) estimate confidence; default to **hidden CoT**; enforce strong **serve/eval parity**.
- **Strategy over syntax:** teach *how to think* via supervision of strategy/operations, subgoals, invariants, analogical structure, and consistency (e.g., paraphrase/rewrites, multi-branch search) rather than mimicking CoT phrasing.
- **Budget awareness:** accuracy must improve or hold while the `<think>` token budget stays constant or drops; support **dynamic budget selection**; penalize overspending.
- **Calibration & safety:** calibrated confidence, leakage≈0 of hidden CoT, and robust config/tag guardrails.

### Code map — components and responsibilities

#### Runtime & scaffolding (`tina/*`)
- **`tina/serve.py`** — IntrospectiveEngine; applies **StopOnTags** and **SlackStop**; mounts **side adapters** via `IntrospectionScaffold`; hosts **generate_cot**; strips `<think>` by default; exposes telemetry (budget, plan, confidence, leakage).
- **`tina/side_adapters.py`** — Additive **LowRankAdapter** with **HardConcrete** per-token gate; records `_last_gate_activity` and `_last_gate_coverage`; `IntrospectionScaffold` attaches adapters to decoder layers; activates only inside `scaffold.think()` (segment-conditional).
- **`tina/metacog_heads.py`** — Pools tapped hidden layers; returns **plan logits**, **budget**, **confidence**; (extensions: cognitive ops, strategy logits, constraint vector, reflection/subgoal heads).
- **`tina/tokenizer_utils.py`** — Enforces atomic special tags (`<think> </think> <answer> </answer>`) and (extensions) `<reflect> </reflect>`, `<program> </program>`; builds segmentation masks.

#### Training & evaluation (`train/*`)
- **`train/data.py`** — JSONL contract; tokenizes **text**; emits `think_mask`, `answer_mask`, `loss_mask`; attaches labels (`plan_targets`, `target_budget`, `correctness`, `difficulty_bin`); strict tag integrity; (extensions) `think_tokens_used`, `style/strategy` vectors, `constraint_vec`, `reflection_summary`, `program_solution`, `step_types`, `subgoal_id`, `rewrite_pair_id`.
- **`train/losses.py`** — Primary **answer CE**; optional **quiet-star**; auxiliary **plan/budget/confidence**; **gate sparsity**; (extensions) **style invariance KL** (change_12), **rewrite consistency KL** (change_19), **constraint BCE** (change_31), **reflection/program CE** (change_32/33), **analogy**, **NLI contradiction**.
- **`train/runner.py`** — Orchestrates train steps; **on-policy** `<think>` sampling via `decode_with_budget`; **RL budget** update; logs gate coverage; (extensions) multi-branch GRPO.
- **`train/eval_loop.py`** — `decode_with_budget` with **service_config** parity; **quality-vs-budget** sweeps; confidence temperature fitting; CSV/JSON exports; answer extraction parity and leakage checks.
- **`train/metrics.py`** — ECE/Brier, token-F1, rubric scoring hooks, aggregation & slicing (e.g., by provenance, style, strategy).
- **`train/rl_loop.py`** — Gaussian budget policy + REINFORCE/DPO-like updates.
- **`train/hooks.py`** — `think_mask_context` to pass segment mask to adapters.

#### Config & validation (`config/*`, `tools/*`)
- **`config/service_config.json`** — Stop sequences (`</think>`, `</answer>`), default visibility, slack ratio, calibration path. Shared by **serve** and **decode_with_budget**.
- **`adapter_config.json`, `generation_config.json`** — Adapter targets & generation defaults (temperature, top-p).
- **`tools/validate_configs.py`** — (should) assert tag atomicity and parity (train/eval/serve); fail fast on mismatches.

### Script–config interdependence

- **Stop/soft-cap parity**: `tina/serve.py` <-> `train/eval_loop.py` both read `config/service_config.json` for stop tags & slack ratio; tokenizer must encode tags as single tokens or stopping breaks.
- **Adapters/heads device & dtype**: `tina/serve.py` derives target device/dtype from base model; training mirrors dtype in forward passes.
- **Calibration**: `train/eval_loop.fit_confidence_temperature_and_save` writes a JSON; `tina/serve.py` loads it (and optionally plan thresholds/budget clip) to adjust outputs.
- **On-policy decode**: `train/runner.py` calls `train/eval_loop.decode_with_budget`, which must honor **service_config**; runner feeds `think_tokens_used` and `correctness` into **rl_loop**.

### Data contract (strict mode)

Every record MUST have `text` containing **exactly one** `<think>...</think>` and **exactly one** `<answer>...</answer>`. Optional fields enrich training:
- **Budget/plan/conf:** `plan_class`, `target_budget`, `correct`.
- **On-policy fields:** `think_tokens_used` (preferred when present).
- **Strategy/ops:** `style_tag`, `strategy_tags`, `cog_ops`.
- **Quality & checks:** `rubric_score`, `evidence_keys`, `invariants`, `premise_spans`.
- **Analogy/grid:** `analogy_sets`, `elimination_grid`.
- **Rewrite pairs:** `rewrite_pair_id` (for change_19 paraphrase consistency).
- **Reflection & program-of-thought:** `reflection_summary`, `program_solution` (hidden by default at serve).

### Current status snapshot (what works vs. needs fixes)

- ✅ **Hidden CoT, stop/soft-cap parity, leakage guard** wired in serve & eval.
- ✅ **On-policy sampling** available in runner (smoke path) and parity decode exists.
- ✅ **Collator strictness** (exact tags) + diagnostics fields present.
- ⚠️ **change_19_consistency_under_rewrite**: **partial** — missing complete collation of `rewrite_pair_id`, robust pairing logic in the main training loop, and symmetric KL over answer-masked logits with variable-length handling.
- ⚠️ **Main trainer integration** of on-policy decode (beyond smoke path) and gate coverage logs should be standardised.
- ⚠️ **Config guardrails** and CI tests for tag atomicity/IDs and stop IDs across train/eval/serve must be enforced.
- ⚠️ **Calibration blob** should persist plan thresholds & budget clip for serve-time parity.

---

1. Executive Summary

- Scope: Reviewed training-time and serve-time code to verify metacognitive capabilities for hidden reasoning, plan/budget/confidence control, adapter-gated “slow thinking,” and parity of evaluation/decoding at serve time. Applied CoT‑Space lens (optimal chain length via bias–variance trade-off; noise↔length relation) to judge operationalization.
- Verdict: Core segmentation and stop/strip logic PASS. Budgeting via metacog heads and gated side adapters is implemented and wired at serve time PASS (with caveats). Confidence head exists but calibration persistence is missing PARTIAL. Config/schema validation, security hooks, and parity helpers PASS. Missing documentation artifacts (README.md root, MANIFEST.txt, theory PDF) FAIL (docs only).
- Key risks/remediations:
  - Calibration artifacts not saved/loaded at serve time (confidence/plan thresholds, temperature) → Add serializer in training and load in `tina/serve.py` (see remediation notes).
  - Metrics/plots for quality vs budget are not produced end-to-end → Add logging hooks and plotting in training/eval.
  - Some “must include” files absent by name; roles covered by present files (mapped below).

Repo snapshot (Step 0)

- Branch/Commit: `main`, `824c1b8c4013e07b79f9f937ca5cc4d33088fd9e`
- Diff status: clean (one modified file in tools)
- Python: `Python 3.12.3`
- Key libs: transformers/accelerate/trl/peft not detected in this environment (pip freeze had no matches). Code is written to run without them in tests; serve-time paths expect them when running a model.

2. File Inventory & Roles

| file (present/mapped) | role | key APIs/symbols | upstream deps | downstream deps |
| --- | --- | --- | --- | --- |
| `tina/tokenizer_utils.py` | Tokenization + segmentation | `ensure_reasoning_tokens`, `segment_and_masks`, `STOP_SEQUENCES` | HF tokenizer | training masks, tests, serve ensure step |
| `tokenizer_utils.py` | Re-export shim | `from tina.tokenizer_utils import *` | `tina/tokenizer_utils.py` | tests |
| `tina/serve.py` | Serve-time engine | `IntrospectiveEngine`, `EngineConfig`, `_extract_answer`, `StopOnTags` | `tina.tokenizer_utils`, `tina.side_adapters`, `tina.metacog_heads` | CLI (`tina_chat_introspective.py`), eval helpers |
| `serve.py` | Minimal FastAPI stub for tests | `create_app` | FastAPI | tests (`test_serve_visible_hidden.py`) |
| `tina/side_adapters.py` | Side adapters (gated LoRA-like) | `LowRankAdapter`, `IntrospectionScaffold`, `_ctx_think_mask` | torch | `tina/serve.py`, train hooks/losses |
| `side_adapters.py` | Re-export shim | `from tina.side_adapters import *` | `tina/side_adapters.py` | tests |
| `tina/metacog_heads.py` | Lightweight metacog heads (serve) | `MetacogHeads`, `MetacogConfig` | torch | `tina/serve.py` |
| `metacog_heads.py` | Training-friendly heads | `MetacogHeads`, `LegacyMetacogHeads` | torch | tools/observe_infer, tests |
| `tina_chat_introspective.py` | CLI runner | `IntrospectiveEngine` integration; logging | HF, PEFT | user CLI, logs/observe |
| `tools/validate_configs.py` | Config/schema validation | `main()` | schemas, model/config dirs | tests |
| `config/service_config.json` | Service config | host/limits/auth flags | schema | serve app |
| `schemas/*.schema.json` | JSON Schemas | adapter/generation/service | — | `tools/validate_configs.py` |
| `model/Base/config.json` | Base model config | hidden size/layers, ids | — | serve init/config checks |
| `model/Base/generation_config.json` | Generation defaults | sampling params, ids | — | config validation |
| `model/Tina/checkpoint-2000/adapter_config.json` | LoRA adapter config | target modules/LoRA params | — | PEFT loading |
| `train/*.py` | Training scaffolding | `segment_and_masks`, `compute_losses`, reward/metrics | torch | tests; (future) trainer |
| `train_config.yaml` | Intent placeholders | loss weights, phase hints | — | (future) trainer |
| `bandit.yaml` | Security/static analysis | test/include filters | — | CI/static analysis |
| `pyproject.toml` | Lint/type config | ruff/mypy | — | dev tooling |
| Missing by name | Docs/intent | `README.md`, `MANIFEST.txt`, `Training_specification_summary.md`, `tina_training_architect_instruction.yaml`, `tina-project-summary.txt`, `2509.04027.pdf` | — | report mapping below |

3. Dependency Graph & Relationships

- Chat/Serve path: `tina_chat_introspective.py` → `tina/serve.py:IntrospectiveEngine` → `tina/side_adapters.py` (gated think) + `tina/metacog_heads.py` (plan/budget/confidence) + `tina/tokenizer_utils.py` (special tokens) → HF generation.
- Train path: `train/runner.py` uses `tina.tokenizer_utils.segment_and_masks` → `train/data.pad_and_stack` → `train/losses.compute_losses` (answer CE + gate reg + optional Quiet-Star aux) → eval extraction via `tina.serve._extract_answer`.
- Config validation path: `tools/validate_configs.py` validates `model/Base/generation_config.json`, `model/Tina/.../adapter_config.json`, `config/service_config.json` against `schemas/*.schema.json` and performs base↔gen ID coherence and adapter target sanity.
- Observation path: `tools/observe_infer.py` → loads base/(optional)adapter; reuses `tina.serve._extract_answer`; logs heads/gates/sections.

4. Validation Findings (CoT‑Space lens)

4.1 Hidden Reasoning & Answer Segmentation — PASS

- WHAT: Presence of atomic tokens, masks, and stop/strip at serve/eval.
- WHY (CoT‑Space): Enables targeting near‑optimal CoT length L by regularizing inside `<think>` while scoring on `<answer>` only (prevents over/under‑thinking per Thm. 3.4/3.7; Fig. 4).
- HOW: `ensure_reasoning_tokens` asserts tags are single IDs and resizes embeddings if needed; `segment_and_masks` builds M_think/M_answer; serve uses `StopOnTags` to stop on `</answer>` and `_extract_answer` to strip when hidden.
- Evidence:
  - Tokenizer tags added and atomic, with round‑trip guard:
    - tina/tokenizer_utils.py:12 and 24–33
      ```python
      REASONING_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]
      # Round-trip atomicity check
      tok_ids = tokenizer.encode(t, add_special_tokens=False)
      if len(tok_ids) != 1:
          raise ValueError(...)
      ```
  - Masks constructed from spans:
    - tina/tokenizer_utils.py:61–94
      ```python
      think_mask = [0] * L; answer_mask = [0] * L
      # between <think>...</think>
      for i in range(t0 + 1, t1): think_mask[i] = 1
      # between <answer>...</answer>
      for i in range(a0 + 1, a1): answer_mask[i] = 1
      ```
  - Serve-time stop/strip:
    - tina/serve.py:35–49 (StopOnTags), 25–33 (strip helper), 330–333 (hidden returns stripped answer)
  - Parity in eval (same extractor):
    - train/eval_loop.py:4–7 → `_extract_answer` reuse
- Relationships:
  - Data/collator → masks: `segment_and_masks()` → `train/data.pad_and_stack()` → labels masked by answer in `train/runner.sft_one_step_smoke()`.
  - Serve/eval parity: `_extract_answer` used by both serve and eval utilities.

4.2 Metacognitive Heads: plan / budget / confidence — PARTIAL PASS

- WHAT: Heads tap hidden states to produce plan logits, a real/integer budget, and confidence; serve uses budget to cap `<think>`; outputs recorded.
- WHY (CoT‑Space): Budget head lets L adapt per instance difficulty (Thm. 3.1, Remark 1), avoiding brittle long chains (Thm. 3.4) and under‑budget truncation (Thm. 3.7). Confidence enables self‑assessment and future redo triggers.
- HOW: Serve registers hooks on decoder layers at tap indices, runs a dry forward with `output_hidden_states=True`, calls heads, clamps/scales, and uses the predicted budget to set a stopping criterion for `<think>`.
- Evidence:
  - Tap hooks and dry forward capture hidden states:
    - tina/serve.py:90–107 (register_forward_hook), 109–119 (dry forward with `output_hidden_states=True`)
  - Heads construct and clamp outputs; budget scaled to `B_max`:
    - tina/metacog_heads.py:21–29 and 43–53
      ```python
      self.plan = nn.Linear(D, 4); self.budget = nn.Linear(D,1); self.confidence = nn.Linear(D,1)
      plan_logits = (self.plan(z) / temp).clamp(-20, 20)
      budget = (budget_raw * B_max).clamp(0, B_max)
      confidence = torch.sigmoid(...)
      ```
  - Serve uses head outputs to decide budget, with min/max guards, and records stats:
    - tina/serve.py:123–133 (select budget, min/max), 164–170 (use in StopOnTags), 127–132 (stats)
- Gaps:
  - Calibration artifacts (temperature, thresholds) are not saved/loaded across train/serve. The `train/metrics.py` provides temperature/ECE tools, but there’s no persistence or serve‑time ingestion.
- Remediation:
  - Add training‑time export: save `plan_thresholds.json` and `confidence_calibration.json` (temperature/iso parameters).
  - Load in `IntrospectiveEngine.__init__` and set `MetacogConfig.head_temp` and any decision thresholds.

4.3 Side Adapters / LoRA with think‑segment gating — PASS

- WHAT: Additive residual low‑rank adapters gated globally (`gate` param) and per‑token via a hard‑concrete unit, activated during `<think>` only; optional mask `_ctx_think_mask` narrows deltas to think tokens at training.
- WHY (CoT‑Space): Focus capacity on reasoning segments and penalize gratuitous activation; reduces overfitting from long CoTs (Thm. 3.4; Remark 2).
- HOW: `IntrospectionScaffold.think()` toggles a contextvar so hooks mutate layer outputs only in think mode; adapter returns `h + gated(delta)` and multiplies by training mask when provided.
- Evidence:
  - Gate default 0; per‑token gate + mask:
    - tina/side_adapters.py:28–36 (gate init), 57–77 (hard‑concrete gate + `_ctx_think_mask` masking)
  - Think‑mode conditional in hooks and context manager:
    - tina/side_adapters.py:41–55 (hook_fn applies only when `_think_flag` true), 79–96 (`think()` context manager)
  - Serve enables think mode around the THINK step only:
    - tina/serve.py:191–196 and 223–239 (generation under `with self.scaffold.think(): ...`)
  - Regularizer present:
    - train/losses.py:6–22, 24–55 (`gate_sparsity_regularizer` and inclusion in `compute_losses`)

4.4 Train/Serve Parity — PASS

- WHAT: Same stop/strip logic for eval and serve; hidden CoT by default; parity tests.
- WHY (CoT‑Space): Ensures measured L/accuracy matches deployed behavior.
- Evidence:
  - Shared `_extract_answer` used by eval and serve: tina/serve.py:25–33; train/eval_loop.py:4–7
  - Stop on `</answer>` in both serve and tools: tina/serve.py:35–49; tools/observe_infer.py:37–54
  - Default hidden CoT: `EngineConfig.visible_cot: False` (tina/serve.py:13–18); CLI toggles it (tina_chat_introspective.py:288–299)
  - Tests: `tests/test_serve_parity.py`, `tests/test_serve_visible_hidden.py`, `tests/test_cot_assembly.py`

5. Dataset & Collation (Step 3) — PASS (lightweight)

- WHAT: Batched segmentation and masks, including “no‑think” possible when spans absent.
- HOW: `segment_and_masks` returns `(input_ids, attention_mask, loss_mask, think_mask, answer_mask)`; `pad_and_stack` pads and stacks to tensors; labels masked to answers in smoke test.
- Evidence:
  - Collator: train/data.py:5–38
  - SFT smoke: train/runner.py:86–115 (`labels` masked by loss mask); extraction parity validated.

6. Parameter Groups & Regularizers (Step 4) — PARTIAL

- WHAT: Gate sparsity regularizer and auxiliary Quiet‑Star loss present; explicit optimizer parameter groups and staged freezing policies are not implemented in code here (not a full trainer yet).
- Evidence:
  - Regularizers and aux: train/losses.py:6–55; train/schedules.py:26–39 (context to enable aux).
- Remediation: When integrating a trainer, define optimizer groups for `{heads, adapters, base_top}`; keep heads in fp32; apply gradient clipping for long contexts.

7. Losses & Objectives (Step 5) — PARTIAL PASS

- WHAT: `answer_ce` required; `gate_reg` supported; optional Quiet‑Star auxiliary; RL/reward shaping includes token‑cost via over‑budget penalty and formatting bonus; think CE and DPO/GRPO hooks are config‑hinted but not implemented.
- Evidence:
  - CE + gate reg + aux: train/losses.py:24–55
  - RL shaping proxy: train/reward.py:1–20 and 47–74
  - Config hints: train_config.yaml:5–17
- Remediation: Add optional `think_ce` masking with anneal schedule, and DPO/GRPO path that penalizes think tokens and leakage explicitly.

8. Metacognition Training Details (Step 6) — PARTIAL

- WHAT: Tap layers are configurable in serve/CLI; calibration utilities exist; no serialization of calibration blobs; budget treated as soft cap at serve time via stopping criterion.
- Evidence:
  - Taps in serve config: tina_chat_introspective.py:244–253
  - Budget soft cap: tina/serve.py:164–170 (StopOnTags with `max_new=budget`)
  - Calibration utils: train/metrics.py:1–40 (temperature_fit), 42–64 (ECE/Brier)
- Remediation: Export temperature and thresholds on validation; load and apply in `MetacogConfig` at serve.

9. Adapters Training Process (Step 7) — PASS (unit-tested behaviors)

- WHAT: Segment‑conditioned gating via context var; schema validation of adapter targets; concurrency isolation.
- Evidence:
  - Context mask plumbing: train/hooks.py:1–18; tests `test_adapters_gate.py`, `test_gate_context_leakage.py`.
  - Adapter targets validated: tools/validate_configs.py:107–116 checks match for Qwen2.
  - Thread isolation: tests/test_adapters_concurrency.py

10. Segmentation Training & Stopping (Step 8) — PASS

- WHAT: Tags asserted atomic; stop on `</answer>`; token ID harmony checks.
- Evidence:
  - Atomic tags tests: tests/test_tokenizer_tags.py, tests/test_tokenizer_roundtrip.py
  - Stop sequences include `</answer>`: tests/test_stop_sequences.py
  - Config ID harmony: tools/assert_config_ids.py with tests/test_config_ids.py

11. Validation Metrics & Integration (Step 9) — PARTIAL

- WHAT: Metrics utilities exist (ECE, Brier); observation tool logs think/answer token counts and gate activity; but no end‑to‑end plotting/curves in repo.
- Evidence:
  - Metrics: train/metrics.py
  - Observe tool: tools/observe_infer.py (logs `sections`, `heads`, `gates`)
- Remediation: Add training logs for think_tokens_used, budget error, plan accuracy; produce quality vs budget curve (Fig. 4 support).

12. Training Phases (Step 10) — PARTIAL

- WHAT: Phases are specified in `train_config.yaml`, but there is no multi‑phase trainer here.
- Remediation: Implement phases A–D with staged freezing and calibration/pruning; save artifacts.

13. Engineering Artifacts & Schema Checks (Step 11) — PASS

- WHAT: JSON Schemas for generation/adapter/service; startup validation script; security config.
- Evidence: `schemas/*.schema.json`, `tools/validate_configs.py`, `bandit.yaml`.

14. Acceptance Criteria (Step 12)

- Tokenizer round‑trip preserves tags; </answer> stopping works → PASS (tests cover).
- Metacog heads budgets bounded; plan/conf present → PASS (bounds); calibration persistence → FAIL (missing persistence).
- At fixed/reduced think budget, quality ≥ baseline; leakage ≈ 0 with visible_cot=false → PARTIAL (framework supports; no benchmark logs).
- Serve‑time extractions match eval‑time extraction → PASS.

Remediation Summary

- Add calibration artifact I/O:
  - Train: fit temperature/thresholds; save JSON (e.g., `artifacts/metacog_calibration.json`).
  - Serve: load file in `IntrospectiveEngine.__init__` and set `MetacogConfig.head_temp`; apply thresholds to `plan`/budget decisions.
- Add trainer scaffolding: optimizer param groups, gradient clipping, staged freezing; implement `think_ce` with annealing; optional DPO with token‑cost and leakage penalty.
- Metrics/plots: compute think/answer token counts, plan accuracy, budget error, ECE; produce quality vs budget curves per Fig. 4.
- Docs: provide `README.md`, `Training_specification_summary.md`, architect instruction YAML, business summary, and include the CoT‑Space PDF (2509.04027.pdf) for traceability.

Complete List of Files Fulfilling Spec (by role)

- Tokenization/segmentation: `tina/tokenizer_utils.py`, `tokenizer_utils.py`
- Serve engine/parity: `tina/serve.py`, `serve.py`, `train/eval_loop.py`
- Metacog heads: `tina/metacog_heads.py`, `metacog_heads.py`
- Side adapters/gating: `tina/side_adapters.py`, `side_adapters.py`, `train/hooks.py`
- Training scaffolding: `train/data.py`, `train/losses.py`, `train/runner.py`, `train/metrics.py`, `train/reward.py`, `train/schedules.py`, `train_config.yaml`
- Configs/schemas/tools: `model/Base/*`, `model/Tina/checkpoint-2000/adapter_config.json`, `config/service_config.json`, `schemas/*.schema.json`, `tools/validate_configs.py`, `tools/assert_config_ids.py`
- Observation/validation: `tools/observe_infer.py`, `tools/validate_observation.py`, `tools/validate_activation_log.py`
- Security/tooling: `bandit.yaml`, `pyproject.toml`

5. Metrics Summary (what’s available now)

- Functional: extraction on `<answer>` only (tests). No EM/F1 scaffolding in repo.
- Reasoning-use proxies: `tools/observe_infer.py` logs `think_tokens`/`answer_tokens` and gate activity.
- Calibration: `train/metrics.py` can compute ECE/temperature; no persistence.
- Ops: Observation tool logs latency and tokens/sec.

6. Reconstructed Recursive Inference Loop

- Plan: Dry forward; tap layers; heads compute `plan_logits`, `budget`, `confidence` (tina/serve.py:109–133).
- Budget: Integer cap b = clamp(min_think, budget, budget_cap) (tina/serve.py:123–124).
- Think: Generate with adapters ON under `with scaffold.think()`; stop on `</think>` or when step count reaches b (tina/serve.py:168–185, 223–239).
- Assess: Record plan/conf/budget in `last_stats`; confidence currently observational.
- Answer: Append `<answer>` and generate; stop on `</answer>` or token cap (tina/serve.py:258–310).
- Finalize: If `visible_cot=False`, extract the text inside `<answer>...</answer>` and return (tina/serve.py:330–333). Else return raw concatenation.

7. Appendix: Raw Evidence and IDs

- Special token IDs logged/resolved via `ensure_reasoning_tokens()` in `tina/tokenizer_utils.py`.
- Config harmony checks in `tools/assert_config_ids.py` and `tools/validate_configs.py`.
- Tests cover: token atomicity, stop sequences, gating isolation, metacog bounds, parity, and reward sensitivity.



