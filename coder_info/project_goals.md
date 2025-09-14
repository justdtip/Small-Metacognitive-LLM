
# Small Metacognitive LLM — Updated Requirements & QA Addendum  
**Version:** 14 September 2025, 10:58 AEST • **Owner:** Engineering • **Scope:** Metacognitive feedback, token budgeting, TUI, data streaming, training & evaluation

---

## 1) What we’re building (high‑level)

**Goal:** Equip a small decoder‑only LLM with a lightweight **metacognitive layer** that (a) inspects per‑layer hidden states, (b) produces **plan / budget / confidence** signals, and (c) **modulates** generation through a **learned feedback gate** and a **token‑budget controller**—so the model learns **when** and **how much** to think and **what strategy** to prefer.

**Desired outcomes:**
- **Better reasoning at small scale** via *internal* slow‑thinking (not just performative CoT).
- **Emergent specialisation** of per‑layer “expert” heads and **learned aggregation**.
- **Token budget discipline** (avoid chronic under/over‑thinking).
- **End‑to‑end training pipeline + TUI** that make runs observable and debuggable.

**Theoretical basis (CoT‑Space):**  
Treat reasoning as **state optimisation** in a continuous semantic space; **CoT length** plays the role of a **learning‑rate** knob—too short underfits, too long overfits—so an **optimal L** exists. Generalisation error scales with the **total CoT capacity** (≈ E[L]·E[|ξ|]·log|A|), motivating token‑budget control and problem‑conditional thought allocation. fileciteturn1file0  
*(See the optimisation analogy and L‑as‑LR insight; also the bias–variance trade‑off and bounds discussed in the paper; e.g., discussion around Fig. 3 and Theorems 3.1/3.4.)* fileciteturn1file0

---

## 2) Architecture (current & required)

### 2.1 Base LM (decoder‑only)
- Causal HF model; **enforce left‑padding** (`tokenizer.padding_side="left"`) for correct causal masking.
- Optional LM adaptation: **ln‑only** or **LoRA**, with **last‑K** or **unfreeze‑all**, and a **low LR multiplier** (e.g., 0.05×).

### 2.2 Metacognitive module
- **Per‑layer taps**: capture **last‑token** hidden state from each transformer block.
- **Shared trunk (PerLayerTrunk)**: 2‑layer MLP projects each layer’s state → `proj_dim`.
- **Per‑layer heads (“experts”)**: for each layer produce `plan_logits_pl[L,K]`, `budget_raw_pl[L,1]`, `conf_raw_pl[L,1]`.
- **Aggregator (LayerAggregator)**: attention/mean over per‑layer projections → global vector `g` and weights `α[L]`.
- **Global heads from `g`**: final **plan**, **budget**, **confidence**.
- **Variance regulariser (optional)**: penalise high variance across **per‑layer budget** predictions.
- **Telemetry**: expose `α`, per‑layer logits, plan‑agreement, budget variance.

> **Where “experts” live:** they’re **per‑layer heads** (one head set per transformer block) that share the trunk. The aggregator learns **which layers to trust** per problem—i.e., a learned mixture‑of‑layers.

### 2.3 Feedback mechanisms
1) **Token‑budget controller (mandatory):** predicted **budget** (clamped) caps **THINK** tokens in **serve + train**.  
2) **Feedback gate (recommended):** register a **forward hook** on a chosen decoder layer; multiply its activations by a **gate vector** computed from `g` (and/or plan/confidence). One‑way: no metacog gradient shortcut into base LM—only standard backprop through losses. Hook is applied **in training as well**, so LoRA/LN/unfrozen layers **see** the modulation and co‑adapt.  
3) **Sampler modulation (optional):** adjust temperature/top‑p/penalties per segment from plan/confidence (decoding‑side only).

---

## 3) Data & token budgeting

- **Data ingestion:** HF streaming by default (e.g., `glaiveai/reasoning-v1-20m`) with TUI toggle to use local files.  
- **Mixed domains** (math, logic, code, QA) to encourage **problem‑conditional specialisation**.  
- **Max‑length probing:** during streaming, **periodically sample** records to update `max_len` upward if a longer sequence appears (never shrink).  
- **Left‑padding** enforced to avoid decoder‑only padding warnings and mis‑masking.

**Why budgeting matters (CoT‑Space):** Learn **optimal CoT lengths** per problem/model capacity; token budgeting is the internal policy that operationalises the optimal‑L principle. fileciteturn1file0

---

## 4) Training pipeline (must‑haves)

1) **Losses**
   - **Answer CE / RL objective** (existing).
   - **Plan loss** (labels or auxiliary consistency).
   - **Budget regression**: MSE between predicted budget and **actual THINK tokens used**; weight `lambda_budget_reg`.
   - **Confidence calibration** (BCE/ranking vs. correctness if available).
   - **Variance regulariser** across per‑layer budgets; weight `var_reg`.
2) **Budget enforcement at train time**  
   Use predicted budget to bound THINK segment; log `think_tokens_used`.
3) **Feedback gate during training**  
   Register the same forward hook around the model call; remove it after each step.
4) **Optimiser param‑groups**  
   Base LM (unfrozen subsets) on a **low LR multiplier** (e.g., 0.05×); metacog/LoRA on full LR.
5) **Checkpoints & mini‑eval**  
   Save every **1000 steps**; every **100 steps** run a tiny eval with simple logic puzzles; render outputs in TUI.

---

## 5) Training UI (TUI)

**Aesthetic:** dark “eDex/HUD” style reused across training and the launcher (replaces CLI args).

**Controls (toggle/adjust):**
- **Data:** dataset name or path, streaming on/off, split, sample limit.
- **Model:** model path (supports `model/Tina`), precision, grad‑checkpoint, flash‑attn.
- **Metacog:** `linked_all_layers`, `proj_dim`, `agg` (attn/mean), `var_reg`, **feedback gate on/off & layer index**, **use predicted budget** on/off.
- **LM adaptation:** enabled, mode (`ln_only`/`lora`), **last‑K** or **unfreeze‑all**, **LR multiplier**.
- **Run:** steps, batch/grad‑accum, save interval (1000), eval interval (100), seed.
- **Max length:** initial cap + **auto‑probe** (probe interval).
- **Logging:** console + JSONL + TB/W&B.

**Metrics panels (live):**
- **Global:** loss/entropy/reward; plan/exec/eval tokens & fractions; LR; grad‑norm; GPU RAM.
- **Metacog:** plan distribution; predicted budget vs. used; confidence histogram; **α stats** (min/mean/max & entropy); **per‑layer plan agreement**.
- **Feedback:** gate norms and per‑time heatmap for the gated layer.
- **Data:** current `max_len`, longest seen, left‑padding status.
- **Eval:** accuracy on canned puzzles; sampled generations.

**Defaults:** streaming **on**; metacog + feedback **on**; LM adaptation **enabled** (ln‑only, last‑K=2, LR mult=0.05); checkpoints every **1000**.

---

## 6) Wiring & dependencies

- **Base LM** → exposes last‑token states per block.
- **Trunk** → projects each layer state.
- **Per‑layer heads (“experts”)** → per‑layer plan/budget/confidence.
- **Aggregator** → attention/mean over layers → `g`, weights `α`.
- **Global heads** → final plan/budget/confidence.
- **Budget controller** → predicted **budget** caps THINK (serve + train).
- **Feedback gate (optional)** → multiplies chosen layer output by gate vector from `g`.
- **Sampler modulation (optional)** → temperature/top‑p from plan/confidence.
- **Losses** → combine main objectives with plan/budget/confidence auxiliaries and variance regulariser.
- **TUI** → config source; deep‑copy on start so defaults are restored after run.

---

## 7) Acceptance criteria

1) **Tokenizer left‑padding** active; no right‑padding warnings.  
2) **Budget active at train time:** `think_tokens_used` > 0 and varies; budget MSE non‑zero when `lambda_budget_reg>0`.  
3) **Feedback gate in train (if enabled):** hook registered; LM gradients depend on gate when adaptation is on.  
4) **Per‑layer telemetry:** `α` stats, plan agreement %, budget variance rendered.  
5) **Max‑len auto‑probe** bumps `max_len` on longer sequences (never shrinks).  
6) **Checkpoints** **every 1000**; evals **every 100** print puzzle outputs.  
7) **TUI** runs without CLI args; large numeric inputs accepted; settings revert to defaults after run.  
8) **Unfreeze‑all** works; base LM uses low LR multiplier; non‑zero grads observed on unfrozen params.

---

## 8) Risks & mitigations

- **Budget not learning** → ensure **both** enforcement (cap) **and** MSE supervision; verify gradients to budget head.
- **Over‑regularisation** from variance penalty → keep `var_reg` small (1e‑4–1e‑3).
- **Instability when unfreezing** → small LR multiplier (≤0.05), grad‑clip, EMA if available.
- **Padding mistakes** → assert `padding_side="left"`; add TUI status light for padding.

---

## 9) Why this matches CoT‑Space (in practice)

We implement **problem‑conditional CoT length control** by learning and enforcing a **token budget** (internal policy), and specialise **which layers to trust** via an attention aggregator over per‑layer “experts.” This instantiates the paper’s view of **reasoning as optimisation** with **L as a noise/step‑size control** and the bias–variance trade‑off that yields an **optimal L**. fileciteturn1file0

---

# Addendum — Strict QA & Test‑Failure Triage (Do Not Bypass Functionality)

**Policy:** Tests must pass by **making the intended features correct**, **not** by stubbing/negating functionality. All fixes must preserve the architecture and behaviours described above.

## A) Immediate test failures and expected fixes

1) **Calibration blob / thresholds / clip (TypeErrors)**
   - **Symptom:** `test_calibration_blob_extend.py::test_thresholds_and_clip_applied` TypeError; `test_heads_calibration_blob.py` similar.
   - **Fix:** Ensure calibration blob **persists** thresholds in a deterministic file/section and **applies** them during forward. Validate I/O schema and dtypes; add unit asserts that post‑load values are floats in expected ranges and clipping is applied exactly once.

2) **Periodic checkpoint saving (missing path)**
   - **Symptom:** `test_checkpoint_saving.py` expects `/tmp/pytest-.../checkpoint-XXXX` but file missing.
   - **Fix:** Create directories proactively, write atomic via temp+rename, and call save on the exact modulus (e.g., `if step % 1000 == 0:`). Log absolute path.

3) **Decode parity / service config tags (ValueError)**
   - **Symptom:** Stop tag `</THK>` missing (`test_decode_parity_with_service_config.py`). 
   - **Fix:** Centralise tag constants (THINK/EXEC/EVAL) and inject into both **training** and **serve** configs; assert their presence at startup. Add tolerant parser that errors with a clear message if tags are absent.

4) **Gate‑coverage metrics = 0** (`eval`, `mean`, `logging` variants)
   - **Symptom:** Coverage expected > 0 or < 0.99, but got 0.0. 
   - **Fix:** During THINK, record a **binary gate mask** for the hooked layer and compute coverage = mean(mask) over think tokens. Ensure THINK spans are non‑empty in tests; avoid zero‑division and honour left‑padding masks.

5) **Heads training integration — `per_layer` NameError**
   - **Symptom:** Several tests reference `per_layer` in losses/train integration.
   - **Fix:** Always return a `per_layer` dict from the heads forward in **debug** and thread it through the training step (aux) and losses. When unavailable, return an **empty but present** dict to avoid NameError.

6) **On‑policy decode — think length missing**
   - **Symptom:** `test_onpolicy_decode.py` expects `think_len` recorded.
   - **Fix:** Log `think_tokens_used` every on‑policy step into the batch metrics. Include in JSON output.

7) **Quality budget curve — monotonicity**
   - **Symptom:** `test_quality_budget_curve.py` expects a monotone relation (e.g., more budget → ≥ think tokens) but observed mismatch.
   - **Fix:** Ensure predicted budget is **clamped** and **enforced**; when sampling with a cap B, never exceed B. Add an assertion in debug mode to fail fast if exceeded.

8) **Style entropy bonus — NameError (per_layer again)**
   - **Fix:** Same as (5); expose `per_layer` consistently to the bonus computation and metrics code.

9) **Observe/infer smoke — HF repo id validation error**
   - **Symptom:** `HFValidationError: Repo id must be ...` for `test_observe_infer_smoke.py`.
   - **Fix:** When `model_name_or_path` points to a local folder (e.g., `model/Tina`), skip HF push/resolve paths and operate purely local; guard any Hub calls behind a config flag (`hub_push: false`) or environment check (`HF_HUB_OFFLINE=1`).

10) **Train entrypoint — config load assertion**
    - **Symptom:** `test_train_entrypoint.py::test_end_to_end_config_load` asserts False.
    - **Fix:** Ensure the **TUI launcher** writes an explicit, validated config; the runner **deep‑copies** it; and defaults restore after run. Add a minimal smoke path for non‑TUI tests that loads from file without interactive deps.

## B) Cross‑cutting QA gates (must be enforced in CI)

- **Tokenizer padding assertion**: fail if not `padding_side="left"` for decoder‑only base.  
- **Budget activation check**: during train, assert `think_tokens_used` is recorded and `budget_mse` is present when enabled.  
- **Feedback gate parity**: if gate is enabled in serve, assert the training loop registers & removes the same forward hook.  
- **Max‑len probe**: when streaming, assert `max_len` can **increase** (if longer examples appear) and never decreases mid‑run.  
- **Checkpoint cadence**: assert a file exists at exactly every 1000 steps (or configured period).  
- **Per‑layer telemetry**: assert `per_layer` is present in debug runs and `α` stats are non‑NaN.  
- **No test bypass**: forbids stubbing out functionality just to satisfy tests.

## C) Sign‑off checklist (release‑blocking)

- All tests above pass with **intended functionality preserved**.  
- Acceptance criteria in §7 are green.  
- TUI reproduces runs without CLI and restores defaults after exit.  
- Logs show non‑zero, varying `think_tokens_used`; budget MSE > 0 when enabled; `α` entropy finite; plan‑agreement reported.  
- No Hub calls when a local model path is used (unless explicitly enabled).

---

**References:** CoT‑Space paper excerpts and figures informing optimal CoT length, noise–length inverse relation, and bias–variance framing (see discussion and theorems). fileciteturn1file0
