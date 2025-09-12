## Interdependence of Scripts, Configs, and Architectural Components (Brief for new coder)

This document serves as a concise primer for the next developer joining the _Small‑Metacognitive‑LLM_ project. It summarises the main scripts and configuration files, explains how they interact to achieve the project’s metacognitive objectives, and highlights the outstanding tasks that must be resolved before implementing the next round of features (changes 20–33).

### Project goals (high‑level)

The overarching aim is to teach a small language model **how to think**, not just mimic chain‑of‑thought (CoT) syntax. Specifically, the model should:

1.  **Remain additive**: keep the base LM untouched and attach only lightweight enhancements—LoRA‑style side‑adapters, metacognitive heads and optional probes. This ensures the base can still benefit from general checkpoints and avoids destructive modifications.
2.  **Learn to plan and budget its reasoning**: a small head should predict a plan type (e.g., “short answer”, “deliberate reasoning”), a token budget for the <think> span, and a calibrated confidence. During inference, generation should respect these budgets via a soft cap (slack ratio) and stop sequences.
3.  **Hide internal reasoning by default**: training samples include <think>…</think> and <answer>…</answer> segments; the serve pipeline must strip the hidden reasoning unless explicitly requested and enforce strict stop criteria to prevent leakage.
4.  **Optimise for quality vs. cost**: auxiliary losses (plan/budget/confidence) and RL loops encourage accurate answers while discouraging unnecessary thinking. The reward functions penalise token usage beyond the budget.
5.  **Develop strategy and self‑reflection capabilities**: later changes (20–33) will add heads for cognitive operations, strategy tags, subgoals, invariants, analogy constraints, reflection summaries, and program‑of‑thought generation. These features are designed to teach reasoning styles and self‑assessment rather than pattern‑matching surface forms.

### Key components and their roles

| Component | Purpose | Key references |
| --- | --- | --- |
| tina/tokenizer_utils.py | Adds special tags (<think>, </think>, <answer>, </answer>, and later <reflect>, <program>). It verifies that each tag encodes as a single token and provides helpers to build masks from tagged text. Failure to enforce atomic tags would break stopping and segmentation. | Tag checks are used by tina/serve.py and by the train/data.py collator. |
| tina/side_adapters.py | Implements LoRA‑style residual adapters with hard‑concrete gates. These gates activate only during <think> segments (via scaffold.think() context) and record coverage statistics. This keeps the base transformer unchanged and enables segment‑conditional adaptation. | Gating and coverage logic ensures adapters do not alter answer tokens. |
| tina/metacog_heads.py | Pools hidden states from tapped transformer layers and outputs: (i) plan logits (which classifies the reasoning regime), (ii) a budget cap (max number of think tokens), and (iii) a raw confidence logit. Extensions add heads for strategy tags, cognitive operations, subgoals, constraint vectors, reflection, program‑of‑thought, etc. | Predictions are used during training (auxiliary losses) and inference (dynamic budget selection). |
| tina/serve.py | Wraps the base model and adapters with an IntrospectiveEngine. It reads config/service_config.json for stop sequences and slack ratio, loads the calibration blob for confidence scaling and optional plan/budget thresholds, and exposes generate_cot(...) to produce <think>…</think><answer>…</answer> pairs. It strips reasoning by default, counts think tokens, and logs metrics (plan, budget, confidence, leakage). | Implements StopOnTags, SlackStop, dynamic budget selection, and leakage guard. |
| train/data.py | Defines the JSONL training record contract. Each line must contain text with exactly one <think>…</think> and one <answer>…</answer> pair. The collate function tokenizes text, builds masks (think_mask, answer_mask, loss_mask), and attaches labels such as plan_class, target_budget, correct, difficulty_bin, and on‑policy think_tokens_used. It can also handle strategy tags, cognitive operations, constraint vectors, rewrite pair IDs, reflection summaries, program solutions, step types, subgoal IDs, elimination grids, etc. | Enforces strict dataset integrity; any missing or extra tags cause a ValueError in strict mode[1]. |
| train/losses.py | Provides the primary answer CE loss and auxiliary losses for plan/budget/confidence heads, gate sparsity, quiet‑star (self‑distillation) loss, and RL reward shaping. Recent changes begin to incorporate rewrite consistency by adding symmetric KL divergence on answer logits for paraphrased thought pairs (change_19). This file will house additional objectives like style invariance, strategy classification, constraints, NLI contradiction penalties, analogical consistency, etc. | Losses are weighted via config. |
| train/runner.py | Orchestrates training loops. It reads the train config, initializes the model, adapters, and heads, builds the DataLoader via the collate function, and executes _train_step on each batch. It logs metrics, performs occasional on‑policy decoding using decode_with_budget (to observe actual think lengths), and updates the Gaussian budget policy via rl_phase_step. Only a smoke test currently shows on‑policy sampling; the main loop must be extended to call it. | Must integrate on‑policy decode, RL updates, gate coverage logging, and rewrite grouping. |
| train/eval_loop.py | Defines decode_with_budget, which uses service_config stop tags and slack ratio to generate think and answer segments. It counts the number of think tokens and ensures </think>/</answer> are produced. Also implements evaluation metrics (accuracy, F1, ECE), fits confidence temperature (and optionally plan thresholds/budget clip) into a calibration file, and exposes quality_vs_budget_curve to sweep budgets and produce a CSV. | Must maintain parity with serve; any difference in stop tags or slack ratio will cause train/serve skew[2]. |
| train/metrics.py | Houses helper functions to compute calibration errors (ECE), Brier score, and to aggregate metrics by slices (e.g., plan source, difficulty bin). Later changes will extend it to include strategy, cognitive operation and constraint slice metrics. |  |
| train/rl_loop.py | Implements a critic‑free Gaussian budget policy using REINFORCE (or optionally DPO) to update budget predictions based on on‑policy think length and correctness. |  |
| config/service_config.json | Defines stop sequences for think and answer, default visibility, soft‑cap slack ratio, and the path to the calibration blob. It must be loaded consistently by both training and serving code to avoid mismatched stopping. |  |
| adapter_config.json and generation_config.json | Provide LoRA target modules, rank, alpha, dropout settings, and generation defaults (temperature, top‑p) respectively. Both training and serving should honour these parameters. |  |
| tools/validate_configs.py | Should validate that the tokenizer encodes each special tag as a single token, that stop sequences match across train/eval/serve, and that config fields are present. This is essential for preventing subtle errors. |  |

### Unresolved areas / Partial implementations

1.  **change\_19 (rewrite consistency)** – Only partially integrated. The dataset can accept a rewrite\_pair\_id, but the main training loop does not yet group examples by ID nor compute the symmetric KL divergence over answer logits (masked/truncated to common length). Tests for handling variable lengths, order‑invariance, and correct weighting are still needed.
2.  **On‑policy sampling in main loop** – The smoke test demonstrates on‑policy decoding and RL updates, but these have not been integrated into the standard training loop (they live in onpolicy\_sft\_and\_rl\_smoke). As a result, the model still trains primarily on teacher‑forced data and does not learn to allocate budgets from its own generations. This must be wired with a configurable cadence.
3.  **Gate coverage logging** – Side adapters track gate activity and coverage, but this metric is not yet surfaced in training or evaluation logs. Without it, we cannot verify that adapters activate only on think tokens.
4.  **Config and tag guardrails** – There is currently no automated check that service\_config.json stop tags match those used in decode\_with\_budget or that the tokenizer encodes tags as single tokens. A single mismatch could cause truncated or leaked outputs.
5.  **Calibration blob parity** – The confidence temperature is saved, but plan thresholds and budget clips are not persisted. Without persisting these values, inference cannot apply the same plan/budget selection logic used in evaluation.
6.  **Strict dataset contract** – The collate function enforces one pair of <think> and <answer> tags[\[1\]](https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/train/data.py#L60-L61), but the error messages may not be explicit enough for easy debugging. In strict mode, malformed records should raise clear exceptions with indices.
7.  **Reward token counts** – The reward function allows fallback to word counts when a tokenizer is absent, but this can misalign budgets. It is critical that training always passes a tokenizer to get accurate token counts for budget penalties and RL rewards.
8.  **Answer invariance under strategy hints** – Upcoming features (strategy tags, cognitive operations) rely on injecting hints like <strategy:probe>. Hidden mode must strip these hints and confirm they do not alter the final answer.

### Immediate next steps (fixes before changes 20–33)

To proceed safely with the more ambitious strategy‑level features, we must solidify the foundations. The following fixes need to be implemented and tested:

1.  **Complete rewrite consistency (change\_19)** – Accept rewrite\_pair\_id and group pairs in the collate. Compute a symmetric KL divergence over answer‑token distributions in train/losses.py, truncating to the minimum length. Log the loss and verify via tests.
2.  **Integrate on‑policy decoding & RL updates** – Add a sample\_every configuration and call decode\_with\_budget in the main training loop. Store think\_tokens\_used in the batch, then update the Gaussian budget policy via rl\_phase\_step. Log RL statistics and verify presence of think\_tokens\_used.
3.  **Gate coverage telemetry** – After each forward pass, average each adapter’s \_last\_gate\_coverage and log it. In evaluation, record this in the CSV/JSON outputs to monitor gating behaviour.
4.  **Strict config validation** – Enhance tools/validate\_configs.py to check tag atomicity and stop sequence parity. Add a command‑line flag to run these checks before training or serving. Fail fast on mismatch.
5.  **Calibration persistence** – Modify the calibration function to persist plan thresholds and budget clips alongside confidence temperature. Load them in tina/serve.py and apply them when selecting budgets and plans.
6.  **Reward token basis** – Always pass a tokenizer to reward\_fn so that budgets are based on token counts. Add tests demonstrating that token‑dense but word‑sparse thoughts incur higher penalties.
7.  **Dataset integrity** – Improve error messages when tags are missing and respect the strict parameter to allow fallback wrapping if necessary. Add tests to ensure malformed data is caught.
8.  **Answer invariance checks** – Ensure that control tokens (like <strategy:...>) do not leak into answers or change them. Add a helper in evaluation to test invariance across hints.
9.  **Wiring config across scripts** – Provide a single train\_config.yaml that defines model paths, LoRA settings, data locations, budgets and lambda weights. All scripts should load from this config rather than hard‑coded values. A run manifest capturing config hashes should be emitted for reproducibility.

By addressing these issues first, the new coder will have a solid base to implement the higher‑level reasoning enhancements (changes 20–33). The following YAML blocks (fix\_01 to fix\_13) provide actionable implementation details for each fix, including modifications, tests, and pass‑fail criteria. The final “wiring\_instructions” block explains how to glue all configurations and scripts together.

[\[1\]](https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/train/data.py#L60-L61) GitHub

[https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/train/data.py](https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/train/data.py)

[\[2\]](https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/tina/serve.py#L342-L358) GitHub

[https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/tina/serve.py](https://github.com/justdtip/Small-Metacognitive-LLM/blob/main/tina/serve.py)