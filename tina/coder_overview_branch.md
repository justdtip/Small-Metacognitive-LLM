# Introspective Tina & AZR Branch: Structure, Intentions and Expected Changes

## 1 Project overview: _Tina_ introspective reasoning

The **Tina** project is a research platform for building small language models capable of reasoning with hidden chain‑of‑thought and self‑monitoring. It adds an _introspective scaffold_ on top of a base causal LLM such as Qwen1.5 B. Key features include:

·      **Reasoning tags.** The model uses <think>…</think> and <answer>…</answer> segments to separate hidden chain‑of‑thought from the final answer. The tokenizer utilities ensure these tags are atomic tokens and provide helpers to find their spans and count think tokens. The service configuration defines stop sequences and soft‑cap budgets for these tags.

·      **LoRA side adapters.** A small LoRA adapter stack is attached to the decoder layers and is gated so that the adapters are only active during the think segment. This allows the base model to remain unmodified while giving an extra reasoning channel.

·      **Metacognitive heads.** A tiny head module reads pooled hidden states and predicts three signals: (i) a plan class (outline, verify, decompose, stop), (ii) a budget for how many tokens to spend thinking, and (iii) a confidence score. The heads can either pool from a few tapped layers or, in the linked‑all‑layers mode, project each layer’s last token through a shared MLP and aggregate these via attention or mean to produce a global embedding[\[1\]](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/metacog_heads.py#:~:text=self._tap_cache%5Blayer_idx%5D%20%3D%20h%5B%3A%2C%20,step%20feature). Optional per‑layer diagnostics (alpha weights, per‑layer plan logits) are returned for debugging.

·      **Budget‑aware decoding.** The serve engine wraps a HuggingFace CausalLM and uses the metacognitive heads to estimate a think‑token budget for each prompt. A SlackStop criterion applies a soft cap (budget ×(1+slack)) so that the model stops thinking if it doesn’t close the think tag by itself[\[2\]](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/serve.py#:~:text=def%20_count_think_tokens,clamped%20by%20actual%20generated%20length). The budget is calibrated via a JSON file that stores temperature and thresholds and is loaded at serve time.

·      **Serving and observation.** tina/serve.py defines IntrospectiveEngine, which attaches adapters, installs forward hooks on selected layers, collects hidden states, computes plan/budget/confidence, and orchestrates generation. Logs summarise plan distributions, budgets, confidence and think tokens used. tools/observe\_infer.py can run prompts and output these diagnostics.

·      **Training pipeline.** The repository includes scripts for supervised training (train/runner.py, train/one\_cycle.py) that combine cross‑entropy on answers with auxiliary losses on plan, budget, confidence, sparsity, over‑budget penalties and DPO/GRPO style objectives. Calibration (train/eval\_loop.py) fits a temperature and thresholds after evaluation. CI guardrails ensure reasoning tokens are atomic and stop sequences are valid. The training specification documents (tina‑project‑summary.txt, Training\_specification\_summary.md, tina\_training\_architect\_instruction.yaml) describe dataset formats and training phases (frozen base, LoRA fine‑tuning, on‑policy RL, etc.).

## 2 AZR (Absolute Zero Reasoner) foundation

Absolute Zero Reasoner (AZR) is a self‑play algorithm that trains a language model without external supervision by having it **propose** and **solve** programming tasks. The model alternates between two roles:

1.        **Proposer.** The model composes small code‑based tasks (a Python function and input–output pairs) conditioned on one of three modes: **deduction** (predict outputs given program and inputs), **abduction**(infer inputs given program and outputs) and **induction** (synthesise a program given examples). A _safe executor_ validates that generated programs are deterministic and safe to run (no imports, no I/O) by executing them in a sandboxed process with a timeout.

2.        **Solver.** Given a validated task, the model attempts to produce the correct answer under a solver prompt. Rewards combine a **learnability score** (Monte‑Carlo success rate of executing the program) and **solver correctness**, minus a small format penalty[\[3\]](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#:~:text=Qwen3). Intuitively, learnability rewards tasks that are neither trivial nor impossible, encouraging the proposer to create useful challenges.

The self‑play loop maintains buffers of past tasks for each mode. In each iteration, it samples examples from the buffer to condition the proposer, generates a new candidate task, validates the task, estimates learnability by running it multiple times, and then lets the solver answer. A simple policy‑gradient update (e.g. Task‑Relative REINFORCE++) increases the probability of generating tasks that lead to higher rewards. Since no external labels are required, this algorithm can continually self‑improve. The AZR foundation thus aligns well with the branch’s goal of training without curated datasets.

## 3 Current repository structure

The current GitHub repository is organised into several top‑level packages and utility scripts:

·      **tina/** **package** – core runtime modules for introspective serving:

·      metacog\_heads.py – defines MetacogConfig, projection trunks, per‑layer heads and aggregators. It implements both the three‑tap pooling and the linked‑all‑layers attention aggregator, returning plan logits, budgets and confidence[\[1\]](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/metacog_heads.py#:~:text=self._tap_cache%5Blayer_idx%5D%20%3D%20h%5B%3A%2C%20,step%20feature).

·      serve.py – wraps a HuggingFace model into IntrospectiveEngine. It attaches LoRA adapters, registers taps on decoder layers, loads calibration data, counts think tokens with soft caps, and orchestrates generation with hidden CoT[\[2\]](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/serve.py#:~:text=def%20_count_think_tokens,clamped%20by%20actual%20generated%20length). The engine config exposes options for visible CoT, maximum think tokens, taps, linked‑all‑layers mode, FiLM conditioning, calibration paths, etc.

·      tokenizer\_utils.py – ensures reasoning and decomposition tags (<think>, </think>, <answer>, </answer> and optional <plan>,<exec>, etc.) are single tokens and provides helpers to format chat prompts, count tokens and find tag spans.

·      side\_adapters.py – implements LoRA‑style residual adapters with gating for the think segment. An IntrospectionScaffold attaches these adapters to selected layers, stores gate activations and coverage for telemetry, and can be disabled by setting rank = 0.

*   tina\_chat\_introspective.py – a CLI script that loads the base model and LoRA adapter, instantiates the introspective engine, runs inference with a think budget and outputs hidden or visible chain‑of‑thought.
*   **train/** **package** – training and evaluation utilities:

·      runner.py – orchestrates supervised training phases (frozen base, LoRA fine‑tuning, on‑policy RL, unfreezing top layers). It loads data, computes losses (answer cross‑entropy, plan/budget/confidence auxiliaries, over‑budget penalties, quiet‑star, rewrite consistency), fits calibration after evaluation, and supports a hybrid self‑play schedule.

·      one\_cycle.py – a simplified one‑cycle smoke test.

·      losses.py – defines loss functions and regularizers (variance regularizer, masked symmetric KL for paraphrase consistency, over‑budget penalty, think CE, gate sparsity, quiet‑star consistency).

·      metrics.py – provides calibration helpers (fit temperature, fit thresholds) and quality‑vs‑budget metrics (exact match, token‑F1, leakage rate, ECE/Brier).

·      self\_play\_azr.py – implements a basic AZR loop: defines ProgramTask and TaskBuffer, generates tasks using the model, validates them with safe\_executor, uses introspective signals to modulate learnability reward, solves tasks and updates the policy via a surrogate REINFORCE objective.

*   safe\_executor.py – sandboxes Python code execution: validates ASTs to reject imports and attribute access, runs code in a subprocess with a timeout, and ensures deterministic outputs by running tasks twice. This is crucial for the self‑play loop.
*   **tools/** – miscellaneous utilities:

·      observe\_infer.py – runs prompts under the serve engine and logs structured diagnostics.

·      plot\_quality\_budget.py – plots quality vs budget curves from evaluation CSVs.

*   validate\_configs.py – validates service and generation configs against JSON schemas, ensuring required stop sequences, atomic tags and adapter settings.
*   **config/** – JSON configuration files (service\_config.json, generation\_config.json) and their schemas. The service config includes defaults for visible CoT, stop sequences (</answer>, </think>), calibration paths and soft‑cap slack ratios.
*   **Docs** – README.md provides an overview, quickstart, training and calibration instructions, CI guardrails, self‑play description and references. The training specifications (uploaded but not visible in the public repo) outline dataset format and phased training schedules.

Interdependencies:

·      serve.py relies on tokenizer\_utils to register reasoning tokens and side\_adapters/metacog\_heads to attach modules. It calls safe\_executor only in the self‑play context.

·      train/runner.py invokes losses.py, metrics.py and uses self\_play\_azr.py in hybrid mode. It also uses validate\_configs.py to check config correctness.

·      self\_play\_azr.py optionally instantiates an introspective engine from tina/serve.py for introspection‑guided task generation.

## 4 Branch purpose and AZR branch

The new branch aims to **investigate Absolute Zero Reasoner training on a powerful reasoning model**. The chosen base model is **Qwen3‑4B‑Thinking‑2507**, a 36‑layer, 4 billion‑parameter model with built‑in _thinking mode_ (the chat template automatically inserts <think> and often only emits the closing tag </think>[\[3\]](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#:~:text=Qwen3)). Its long context and improved reasoning benchmarks make it an ideal test bed for AZR self‑play.

The branch’s goals are:

·      **Replace dataset‑driven training with self‑play.** All supervised training loops and losses can be removed; training will instead consist solely of proposing tasks, validating them via the safe executor, solving them and updating the model via policy gradients. This eliminates the need for curated datasets or answer labels.

·      **Retain the introspective architecture.** The metacognitive heads and attention aggregator from Tina remain. They will now operate on Qwen3’s hidden states to predict plan, budget and confidence. Side adapters are disabled (rank = 0), because Qwen3 does not have a LoRA adapter. Taps may change (e.g. every 8th layer) but the mechanism is the same.

·      **Evaluate the impact of introspective signals on self‑play.** The AZR trainer will log plan distributions, budgets and confidence during task selection and policy updates. Introspective guidance can modulate the learnability reward to prioritise tasks where the model exhibits low confidence or predicts a “stop” plan. A separate evaluation script will compare solver accuracy and reward distributions with and without introspection.

## 5 Expected structure after changes

After applying the proposed modifications (see YAML plan), the repository will evolve as follows:

| Component | Before | After (AZR branch) | Interdependencies |
| --- | --- | --- | --- |
| LoRA side adapters (side_adapters.py) | Attached by default; rank configurable. Used for small models to add capacity. | Disabled by default (side_rank=0); IntrospectiveEngine skips adapter injection. side_adapters.py remains for compatibility. | The engine no longer depends on side_adapters when side_rank = 0. No gating or gate telemetry is recorded. |
| Tokenizer utilities | Register <think>, </think>, <answer>, </answer> and decomposition tags as atomic tokens. Assume both opening and closing tags appear in generation. | Updated to support Qwen3’s thinking‑only mode: only </think>may appear; missing opening tags are tolerated. A flag thinking_only_mode selects a reduced set of reasoning tokens. | serve.py calls ensure_reasoning_tokens()with the flag when it detects the model’s model_type is qwen3. The counting of think tokens and span finding handle missing <think> tags. |
| Serve engine (tina/serve.py) | Initializes LoRA adapters and metacog heads; uses taps or linked‑all‑layers aggregator; counts think tokens with soft cap; loads calibration; supports FiLM conditioning. | Adds Qwen3 detection: sets taps to (8,16,24,32), max think tokens = 4096, slack ratio = 0.5; registers only </think>stop tag; disables adapters; still attaches metacog heads. | It still depends on tokenizer_utils for tag registration and metacog_heads for heads; the safe executor is used only by self‑play. |
| Training (train/runner.py, train/one_cycle.py, train/losses.py, train/metrics.py) | Supports supervised phases with answer CE, auxiliary losses, RL penalties, hybrid schedule. | Supervised code is disabled; runner raises an error if mode != self_play. One‑cycle script is removed or deprecated. Loss functions become no‑ops; metrics remain for evaluation only. | train/self_play_azr.pybecomes the sole training loop; runner.pysimply instantiates SelfPlayAZRTrainer and calls train_loop. Calibration fitting from metrics.py may still be used to adjust plan/budget temperatures. |
| Self‑play trainer (train/self_play_azr.py) | Provides a basic AZR loop with introspection modulation; uses a safe executor; returns tasks with rewards and updates policy. | Tuned for Qwen3: fewer proposals per iteration, longer generation length, introspection signals logged in tasks. Rewards include a term proportional to introspection score. A new config file (config_qwen3_self_play.yaml) sets hyperparameters such as buffer size and introspection coefficient. | Depends on serve.py to instantiate an introspective engine for introspection guidance; uses safe_executor for validating programs; logs introspection metrics for evaluation. |
| Safe executor (train/safe_executor.py) | Validates ASTs (no imports, no attributes), runs code in subprocess with timeout, ensures determinism. | Unchanged; reused to check tasks in self‑play. | Used only by self_play_azr.py. The serve engine does not call it in normal inference. |
| Evaluation scripts | train/eval_loop.py fits calibration, sweeps budgets and exports CSV/plots; tools/observe_infer.py logs inference diagnostics; tools/plot_quality_budget.pyplots budgets. | A new evaluation script (tools/evaluate_self_play_introspection.py) runs self‑play on a buffer snapshot, logs solver accuracy, learnability and introspection distributions, and outputs CSV/PNG files. Existing evaluation scripts may still be used to study budgets on labelled datasets, but they are not part of this branch’s training loop. | Evaluation depends on self_play_azr.py for generating tasks, serve.py for introspection and tokenizer_utils.py for counting tokens. |
| Documentation | README describes Tina introspective reasoning, training phases, calibration and self‑play mode. Training specification documents outline dataset format and phase schedules. | README gains a new section “Qwen3‑4B‑Thinking‑2507 AZR Branch” explaining the branch’s goals, how to run self‑play training and evaluation, and how to disable introspection for ablations. Training specification is updated to remove dataset requirements and LoRA phases. | The docs now emphasise self‑play, introspection impact and Qwen3 features. Contributors can compare the new and old specifications to understand changes. |

## 6 Before‑and‑after interdependencies

### Before (Tina mainline)

·      The base model is a small Qwen2 variant. **LoRA side adapters** provide extra capacity; their gates are only active during the think segment. **Metacog heads** read tapped or aggregated hidden states to predict plan, budget and confidence. **Tokenizer utilities** register both <think> and </think>. **Serve**attaches adapters and heads, and orchestrates decoding with a soft cap. **Training** uses labelled data and multiple phases (frozen base, LoRA fine‑tuning, RL). **Self‑play** is optional (hybrid mode); it uses introspection for guidance but is not the main training method.

### After (AZR branch on Qwen3‑4B)

·      The base model is Qwen3‑4B‑Thinking‑2507, which already has strong reasoning ability[\[3\]](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#:~:text=Qwen3). **LoRA side adapters are disabled** (rank = 0) because the model does not need extra capacity. **Metacog heads**remain and operate on hidden states of the 36‑layer model; the default taps are adjusted (e.g. every 8th layer) and think budgets are increased (default 4k tokens). The head aggregator still supports attention or mean pooling. **Tokenizer utilities** are modified to recognise only closing think tags, accommodating Qwen3’s template that sometimes omits <think>[\[4\]](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#:~:text=Additionally%2C%20to%20enforce%20model%20thinking%2C,think%3E%60%20tag). **Serve** now detects Qwen3, sets appropriate defaults and does not attach adapters; it still enforces soft‑cap budgets and logs introspection telemetry.

·      **Training** is entirely replaced by **AZR self‑play**. The supervised loops and data requirements vanish. train/self\_play\_azr.py becomes the core trainer, and train/runner.py simply instantiates it based on the config. Rewards combine learnability, solver correctness, format penalty and introspection signals. A safe executor ensures tasks are valid. Calibration fitting remains available but is optional.

·      **Evaluation** focuses on self‑play metrics (solver accuracy, average reward, plan distributions) and introspection impact. A new script (tools/evaluate\_self\_play\_introspection.py) runs evaluation and outputs CSV/plots for analysis. Existing budget‑sweep tools may still be useful for labelled benchmarks but are outside this branch.

### New interdependencies

·      self\_play\_azr.py depends more strongly on serve.py: it creates an IntrospectiveEngine to obtain plan, budget and confidence predictions for each candidate task, modulating the learnability reward. It logs these signals per task for evaluation.

·      serve.py depends on tokenizer\_utils.py with thinking\_only\_mode for registering </think>; it no longer uses side\_adapters when side\_rank=0 but still uses metacog\_heads for introspection.

·      Evaluation scripts depend on self\_play\_azr.py to generate tasks and on serve.py to compute introspective signals; they also read the introspection logs stored in tasks.

## 7 Intended use and next steps for the coder

For this branch, the coder should:

1.        **Setup the environment.** Install transformers>=4.51.0 and ensure GPU memory is sufficient for Qwen3‑4B‑Thinking‑2507. Verify that the base model and tokenizer load correctly and register reasoning tags.

2.        **Load the introspective engine.** Instantiate IntrospectiveEngine with Qwen3, passing side\_rank=0 and enabling linked\_all\_layers or the default taps. Optionally supply a calibration JSON if available. Confirm that plan, budget and confidence are returned and that think tokens are counted correctly despite missing <think> tags.

3.        **Run self‑play training.** Use the new config\_qwen3\_self\_play.yaml to configure buffer sizes, propose counts, Monte‑Carlo rollouts, and the introspection coefficient. Call python train/runner.py --train-config config\_qwen3\_self\_play.yaml --steps N to perform N self‑play iterations. Monitor logs for average rewards, introspection scores and solver accuracy.

4.        **Evaluate introspective impact.** After training, run tools/evaluate\_self\_play\_introspection.py on a snapshot of the buffers. Examine CSV/plots to see how plan distributions, budgets and confidence correlate with task difficulty and solver success. Compare against a baseline run with introspection\_coeff=0 to assess the benefit of introspection.

5.        **Iterate and refine.** Adjust hyperparameters (introspection coefficient, buffer size, number of proposal samples) to study their impact. Consider adding new task types or reward terms to challenge the model further. Document findings and update the README and training specification accordingly.

By following this plan, the coder can systematically explore how adding Tina’s introspective functionality to a strong reasoning model like Qwen3 influences AZR self‑play, both in terms of play decisions (which tasks are proposed and solved) and overall performance.

* * *

[\[1\]](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/metacog_heads.py#:~:text=self._tap_cache%5Blayer_idx%5D%20%3D%20h%5B%3A%2C%20,step%20feature) raw.githubusercontent.com

[https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/metacog\_heads.py](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/metacog_heads.py)

[\[2\]](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/serve.py#:~:text=def%20_count_think_tokens,clamped%20by%20actual%20generated%20length) raw.githubusercontent.com

[https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/serve.py](https://raw.githubusercontent.com/justdtip/Small-Metacognitive-LLM/main/tina/serve.py)

[\[3\]](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#:~:text=Qwen3) [\[4\]](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#:~:text=Additionally%2C%20to%20enforce%20model%20thinking%2C,think%3E%60%20tag) Qwen/Qwen3-4B-Thinking-2507 · Hugging Face

[https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)