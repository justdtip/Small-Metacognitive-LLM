# Repository Overview and Critical Fixes

This document summarizes the current state of the Small‑Metacognitive‑LLM repository and lists the critical fixes needed to bring the codebase into alignment with the project requirements and the failing test suite. It is intended for a new coder who will be tasked with implementing the changes and verifying them through rigorous testing.

## Current State of the Repository

### Overall structure

The repository implements a **small decoder‑only language model** with an additional **metacognitive layer**. Key components include:

·      **Tokenizer utilities** (tina/tokenizer\_utils.py): functions to register reasoning and decomposition tags as atomic tokens (ensure\_reasoning\_tokens), to convert text into think/answer segments with corresponding masks (segment\_and\_masks), and to find spans. These utilities enforce left‑padding and ensure that special tags are encoded as single tokens.

·      **Metacognitive heads** (tina/metacog\_heads.py): a module that pools hidden states from selected decoder layers, optionally aggregates them (mean or attention), and predicts plan logits, a think‑token budget, a confidence score and decoding policy parameters. It supports multi‑expert mixtures and can return per‑layer diagnostics when configured.

·      **Side adapters and gating** (tina/side\_adapters.py): implements low‑rank residual adapters with a hard‑concrete gate per token. The gate is activated only during THINK segments via a context variable and records gate activity and coverage metrics.

·      **Runtime inference engine** (tina/serve.py): wraps a base model with the introspective scaffold, loads calibration data if available, controls decoding via **hard** and **soft** think budgets (StopOnTags and SlackStop), and maps metacognitive predictions into decode parameters.

·      **Training pipeline** (train/runner.py and associated modules): builds the base model and tokenizer, attaches the introspection scaffold, registers hooks for side adapters, computes multiple loss terms (answer cross‑entropy, plan cross‑entropy, budget regression, confidence calibration and optional regularizers), and implements periodic light evaluation. It also includes helper smoke tests.

·      **Data loading** (train/data.py): provides dataset wrappers and collators that enforce exactly one <think>…</think><answer>…</answer> pair per record, derive masks and optional decomposition sub‑masks, and stream external datasets when required.

·      **Evaluation and calibration** (train/eval\_loop.py): functions to fit a confidence temperature, derive plan thresholds and budget clips, and sweep quality versus budget.

·      **Observation tool** (tools/observe\_infer.py): allows logging of hidden state norms, attention entropies, gate statistics and metacognitive outputs during inference.

### Current issues highlighted by the test suite and specification

1.        **Checkpoint naming and saving** – the training loop writes checkpoint files to artifacts/checkpoints/step‑{step:06d}.pt, but the tests expect checkpoint files named checkpoint‑{step} within a configurable save\_dir.

2.        **Metacog forward signature** – the engine always calls MetacogHeads.forward(B\_max=…, return\_state=True). When tests monkey‑patch forward with a function that accepts only B\_max, this raises a TypeError.

3.        **Calibration application** – the engine loads calibration blobs (confidence temperature, plan thresholds, budget clip) but does not apply them consistently. Tests indicate that plan thresholds and budget clips are ignored during budget estimation.

4.        **Gate coverage and per‑layer telemetry** – although the adapter modules record \_last\_gate\_activity and \_last\_gate\_coverage, these metrics are not surfaced in training logs or evaluation, leading to test failures for gate‑coverage checks. Similarly, per‑layer diagnostics are not returned during decoding, causing undefined variables.

5.        **Stop‑tag atomicity** – custom stop tags must be registered as additional special tokens to guarantee single‑token encoding. Some tests show that the default stop sequences are splitting into multiple tokens, violating parity between training and serving.

6.        **On‑policy decoding logs and budget monotonicity** – the training runner does not log on‑policy decoding metrics (e.g. think\_tokens\_used, policy vectors) and does not ensure that predicted budgets never increase across steps, causing failures in budget monotonicity tests.

7.        **Config validation and model selection** – the runner silently falls back to a tiny dummy model when the specified Hugging Face model path does not exist. Tests require explicit validation of the model path and clear errors when model.base or adapter directories are missing.

8.        **Absence of the training HUD/TUI** – the specification calls for a curses‑based UI to monitor training metrics and adjust parameters. No such interface exists in the repository.

## Critical Fixes to Implement

Below is a high‑level list of fixes that should be addressed. Each item will be expanded into detailed implementation instructions in subsequent YAML blocks.

1.        **Checkpoint Saving Alignment** – modify the training runner to use cfg\['save\_interval'\] and cfg\['save\_dir'\]for periodic checkpointing and to write files named checkpoint‑{step}. Ensure the checkpoint directory is created if missing.

2.        **Flexible Metacog Forward** – update the invocation of MetacogHeads.forward in the engine to attempt forward(B\_max) first and only pass return\_state=True if the signature supports it. This prevents crashes when tests monkey‑patch the method.

3.        **Calibration Enforcement** – load calibration data (confidence temperature, plan thresholds, budget clip) in the engine and apply it: temperature‑scale the confidence logit, threshold plan logits to choose a plan and apply a minimum budget equal to the clip. Extend \_estimate\_budget to respect these parameters.

4.        **Gate Activity and Per‑Layer Telemetry** – expose adapter gate coverage and activity metrics in training logs and evaluation results. Include a gate\_coverage field in the returned metacog outputs and ensure per‑layer diagnostics are returned under a per\_layer key. Update the training loop to log these values.

5.        **Stop‑Tag Registration** – ensure that both training and serving register all stop sequences (e.g. </think>, </answer> and any custom sequences) as additional\_special\_tokens and that unique\_no\_split\_tokens is updated. Validate their atomicity after registration.

6.        **On‑Policy Decode Logging & Budget Monotonicity** – after each on‑policy decode during training, log metrics such as predicted budget, actual think tokens used, policy vector and gate coverage. Implement a check that the predicted budget never increases from one call to the next when return\_stateindicates the same input; clamp if necessary.

7.        **Config and Model Path Validation** – before training starts, validate that the configured model.basedirectory exists and that any adapter directories exist. Raise a clear error if they are missing rather than silently falling back to the dummy model.

8.        **TUI/Training HUD (optional but recommended)** – implement a simple curses‑based HUD that displays running loss components, gate coverage, budget statistics, plan distribution and allows pausing/adjusting learning‑rate or budget caps. If time does not permit full implementation, stub out a minimal interface that prints these metrics to the console at regular intervals.

These fixes, when implemented and properly tested, should substantially reduce the number of failing tests and move the project closer to the goals specified in the requirements.

* * *