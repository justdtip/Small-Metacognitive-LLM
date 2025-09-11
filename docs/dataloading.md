Data Loading and Label Heuristics
=================================

This document describes how plan/budget/correctness labels and their provenance are attached to each training sample to guarantee dense supervision of metacognitive heads.

Batch Schema
------------

The collate function `train.data.make_collate_fn()` returns a batch dict with at least:

- `input_ids`, `attention_mask`, `loss_mask`, `think_mask`, `answer_mask`
- `plan_targets`: Long tensor [B]
- `target_budget`: Long tensor [B]
- `correctness`: Long tensor [B] (−1 if unknown)
- `plan_src`: list[str] of length B, each in {`gold`,`teacher`,`heuristic`}
- `budget_src`: list[str] of length B, each in {`gold`,`teacher`,`heuristic`}
- `think_tokens_used`: list[int] of length B (tokens strictly inside `<think>…</think>`)

Provenance and Heuristics
-------------------------

- If an example provides `plan_class` and/or `target_budget` fields, they are used as-is and the corresponding provenance defaults to `gold` unless `plan_src`/`budget_src` are explicitly provided as `teacher`.
- When labels are absent, the collate function backfills:
  - `plan_class` via a length-based bucket over think tokens using thresholds `(lo, hi) = (32, 128)` to produce classes {0: short, 1: medium, 2: long}.
  - `target_budget` via `round(max(1, α · think_tokens_used))` with α = 1.0 by default.
  - Both backfilled labels are marked with provenance `heuristic`.

Rationale
---------

Providing dense supervision enables the plan/budget/confidence heads to learn meaningful controls even when gold labels are sparse. Provenance flags allow slice-based evaluation (gold vs teacher vs heuristic). Think-token counts come from `segment_and_masks()` to maintain a single source of truth for segmentation.

