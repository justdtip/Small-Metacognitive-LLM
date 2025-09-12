Dataset JSONL Fields (Reasoning Style Taxonomy)
==============================================

Each training example should include the following fields (minimum):

- text: string with hidden-CoT format "<think> ... </think> <answer> ... </answer>" (required)
- plan_class: optional int for plan supervision
- target_budget: optional int for budget supervision
- correct/correctness: optional 0/1 flag or score for correctness
- think_tokens_used: optional int, preferred when on-policy decode measured think length

Optional decomposition tags inside <think>
-----------------------------------------

You may include latent problem decomposition inside the <think> segment using sub‑tags:

- <plan> ... </plan>
- <exec> ... </exec>
- <eval> ... </eval>

These tags are optional and unconstrained; the collator will derive soft boolean masks (plan_mask, exec_mask, eval_mask)
when present and leave them all‑zero when absent or malformed. Training losses and stop rules remain based on </think>
and </answer> only; the new tags do not affect stopping and are not required by the schema.

Example:

```
<think>
  <plan> outline the approach </plan>
  <exec> compute intermediate steps </exec>
  <eval> check the result </eval>
</think>
<answer> final result </answer>
```

Reasoning Style (optional diagnostics/aux):

- style_tag: string in {checklist, explainer, simulate, prove, counterfactual, program, visualize}
- style_id: int identifier mapping style_tag → {0..6}; if absent, derived from style_tag; default -1 if unknown

Collate Output (train.data.make_collate_fn):

- style_id: LongTensor [B], values in {0..6} or -1
- style_tag: list[str] of length B (empty string for unknown)
- plan_mask / exec_mask / eval_mask: LongTensor [B,T] (0/1) optional, only attached when at least one sub‑segment appears in the batch.

These fields do not affect masks/labels by default; they enable style-balanced sampling, style-controlled decoding,
and slice-wise evaluation.
