Dataset JSONL Fields (Reasoning Style Taxonomy)
==============================================

Each training example should include the following fields (minimum):

- text: string with hidden-CoT format "<think> ... </think> <answer> ... </answer>" (required)
- plan_class: optional int for plan supervision
- target_budget: optional int for budget supervision
- correct/correctness: optional 0/1 flag or score for correctness
- think_tokens_used: optional int, preferred when on-policy decode measured think length

Reasoning Style (optional diagnostics/aux):

- style_tag: string in {checklist, explainer, simulate, prove, counterfactual, program, visualize}
- style_id: int identifier mapping style_tag → {0..6}; if absent, derived from style_tag; default -1 if unknown

Collate Output (train.data.make_collate_fn):

- style_id: LongTensor [B], values in {0..6} or -1
- style_tag: list[str] of length B (empty string for unknown)

These fields do not affect masks/labels by default; they enable style-balanced sampling, style-controlled decoding,
and slice-wise evaluation.
