from train.eval_loop import compute_eval_metrics, quality_vs_budget_curve


def test_compute_eval_metrics_smoke():
    recs = [
        {
            "body": "<think> step a </think> <answer>ok</answer>",
            "gold": "ok",
            "plan_true": 1,
            "plan_pred": 1,
            "budget_pred": 64,
            "budget_target": 64,
            "conf_prob": 0.8,
        },
        {
            "body": "<think> step b step c </think> <answer>no</answer>",
            "gold": "yes",
            "plan_true": 2,
            "plan_pred": 0,
            "budget_pred": 128,
            "budget_target": 64,
            "conf_prob": 0.2,
        },
    ]
    m = compute_eval_metrics(recs)
    # Keys exist
    for k in ("accuracy", "think_tokens_mean", "plan_accuracy", "budget_mae", "budget_me", "leakage_rate"):
        assert k in m


def test_quality_vs_budget_curve_monotone_synthetic():
    recs = [
        {"budget_target": 2},
        {"budget_target": 4},
        {"budget_target": 6},
    ]

    def quality_fn(r, B: int) -> float:
        # synthetic: correct if budget >= target
        return 1.0 if B >= int(r["budget_target"]) else 0.0

    out = quality_vs_budget_curve(recs, budgets=[1, 3, 5, 7], quality_fn=quality_fn)
    assert "curve" in out and len(out["curve"]) == 4
    accs = [c["accuracy"] for c in out["curve"]]
    # Monotone non-decreasing
    assert all(accs[i] <= accs[i + 1] + 1e-9 for i in range(len(accs) - 1))
