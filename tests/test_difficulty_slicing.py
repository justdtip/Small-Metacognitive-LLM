def test_difficulty_bin_slice():
    recs = [
        {"correct": 1, "difficulty_bin": 0, "plan_src": "gold"},
        {"correct": 0, "difficulty_bin": 1, "plan_src": "gold"},
    ]
    from train.metrics import aggregate
    from train.eval_loop import compute_eval_metrics
    m = aggregate(recs, compute_eval_metrics)
    assert "slice:difficulty_bin=0" in m
    assert "slice:difficulty_bin=1" in m
