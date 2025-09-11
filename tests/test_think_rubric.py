def test_think_rubric_score_no_span():
    from train.metrics import think_rubric_score
    assert think_rubric_score("no think here", lambda s: 1.0) is None

