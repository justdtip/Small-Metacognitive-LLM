def test_think_rubric_score_no_span():
    from train.metrics import think_rubric_score
    assert think_rubric_score("no think here", lambda s: 1.0) is None


def test_think_rubric_extraction():
    from train.metrics import think_rubric_score
    # Exact extraction and clamping within [0,1]
    assert think_rubric_score("<think>foo</think>", lambda s: 0.5) == 0.5
    assert think_rubric_score("<think>bar</think>", lambda s: 2.0) == 1.0
    assert think_rubric_score("<think>baz</think>", lambda s: -1.0) == 0.0
    # Absent tags returns None
    assert think_rubric_score("no tags", lambda s: 1.0) is None
