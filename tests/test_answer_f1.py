def test_f1_basic():
    from train.metrics import f1_token
    # pred: a b c; gold: a c
    # overlap=2; precision=2/3; recall=2/2=1 -> F1=2*prec*rec/(prec+rec)=4/5
    assert abs(f1_token("a b c", "a c") - (2*2/(3+2))) < 1e-9

