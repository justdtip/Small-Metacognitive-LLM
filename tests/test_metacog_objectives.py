import pytest
torch = pytest.importorskip("torch")

from train.losses import compute_losses


def test_compute_losses_includes_metacog_terms():
    B, T, V = 2, 5, 7
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    # labels: set two valid positions and the rest ignored
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[0, 1] = 3
    labels[1, 2] = 1

    plan_logits = torch.randn(B, 4)
    plan_targets = torch.randint(0, 4, (B,))
    budget_pred = torch.rand(B, 1) * 128.0
    budget_target = torch.full((B, 1), 64.0)
    conf_logits = torch.randn(B, 1)
    conf_labels = torch.randint(0, 2, (B, 1))

    out = compute_losses(
        logits, labels,
        gate_modules=None,
        weights={"answer_ce": 1.0, "plan_ce": 1.0, "budget_reg": 1.0, "conf_cal": 1.0, "aux_mix": 0.0, "gate_reg": 0.0},
        plan_logits=plan_logits, plan_targets=plan_targets,
        budget_pred=budget_pred, budget_target=budget_target,
        conf_logits=conf_logits, conf_labels=conf_labels,
    )

    # Keys present
    for k in ("answer_ce", "plan_ce", "budget_reg", "conf_cal", "total"):
        assert k in out

    # Total should be the sum of the individual terms when other weights are zero
    expected = out["answer_ce"] + out["plan_ce"] + out["budget_reg"] + out["conf_cal"]
    assert torch.isclose(out["total"], expected, rtol=1e-5, atol=1e-6)

