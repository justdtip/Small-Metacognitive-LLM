import pytest
torch = pytest.importorskip("torch")

from train.losses import rewrite_consistency_kl, compute_losses


def _make_logits(T: int, V: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, V, generator=g)


def test_symmetric_kl_on_answers_matches_zero_when_equal():
    T, V = 12, 7
    # Answer spans occupy positions 4..9 (6 tokens)
    mask = torch.zeros(T, dtype=torch.long)
    mask[4:10] = 1
    # Construct two identical distributions on answer positions
    logits_a = _make_logits(T, V, seed=13)
    logits_b = logits_a.clone()
    kl = rewrite_consistency_kl(logits_a, logits_b, mask, mask, tau=1.0)
    assert torch.is_tensor(kl)
    assert float(kl.item()) <= 1e-6

    # Also exercise compute_losses with a group
    B = 2
    logits = torch.stack([logits_a, logits_b], dim=0)  # (2,T,V)
    labels = torch.full((B, T), -100, dtype=torch.long)
    am = torch.stack([mask, mask], dim=0)
    out = compute_losses(
        logits, labels,
        weights={"answer_ce": 0.0, "rewrite_consistency": 1.0},
        answer_mask=am,
        rewrite_groups=[[0, 1]],
    )
    assert "rewrite_kl" in out
    assert float(out["rewrite_kl"].item()) <= 1e-6


def test_symmetric_kl_on_answers_positive_when_different():
    T, V = 10, 5
    mask_a = torch.zeros(T, dtype=torch.long)
    mask_b = torch.zeros(T, dtype=torch.long)
    mask_a[2:8] = 1  # 6 tokens
    mask_b[2:8] = 1
    logits_a = _make_logits(T, V, seed=1)
    logits_b = _make_logits(T, V, seed=2)  # different distributions
    kl = rewrite_consistency_kl(logits_a, logits_b, mask_a, mask_b, tau=1.0)
    assert float(kl.item()) > 0.0


def test_variable_length_handling_truncates_safely():
    # One answer length is 8 tokens, the other is 10; ensure no errors and finite KL
    T, V = 16, 11
    mask_a = torch.zeros(T, dtype=torch.long)
    mask_b = torch.zeros(T, dtype=torch.long)
    mask_a[3:11] = 1   # 8 tokens
    mask_b[3:13] = 1   # 10 tokens
    logits_a = _make_logits(T, V, seed=10)
    logits_b = _make_logits(T, V, seed=20)
    kl = rewrite_consistency_kl(logits_a, logits_b, mask_a, mask_b, tau=0.7)
    assert torch.isfinite(kl)
    assert float(kl.item()) >= 0.0

