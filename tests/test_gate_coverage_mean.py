import pytest
import torch

from train.runner import adapter_gate_step_smoke
from side_adapters import LowRankAdapter, ResidualAdapterConfig
from train.hooks import think_mask_context


def test_coverage_nonzero_on_think_and_drops_on_no_think():
    # With think activity
    out = adapter_gate_step_smoke()
    cov_mean = out.get("gate_coverage_mean")
    assert cov_mean is not None and cov_mean > 0.0

    # Construct a manual adapter pass with zero think mask to compare
    B, T, H = 1, 6, 16
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()
    h = torch.randn(B, T, H)
    zeros = torch.zeros(B, T)
    with think_mask_context(zeros):
        _ = adap(h)
    cov0 = getattr(adap, "_last_gate_coverage", None)
    if torch.is_tensor(cov0):
        cov0 = float(cov0.item())
    assert cov0 is not None
    assert cov0 <= cov_mean
    assert cov0 <= 0.01  # near-zero without think activity

