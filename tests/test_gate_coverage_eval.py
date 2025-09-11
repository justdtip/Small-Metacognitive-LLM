import pytest

from train.runner import adapter_gate_step_smoke
from side_adapters import LowRankAdapter, ResidualAdapterConfig
from train.hooks import think_mask_context
import torch


def test_gate_coverage_fraction():
    out = adapter_gate_step_smoke()
    cov = out.get("coverage")
    assert cov is not None
    assert 0.0 <= cov <= 1.0
    # In this synthetic setup, coverage should be less than 1.0 since some non-think positions exist
    assert cov < 1.0


def test_gate_coverage_value():
    out = adapter_gate_step_smoke()
    cov = out.get("coverage")
    assert cov is not None
    assert 0.0 < cov <= 1.0


def test_gate_coverage_all_ones_mask():
    B, T, H = 1, 8, 16
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()  # deterministic gating
    h = torch.randn(B, T, H)
    ones = torch.ones(B, T)
    with think_mask_context(ones):
        _ = adap(h)
    cov = getattr(adap, "_last_gate_coverage", None)
    if torch.is_tensor(cov):
        cov = float(cov.item())
    assert cov is not None
    assert 0.0 <= cov <= 1.0
    assert cov >= 0.99  # near-full coverage under all-ones mask


def test_gate_coverage_all_zeros_mask():
    B, T, H = 1, 8, 16
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()
    h = torch.randn(B, T, H)
    zeros = torch.zeros(B, T)
    with think_mask_context(zeros):
        _ = adap(h)
    cov = getattr(adap, "_last_gate_coverage", None)
    if torch.is_tensor(cov):
        cov = float(cov.item())
    assert cov is not None
    assert 0.0 <= cov <= 1.0
    assert cov <= 0.01  # near-zero coverage under all-zeros mask


def test_gate_coverage_monotonic_inclusion():
    B, T, H = 1, 8, 16
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()
    h = torch.randn(B, T, H)
    m_small = torch.zeros(B, T); m_small[0, :2] = 1.0
    m_large = torch.zeros(B, T); m_large[0, :5] = 1.0
    with think_mask_context(m_small):
        _ = adap(h)
    cov_small = getattr(adap, "_last_gate_coverage", None)
    if torch.is_tensor(cov_small):
        cov_small = float(cov_small.item())
    with think_mask_context(m_large):
        _ = adap(h)
    cov_large = getattr(adap, "_last_gate_coverage", None)
    if torch.is_tensor(cov_large):
        cov_large = float(cov_large.item())
    assert cov_small is not None and cov_large is not None
    assert 0.0 <= cov_small <= cov_large <= 1.0
