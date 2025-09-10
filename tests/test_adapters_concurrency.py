import threading
import pytest
from side_adapters import IntrospectionScaffold, ResidualAdapterConfig
torch = pytest.importorskip("torch")

def test_isolated_think_modes():
    m = torch.nn.Identity()
    sc = IntrospectionScaffold(m, hidden_size=16, num_layers=2, cfg=ResidualAdapterConfig(hidden_size=16))
    out = []
    def worker(flag):
        sc.set_think_mode(flag)
        out.append(flag)
    th1 = threading.Thread(target=worker, args=(True,))
    th2 = threading.Thread(target=worker, args=(False,))
    th1.start(); th2.start(); th1.join(); th2.join()
    assert out.count(True)==1 and out.count(False)==1
