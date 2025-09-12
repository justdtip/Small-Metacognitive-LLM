import pytest

torch = pytest.importorskip("torch")

from tina.serve import _count_think_tokens, SlackStop


def make_seq(base_len, new_tokens):
    # simple sequence of ints 0..N-1; think starts at base_len
    return list(range(base_len + new_tokens))


def test_stops_at_first_close():
    base = 10
    # generate 50 new tokens beyond base
    g = make_seq(base, 50)
    # closing pattern occurs at base+7
    close_pos = base + 7
    close_id = [g[close_pos]]
    used = _count_think_tokens(g, [close_id], base, cap=100, slack_ratio=0.2)
    assert used == 7


def test_respects_soft_cap_without_close():
    base = 5
    cap = 20
    slack = 0.25
    limit = int(cap * (1 + slack))
    # generate more tokens than soft cap allows
    g = make_seq(base, limit + 30)
    used = _count_think_tokens(g, [], base, cap=cap, slack_ratio=slack)
    assert used == limit


def test_tie_breaker_close_vs_softcap():
    base = 3
    cap = 10
    slack = 0.5
    limit = int(cap * (1 + slack))  # 15
    g = make_seq(base, limit + 5)
    # place close at base+12 (< base+limit)
    close_pos = base + 12
    close_id = [g[close_pos]]
    used = _count_think_tokens(g, [close_id], base, cap=cap, slack_ratio=slack)
    assert used == 12
    # if close after limit, we respect soft-cap
    close_pos2 = base + limit + 2
    close_id2 = [g[close_pos2]] if close_pos2 < len(g) else [999999]
    used2 = _count_think_tokens(g, [close_id2], base, cap=cap, slack_ratio=slack)
    assert used2 == limit


def test_off_by_one_at_boundary():
    base = 8
    cap = 16
    slack = 0.25
    limit = int(cap * (1 + slack))
    # Exactly at boundary
    g = make_seq(base, limit)
    used = _count_think_tokens(g, None, base, cap=cap, slack_ratio=slack)
    assert used == limit
    # SlackStop should allow exactly limit tokens and stop when exceeding
    stop = SlackStop(base_len=base, budget=cap, slack_ratio=slack)
    # build ids with exactly limit new tokens
    ids_at = torch.tensor([[0] * (base + limit)], dtype=torch.long)
    assert stop(ids_at, None) is False
    # one more should stop
    ids_over = torch.tensor([[0] * (base + limit + 1)], dtype=torch.long)
    assert stop(ids_over, None) is True

