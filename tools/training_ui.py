from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return None


def _to_list(x: Any) -> Optional[List[float]]:
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return [float(v) for v in x.detach().flatten().tolist()]
    except Exception:
        pass
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _softmax(vals: List[float]) -> List[float]:
    if not vals:
        return []
    import math
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _bar(value: float, maximum: float, width: int = 20, color: str = "cyan") -> Text:
    if maximum is None or maximum <= 0:
        maximum = 1.0
    v = max(0.0, min(float(value), float(maximum)))
    ratio = v / float(maximum)
    filled = max(0, min(width, int(round(ratio * width))))
    empty = max(0, width - filled)
    txt = Text()
    if filled:
        txt.append("█" * filled, style=color)
    if empty:
        txt.append("░" * empty, style="dim")
    txt.append(f" {v:.2f}", style="bold")
    return txt


class TrainingDashboard:
    """
    Rich-based dashboard to visualize training metrics for the introspective model.

    Regions:
    - Top bar: clock and step
    - Left: summary metrics (losses, rewards, tokens, budget/confidence)
    - Center: aggregated outputs (budget, confidence, plan distribution)
    - Right: per-layer alpha weights and per-layer plan logits
    - Bottom: rolling log
    """

    def __init__(self, num_layers: int, width: int = 120, height: int = 40):
        self.num_layers = int(max(1, num_layers))
        self.width = int(max(60, width))
        self.height = int(max(24, height))

        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="top", size=3),
            Layout(name="body"),
            Layout(name="bottom", size=8),
        )
        self.layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="center", ratio=3),
            Layout(name="right", ratio=3),
        )

        # Internal state
        self._step: int = 0
        self._log: List[Text] = []
        self.live: Optional[Live] = None

        # initial renderables
        self._render_top()
        self._render_left(Table.grid())
        self._render_center(Table.grid())
        self._render_right(Table.grid())
        self._render_bottom()

    # ---- Rendering helpers ---------------------------------------------------
    def _render_top(self, clock: Optional[str] = None):
        now = clock or time.strftime("%H:%M:%S")
        title = Text(f" eDex Training Dashboard ", style="bold cyan")
        info = Text(f" {now} | step {self._step} ", style="bold dim")
        self.layout["top"].update(Panel.fit(Text.assemble(title, info), style="on black", border_style="cyan"))

    def _render_left(self, table: Table):
        self.layout["left"].update(Panel(table, title="Summary", title_align="left", style="on black", border_style="cyan"))

    def _render_center(self, table: Table):
        self.layout["center"].update(Panel(table, title="Aggregates", title_align="left", style="on black", border_style="cyan"))

    def _render_right(self, renderable):
        self.layout["right"].update(Panel(renderable, title="Per-Layer", title_align="left", style="on black", border_style="cyan"))

    def _render_bottom(self):
        # Keep last ~200 lines
        if len(self._log) > 200:
            self._log = self._log[-200:]
        log_text = Text()
        for line in self._log[-10:]:
            log_text.append(line)
            log_text.append("\n")
        self.layout["bottom"].update(Panel(log_text, title="Log", title_align="left", style="on black", border_style="cyan"))

    # ---- Public API -----------------------------------------------------------
    def update(self, metrics: Dict[str, Any]):
        # Step
        self._step = int(metrics.get("step") or metrics.get("t") or 0)
        self._render_top()

        # Left summary
        left = Table(show_header=False, box=None, padding=(0, 1))
        left.add_row(Text("loss_total:", style="bold cyan"), Text(f"{_as_float(metrics.get('loss_total'))}") )
        la = metrics.get("loss_answer")
        lr = metrics.get("loss_rl")
        left.add_row(Text("loss_answer:", style="cyan"), Text(f"{_as_float(la)}"))
        left.add_row(Text("loss_rl:", style="cyan"), Text(f"{_as_float(lr)}"))
        left.add_row(Text("reward_mean:", style="cyan"), Text(f"{_as_float(metrics.get('reward_mean'))}"))
        left.add_row(Text("think_tokens:", style="cyan"), Text(f"{metrics.get('think_tokens')}"))
        left.add_row(Text("budget_pred:", style="cyan"), Text(f"{_as_float(metrics.get('budget_pred'))}"))
        left.add_row(Text("budget_target:", style="cyan"), Text(f"{_as_float(metrics.get('budget_target'))}"))
        left.add_row(Text("plan_pred:", style="cyan"), Text(f"{metrics.get('plan_pred')}"))
        left.add_row(Text("confidence:", style="cyan"), Text(f"{_as_float(metrics.get('confidence'))}"))
        self._render_left(left)

        # Center aggregates: budget & confidence as bars; plan distribution
        center = Table.grid(padding=(0, 1))
        center.add_row(Text("Budget", style="bold cyan"))
        bp = _as_float(metrics.get("budget_pred")) or 0.0
        bt = _as_float(metrics.get("budget_target")) or 0.0
        max_b = max(bp, bt, 1.0)
        center.add_row(_bar(bp, max_b, width=30, color="bright_cyan"))
        center.add_row(Text("Confidence", style="bold cyan"))
        conf = _as_float(metrics.get("confidence")) or 0.0
        center.add_row(_bar(conf, 1.0, width=30, color="bright_cyan"))

        # Plan distribution (global)
        plog = metrics.get("plan_logits")
        plog_list = _to_list(plog) or []
        # If per-layer plan present, reduce by mean over layers
        if not plog_list:
            pl_pl = metrics.get("per_layer_plan")
            try:
                import torch  # type: ignore
                if isinstance(pl_pl, torch.Tensor) and pl_pl.dim() == 3:
                    # [B,L,K] → [K]
                    plog_list = [float(v) for v in pl_pl.mean(dim=(0, 1)).flatten().tolist()]
            except Exception:
                pass
        if plog_list:
            probs = _softmax(plog_list[:8])
            pd_tbl = Table(show_header=True, header_style="bold cyan")
            pd_tbl.add_column("Plan")
            pd_tbl.add_column("p")
            for i, p in enumerate(probs):
                pd_tbl.add_row(f"{i}", _bar(p, 1.0, width=12, color="bright_cyan"))
            center.add_row(Text("Plan Distribution", style="bold cyan"))
            center.add_row(pd_tbl)
        self._render_center(center)

        # Right: per-layer alpha and per-layer plan logits
        right = Table.grid(padding=(0, 1))
        # Alpha weights
        alpha_any = metrics.get("alpha")
        alpha: Optional[List[float]] = None
        try:
            import torch  # type: ignore
            if isinstance(alpha_any, torch.Tensor):  # [B,L]
                a = alpha_any.detach().float()
                if a.dim() == 2:
                    alpha = [float(v) for v in a.mean(dim=0).tolist()]
        except Exception:
            pass
        if alpha is None:
            alpha = _to_list(alpha_any)
        if alpha:
            right.add_row(Text("Alpha", style="bold cyan"))
            a_tbl = Table(show_header=False, box=None)
            for li, av in enumerate(alpha[: self.num_layers]):
                a_tbl.add_row(Text(f"L{li:02d}", style="dim"), _bar(float(av), 1.0, width=18, color="cyan"))
            right.add_row(a_tbl)

        # Per-layer plan logits
        pl_pl = metrics.get("per_layer_plan")
        try:
            import torch  # type: ignore
            if isinstance(pl_pl, torch.Tensor) and pl_pl.dim() >= 3:
                # [B,L,K] → mean over batch → [L,K]
                arr = pl_pl.detach().float()
                mean_lk = arr.mean(dim=0)  # [L,K]
                right.add_row(Text("Per-layer Plan", style="bold cyan"))
                p_tbl = Table(show_header=True, header_style="bold cyan")
                p_tbl.add_column("Layer")
                p_tbl.add_column("Top Plan")
                p_tbl.add_column("Prob")
                for li in range(min(self.num_layers, mean_lk.shape[0])):
                    row = mean_lk[li]
                    probs = _softmax([float(v) for v in row.tolist()])
                    top_idx = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else 0
                    p_tbl.add_row(f"L{li:02d}", str(top_idx), _bar(probs[top_idx] if probs else 0.0, 1.0, width=10, color="cyan"))
                right.add_row(p_tbl)
        except Exception:
            pass
        self._render_right(right)

        # Log line
        msg = Text.assemble(
            (f"step {self._step}", "bold cyan"),
            (" | ", "dim"),
            (f"loss={_as_float(metrics.get('loss_total'))}", "bold"),
            (" | ", "dim"),
            (f"budget={_as_float(metrics.get('budget_pred'))}", "bold"),
        )
        self._log.append(msg)
        self._render_bottom()

        # Paint
        if self.live is not None:
            self.live.update(self.layout)

    # ---- Context manager ------------------------------------------------------
    def __enter__(self) -> "TrainingDashboard":
        self.live = Live(self.layout, refresh_per_second=4, screen=True)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)
            self.live = None


def main() -> int:
    """Standalone: read JSON dict metrics per line on stdin and render dashboard."""
    # Default to 24 layers unless overridden by first record
    num_layers = 24
    dash: Optional[TrainingDashboard] = None
    try:
        for line in sys.stdin:
            line = (line or "").strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if dash is None:
                nl = m.get("num_layers") or m.get("layers")
                try:
                    num_layers = int(nl) if nl is not None else num_layers
                except Exception:
                    pass
                dash = TrainingDashboard(num_layers)
                dash.__enter__()
            dash.update(m)
    finally:
        if dash is not None:
            dash.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

