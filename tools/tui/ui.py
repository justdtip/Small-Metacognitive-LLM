from __future__ import annotations

import json
import os
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


def _entropy(p: List[float], eps: float = 1e-12) -> float:
    import math
    if not p:
        return 0.0
    s = 0.0
    for x in p:
        x = max(eps, min(1.0, float(x)))
        s -= x * math.log(x)
    return s


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
    txt.append(f" {v:.3f}", style="bold")
    return txt


def _spark(values: List[float], width: int = 24, color: str = "cyan") -> Text:
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return Text("" )
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    step = max(1, len(values) // max(1, width))
    vals = values[::step]
    txt = Text()
    for v in vals:
        idx = int(round((v - lo) / rng * (len(blocks) - 1)))
        idx = max(0, min(len(blocks) - 1, idx))
        txt.append(blocks[idx], style=color)
    return txt


class CoTSpaceTUI:
    """
    Rich-based TUI exposing CoT-space knobs and outcomes.

    Panels:
    - Meta Policy: temperature, top_p, repetition_penalty, B_max, think_ratio
    - Expert Mix: weights per expert (bar chart) and entropy
    - Layer Aggregator: alpha min/mean/max + attention entropy, sparkline
    - CoT Complexity: E[L], E[|xi|], product trend
    - Verifier: pass rate, ECE/Brier (uses provided metrics if present)
    - Memory & CUDA: RAM/GPU usage (if available)

    Toggle: send a JSON line {"toggle":"thoughts"} to show/hide recent thoughts if supplied.
    """

    def __init__(self, width: int = 120, height: int = 42):
        self.width = int(max(80, width))
        self.height = int(max(24, height))
        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="top", size=3),
            Layout(name="body"),
            Layout(name="bottom", size=10),
        )
        self.layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="center", ratio=3),
            Layout(name="right", ratio=3),
        )
        self.live: Optional[Live] = None
        self._step = 0
        self._log: List[Text] = []
        self._think_trend: List[float] = []
        self._answer_trend: List[float] = []
        self._show_thoughts: bool = False
        self._render_all()

    def _render_all(self):
        self._render_top()
        self._render_left(Table.grid())
        self._render_center(Table.grid())
        self._render_right(Table.grid())
        self._render_bottom()

    def _render_top(self, clock: Optional[str] = None):
        now = clock or time.strftime("%H:%M:%S")
        title = Text(f" CoT-Space TUI ", style="bold cyan")
        info = Text(f" {now} | step {self._step} ", style="bold dim")
        self.layout["top"].update(Panel.fit(Text.assemble(title, info), style="on black", border_style="cyan"))

    def _render_left(self, renderable):
        self.layout["left"].update(Panel(renderable, title="Meta + Complexity + Memory", title_align="left", style="on black", border_style="cyan"))

    def _render_center(self, renderable):
        self.layout["center"].update(Panel(renderable, title="Experts + Aggregator", title_align="left", style="on black", border_style="cyan"))

    def _render_right(self, renderable):
        self.layout["right"].update(Panel(renderable, title="Verifier + Telemetry", title_align="left", style="on black", border_style="cyan"))

    def _render_bottom(self):
        # Keep last ~200 lines
        if len(self._log) > 200:
            self._log = self._log[-200:]
        log_text = Text()
        for line in self._log[-12:]:
            log_text.append(line)
            log_text.append("\n")
        self.layout["bottom"].update(Panel(log_text, title="Log / Thoughts (toggle via {\"toggle\":\"thoughts\"})", title_align="left", style="on black", border_style="cyan"))

    # Panels ------------------------------------------------------------------
    def _panel_meta_policy(self, m: Dict[str, Any]) -> Table:
        t = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
        t.add_column("Param")
        t.add_column("Value")
        keys = [
            ("temperature", m.get("gen_temperature") or m.get("temperature") or m.get("policy", {}).get("temperature")),
            ("top_p", m.get("gen_top_p") or m.get("top_p") or m.get("policy", {}).get("top_p")),
            ("rep_penalty", m.get("gen_rep_penalty") or m.get("repetition_penalty") or m.get("policy", {}).get("repetition_penalty")),
            ("B_max", m.get("B_max") or (m.get("policy") or {}).get("B_max")),
            ("think_ratio", m.get("think_ratio") or (m.get("policy") or {}).get("think_ratio")),
        ]
        for name, val in keys:
            valf = _as_float(val)
            t.add_row(str(name), f"{valf if valf is not None else ''}")
        return t

    def _panel_expert_mix(self, m: Dict[str, Any]) -> Table:
        weights = _to_list(m.get("expert_weights") or (m.get("policy") or {}).get("expert_weights")) or []
        tbl = Table(show_header=True, header_style="bold cyan")
        tbl.add_column("Expert")
        tbl.add_column("Weight")
        for i, w in enumerate(weights):
            tbl.add_row(f"E{i}", _bar(float(w), 1.0, width=16, color="magenta"))
        H = _entropy(weights) if weights else 0.0
        tbl.caption = f"entropy={H:.3f}"
        return tbl

    def _panel_aggregator(self, m: Dict[str, Any]) -> Table:
        alpha_any = m.get("alpha") or (m.get("per_layer") or {}).get("alpha")
        alpha = _to_list(alpha_any) or []
        tbl = Table(show_header=True, header_style="bold cyan")
        tbl.add_column("alpha stats")
        if alpha:
            import statistics
            a_min = min(alpha); a_mean = statistics.fmean(alpha); a_max = max(alpha)
            Hn = 0.0
            try:
                import math
                H = _entropy(alpha)
                Hn = H / max(1e-6, math.log(float(len(alpha))))
            except Exception:
                Hn = 0.0
            tbl.add_row(f"min={a_min:.3f} mean={a_mean:.3f} max={a_max:.3f} Hn={Hn:.3f}")
            tbl.add_row(_spark(alpha, width=36, color="cyan"))
        else:
            tbl.add_row("no alpha")
        return tbl

    def _panel_complexity(self, m: Dict[str, Any]) -> Table:
        think_mean = _as_float(m.get("think_tokens_mean") or (m.get("batch_stats") or {}).get("think_tokens_mean")) or 0.0
        answer_mean = _as_float(m.get("answer_tokens_mean") or (m.get("batch_stats") or {}).get("answer_tokens_mean")) or 0.0
        prod = think_mean * answer_mean
        # track trend
        self._think_trend.append(think_mean); self._think_trend = self._think_trend[-200:]
        self._answer_trend.append(answer_mean); self._answer_trend = self._answer_trend[-200:]
        t = Table.grid(padding=(0, 1))
        t.add_row(Text("CoT Complexity", style="bold cyan"))
        t.add_row(Text(f"E[L]={think_mean:.2f}  E[|ξ|]={answer_mean:.2f}  product={prod:.2f}", style="dim"))
        t.add_row(Text("L trend:") , _spark(self._think_trend, width=36, color="green"))
        t.add_row(Text("|ξ| trend:"), _spark(self._answer_trend, width=36, color="yellow"))
        return t

    def _panel_verifier(self, m: Dict[str, Any]) -> Table:
        t = Table(show_header=False, box=None)
        acc = _as_float(m.get("accuracy") or (m.get("verifier") or {}).get("pass_rate") or (m.get("overall") or {}).get("accuracy"))
        ece = _as_float(m.get("ece") or (m.get("verifier") or {}).get("ece"))
        brier = _as_float(m.get("brier") or (m.get("verifier") or {}).get("brier"))
        t.add_row(Text("success@1:", style="cyan"), Text(f"{acc}"))
        t.add_row(Text("ECE:", style="cyan"), Text(f"{ece}"))
        t.add_row(Text("Brier:", style="cyan"), Text(f"{brier}"))
        return t

    def _panel_memory(self) -> Table:
        t = Table(show_header=False, box=None)
        # System memory via psutil if available
        ram = None
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            ram = (vm.used / (1024**3), vm.total / (1024**3))
        except Exception:
            pass
        if ram is not None:
            t.add_row(Text("RAM:", style="cyan"), _bar(ram[0], ram[1], width=20, color="cyan"))
        # CUDA memory if available
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                reserved = torch.cuda.memory_reserved(idx) / (1024**3)
                allocated = torch.cuda.memory_allocated(idx) / (1024**3)
                t.add_row(Text("CUDA res:", style="cyan"), _bar(reserved, total, width=20, color="magenta"))
                t.add_row(Text("CUDA alloc:", style="cyan"), _bar(allocated, total, width=20, color="magenta"))
        except Exception:
            pass
        return t

    # Public ------------------------------------------------------------------
    def update(self, metrics: Dict[str, Any]):
        # toggle thoughts
        if metrics.get("toggle") == "thoughts":
            self._show_thoughts = not self._show_thoughts
            return
        self._step = int(metrics.get("step") or metrics.get("t") or 0)
        self._render_top()

        # LEFT: Meta Policy + Complexity + Memory
        left = Table.grid(padding=(0, 1))
        left.add_row(self._panel_meta_policy(metrics))
        left.add_row(self._panel_complexity(metrics))
        left.add_row(self._panel_memory())
        self._render_left(left)

        # CENTER: Expert mix + Aggregator
        center = Table.grid(padding=(0, 1))
        center.add_row(Text("Expert Mix", style="bold cyan"))
        center.add_row(self._panel_expert_mix(metrics))
        center.add_row(Text("Layer Aggregator", style="bold cyan"))
        center.add_row(self._panel_aggregator(metrics))
        self._render_center(center)

        # RIGHT: Verifier + telemetry
        right = Table.grid(padding=(0, 1))
        right.add_row(self._panel_verifier(metrics))
        # Telemetry extras if provided
        if metrics.get("alpha_summary"):
            s = metrics.get("alpha_summary") or {}
            right.add_row(Text(f"alpha min={s.get('min')} mean={s.get('mean')} max={s.get('max')}", style="dim"))
        if metrics.get("selector_entropy") is not None:
            right.add_row(Text(f"selector_entropy={_as_float(metrics.get('selector_entropy'))}", style="dim"))
        self._render_right(right)

        # Bottom log: recent thoughts if available
        msg = Text.assemble(
            (f"step {self._step}", "bold cyan"),
            (" | ", "dim"),
            (f"temp={_as_float(metrics.get('gen_temperature') or (metrics.get('policy') or {}).get('temperature'))}", "bold"),
            (" | ", "dim"),
            (f"B_max={_as_float(metrics.get('B_max'))}", "bold"),
        )
        self._log.append(msg)
        # thoughts
        if self._show_thoughts:
            ths: List[str] = metrics.get("thoughts") or []
            stop_reason = str(metrics.get("stop_reason") or "")
            for s in ths[-5:]:
                style = "green" if stop_reason == "early" else ("yellow" if stop_reason == "budget" else "white")
                self._log.append(Text(s, style=style))
        self._render_bottom()

        if self.live is not None:
            self.live.update(self.layout)

    def __enter__(self) -> "CoTSpaceTUI":
        self.live = Live(self.layout, refresh_per_second=4, screen=True)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)
            self.live = None


def main() -> int:
    """Standalone: read JSON dict metrics per line on stdin and render dashboard."""
    dash: Optional[CoTSpaceTUI] = None
    try:
        for line in sys.stdin:
            line = (line or "").strip()
            if not line:
                continue
            # Support bare toggle lines
            if line.lower() in {"t", "thoughts"}:
                if dash is not None:
                    dash.update({"toggle": "thoughts"})
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if dash is None:
                dash = CoTSpaceTUI()
                dash.__enter__()
            dash.update(m)
    finally:
        if dash is not None:
            dash.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

