from __future__ import annotations

"""
Interactive training TUI driven by Textual.

If Textual is not installed, prints a helpful message and exits.
"""

import io
import json
import queue
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Button, Input, Label, Select, Checkbox, Static
    from textual.reactive import reactive
except Exception as e:  # pragma: no cover
    print("Textual is required for the training TUI.\n"
          "Install it with: pip install textual>=0.44 rich>=13.4\n"
          f"Import error: {type(e).__name__}: {e}")
    raise SystemExit(1)


class _QueueWriter(io.TextIOBase):
    """File-like writer that pushes complete lines into a queue."""

    def __init__(self, q: queue.Queue[str]):
        super().__init__()
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put_nowait(line)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._buf:
            self.q.put_nowait(self._buf)
            self._buf = ""


def _safe_int(val: str | None, default: int) -> int:
    try:
        return int(str(val).strip())
    except Exception:
        return int(default)


@dataclass
class TrainingParams:
    base_model_path: str = "model/Base"
    data_path: str = "data/train.jsonl"
    dataset_name: str = "jsonl"
    max_steps: int = 5
    dataset_limit: int = 100
    save_interval: int = 1000
    eval_interval: int = 100
    num_experts: int = 1
    feedback: bool = False
    agg: str = "attn"


class StatusPanel(Static):
    last_json: reactive[Optional[Dict[str, Any]]] = reactive(None)  # type: ignore[type-arg]
    last_text: reactive[str] = reactive("")

    def render(self) -> Any:  # rich renderable
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        if self.last_json:
            js = self.last_json or {}
            t = Table(show_header=True, header_style="bold cyan", expand=True)
            t.add_column("Key"); t.add_column("Value")
            # Surface a handful of interesting keys if present
            if "train_step" in js:
                t.add_row("step", str(js.get("train_step")))
            rl = js.get("rl") if isinstance(js.get("rl"), dict) else {}
            for k in ("think_tokens_used", "mu_mean", "mu_after", "reward_mean", "plan_fraction", "exec_fraction", "eval_fraction"):
                if k in rl:
                    t.add_row(k, str(rl.get(k)))
            if "manifest" in js:
                t.add_row("manifest", "written")
            return Panel(t, title="Status", border_style="cyan")
        # Fallback: show last text
        return Panel(Text(self.last_text or "Waiting...", style="green"), title="Status", border_style="cyan")


class TrainingApp(App):
    CSS = """
    Screen { background: black; color: cyan; }
    .header { color: ansi_bright_cyan; }
    .label { width: 24; }
    .input { width: 48; }
    .narrow { width: 20; }
    .btn { background: #004444; color: white; }
    #form { padding: 1 2; }
    #status { height: 1fr; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.params = TrainingParams()
        self._q: queue.Queue[str] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self.status = StatusPanel(id="status")

    def compose(self) -> ComposeResult:
        yield Label("Training TUI (eDex style)", classes="header")
        with Horizontal():
            with VerticalScroll(id="form"):
                yield Label("Base model path", classes="label")
                self.in_model = Input(placeholder=self.params.base_model_path, classes="input")
                yield self.in_model

                yield Label("Data path (jsonl)", classes="label")
                self.in_data = Input(placeholder=self.params.data_path, classes="input")
                yield self.in_data

                yield Label("Dataset name", classes="label")
                self.sel_ds = Select((("jsonl", "jsonl"), ("glaive", "glaive"), ("gsm8k", "gsm8k"), ("math", "math")))
                yield self.sel_ds

                yield Label("Max steps", classes="label")
                self.in_steps = Input(placeholder=str(self.params.max_steps), classes="narrow")
                yield self.in_steps

                yield Label("Dataset limit", classes="label")
                self.in_limit = Input(placeholder=str(self.params.dataset_limit), classes="narrow")
                yield self.in_limit

                yield Label("Save interval", classes="label")
                self.in_save = Input(placeholder=str(self.params.save_interval), classes="narrow")
                yield self.in_save

                yield Label("Eval interval", classes="label")
                self.in_eval = Input(placeholder=str(self.params.eval_interval), classes="narrow")
                yield self.in_eval

                yield Label("Num experts", classes="label")
                self.in_experts = Input(placeholder=str(self.params.num_experts), classes="narrow")
                yield self.in_experts

                yield Label("Feedback", classes="label")
                self.cb_feedback = Checkbox(value=self.params.feedback)
                yield self.cb_feedback

                yield Label("Aggregator (agg)", classes="label")
                self.sel_agg = Select((("attn", "attn"), ("mean", "mean")))
                yield self.sel_agg

                yield Button("Run", id="run", classes="btn")

            yield self.status

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id != "run":
            return
        # Collect values safely
        p = TrainingParams(
            base_model_path=self.in_model.value or self.in_model.placeholder or self.params.base_model_path,
            data_path=self.in_data.value or self.in_data.placeholder or self.params.data_path,
            dataset_name=str(self.sel_ds.value or self.params.dataset_name),
            max_steps=_safe_int(self.in_steps.value, self.params.max_steps),
            dataset_limit=_safe_int(self.in_limit.value, self.params.dataset_limit),
            save_interval=_safe_int(self.in_save.value, self.params.save_interval),
            eval_interval=_safe_int(self.in_eval.value, self.params.eval_interval),
            num_experts=_safe_int(self.in_experts.value, self.params.num_experts),
            feedback=bool(self.cb_feedback.value),
            agg=str(self.sel_agg.value or self.params.agg),
        )
        self.params = p
        self._start_training(p)

    def _start_training(self, p: TrainingParams) -> None:
        # Build train config dict compatible with run_from_config
        cfg: Dict[str, Any] = {
            "model": {"base": p.base_model_path},
            "data": {"dataset_name": p.dataset_name, "jsonl": p.data_path, "limit_train": p.dataset_limit},
            "save_interval": p.save_interval,
            "save_dir": "checkpoints",
            "eval_interval": p.eval_interval,
            "eval_prompts": ["1+1=?"],
            "metacog": {"linked_all_layers": True, "agg": p.agg, "num_experts": p.num_experts, "feedback": p.feedback},
        }
        # Serialize to a temp file (run_from_config expects a path)
        tf = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml")
        json.dump(cfg, tf)
        tf.flush(); tf.close()

        # Worker thread: run training and push stdout lines to queue
        def _worker():
            # Push a starting message so the UI doesn't appear idle
            self._q.put_nowait("[INFO] Starting training...")
            # Import inside thread and report failures
            try:
                from train.runner import run_from_config as _run
            except Exception as e:
                self._q.put_nowait(f"[ERROR] Failed to import training runner: {type(e).__name__}: {e}")
                return
            qwriter = _QueueWriter(self._q)
            old_stdout = sys.stdout
            try:
                sys.stdout = qwriter  # type: ignore[assignment]
                _run(tf.name, steps=int(max(1, p.max_steps)))
            except Exception as e:
                self._q.put_nowait(f"[ERROR] {type(e).__name__}: {e}")
            finally:
                sys.stdout = old_stdout

        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=_worker, daemon=True)
        self._worker.start()
        # Poll queue periodically
        self.set_interval(0.25, self._drain_queue)

    def _drain_queue(self) -> None:
        # Consume a few messages per tick
        for _ in range(10):
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                break
            # Try to parse json lines
            try:
                js = json.loads(line)
                self.status.last_json = js
            except Exception:
                self.status.last_text = line


def main() -> int:
    app = TrainingApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
