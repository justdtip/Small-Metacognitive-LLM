from __future__ import annotations

import asyncio
import io
import json
import queue
import sys as _sys
import threading
from pathlib import Path
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Header,
        Footer,
        Static,
        Input,
        Button,
        Checkbox,
        Label,
        Log,
    )
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
except Exception as e:  # pragma: no cover
    print(
        "Textual is required for the training TUI.\n"
        "Install it with: pip install textual>=0.44 rich>=13.4\n"
        f"Import error: {type(e).__name__}: {e}"
    )
    raise SystemExit(1)

from pathlib import Path as _Path
# Ensure project root is importable when launching as a script
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from pathlib import Path as _P


class _QueueWriter(io.TextIOBase):
    """File-like writer that enqueues lines into a queue (for stdout capture)."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self._q = q
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            try:
                self._q.put_nowait(line)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._buf:
            try:
                self._q.put_nowait(self._buf)
            except Exception:
                pass
            self._buf = ""


class RunConfigForm(Static):
    """A form for configuring and launching training runs."""

    model_path = reactive("")
    data_path = reactive("")
    steps = reactive(1000)
    data_limit = reactive(100000)
    save_interval = reactive(1000)
    eval_interval = reactive(100)
    num_experts = reactive(3)
    feedback = reactive(True)
    run_name = reactive("run1")
    streaming = reactive(True)
    feedback_dim = reactive("")
    lm_adapt = reactive(True)
    adapt_mode = reactive("ln_only")
    adapt_layers = reactive(2)
    agg = reactive("attn")
    var_reg = reactive("0.001")
    mix_dataset = reactive("glaive")
    link_all = reactive(True)
    dump_per_layer = reactive(True)
    apply_all_layers = reactive(False)
    adapt_layers_list = reactive("")
    sample_every = reactive(0)
    budget_cap = reactive(16)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="form"):
            yield Label("Model Path:")
            yield Input(id="model_path", placeholder="e.g. models/Qwen1.5B or model/Base", value=self.model_path)
            yield Label("Data Path (HF or local):")
            yield Input(id="data_path", placeholder="e.g. glaive or data/train.jsonl", value=self.data_path)
            yield Checkbox(label="Streaming (HF datasets)", value=self.streaming, id="streaming_chk")
            yield Label("Dataset Name or Mixture Key:")
            yield Input(id="mix_dataset", value=self.mix_dataset)
            yield Label("Steps:")
            yield Input(id="steps", value=str(self.steps))
            yield Label("Dataset Limit (#examples):")
            yield Input(id="data_limit", value=str(self.data_limit))
            yield Label("Save Interval:")
            yield Input(id="save_interval", value=str(self.save_interval))
            yield Label("Eval Interval:")
            yield Input(id="eval_interval", value=str(self.eval_interval))
            yield Label("Sample Every:")
            yield Input(id="sample_every", value=str(self.sample_every))
            yield Label("Budget Cap:")
            yield Input(id="budget_cap", value=str(self.budget_cap))
            yield Label("Number of Experts:")
            yield Input(id="num_experts", value=str(self.num_experts))
            yield Checkbox(label="Enable Feedback", value=self.feedback, id="feedback_chk")
            yield Label("Feedback Dim (blank for auto):")
            yield Input(id="feedback_dim", value=self.feedback_dim)
            yield Label("LM Adaptation:")
            yield Checkbox(label="Enable LM Adaptation", value=self.lm_adapt, id="lm_adapt_chk")
            yield Label("Adaptation Mode (ln_only/lora):")
            yield Input(id="adapt_mode", value=self.adapt_mode)
            yield Label("Last K Layers to Adapt:")
            yield Input(id="adapt_layers", value=str(self.adapt_layers))
            yield Label("Aggregator (attn or mean):")
            yield Input(id="agg", value=self.agg)
            yield Label("Variance Regularizer:")
            yield Input(id="var_reg", value=self.var_reg)
            yield Checkbox(label="Attach heads to all layers (linked_all_layers)", value=self.link_all, id="link_all_chk")
            yield Checkbox(label="Dump per-layer diagnostics", value=self.dump_per_layer, id="dump_pl_chk")
            yield Checkbox(label="Adaptation: apply to all layers", value=self.apply_all_layers, id="apply_all_layers_chk")
            yield Label("Adapt Layers List (comma-separated indices):")
            yield Input(id="adapt_layers_list", value=self.adapt_layers_list)
            yield Label("Run Name:")
            yield Input(id="run_name", value=self.run_name)
            yield Button("Start Training", id="start_btn", variant="success")
            self.status = Log(id="status", highlight=False, auto_scroll=True)
            yield self.status

    async def on_mount(self) -> None:  # type: ignore[override]
        # Write initial line once the app is active
        try:
            self.status.write_line("Ready")
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:  # type: ignore[override]
        """Update reactive fields when inputs change."""
        try:
            if event.input.id == "steps":
                self.steps = int(event.value)
            elif event.input.id == "data_limit":
                self.data_limit = int(event.value)
            elif event.input.id == "save_interval":
                self.save_interval = int(event.value)
            elif event.input.id == "eval_interval":
                self.eval_interval = int(event.value)
            elif event.input.id == "sample_every":
                self.sample_every = int(event.value)
            elif event.input.id == "budget_cap":
                self.budget_cap = int(event.value)
            elif event.input.id == "num_experts":
                self.num_experts = int(event.value)
            elif event.input.id == "model_path":
                self.model_path = event.value.strip()
            elif event.input.id == "data_path":
                self.data_path = event.value.strip()
            elif event.input.id == "run_name":
                self.run_name = event.value.strip()
            elif event.input.id == "feedback_dim":
                self.feedback_dim = event.value.strip()
            elif event.input.id == "adapt_mode":
                self.adapt_mode = event.value.strip()
            elif event.input.id == "adapt_layers":
                try:
                    self.adapt_layers = int(event.value)
                except ValueError:
                    pass
            elif event.input.id == "agg":
                self.agg = event.value.strip()
            elif event.input.id == "var_reg":
                self.var_reg = event.value.strip()
            elif event.input.id == "mix_dataset":
                self.mix_dataset = event.value.strip()
            elif event.input.id == "adapt_layers_list":
                self.adapt_layers_list = event.value.strip()
        except ValueError:
            # Ignore non-integer input but keep previous state
            pass

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:  # type: ignore[override]
        if event.checkbox.id == "feedback_chk":
            self.feedback = event.value
        elif event.checkbox.id == "streaming_chk":
            self.streaming = event.value
        elif event.checkbox.id == "lm_adapt_chk":
            self.lm_adapt = event.value
        elif event.checkbox.id == "link_all_chk":
            self.link_all = event.value
        elif event.checkbox.id == "dump_pl_chk":
            self.dump_per_layer = event.value
        elif event.checkbox.id == "apply_all_layers_chk":
            self.apply_all_layers = event.value

    async def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id != "start_btn":
            return
        # Build a config dict and run training asynchronously
        # Dataset selection precedence:
        # - If mix_dataset is a known HF key (not 'jsonl'), use it
        # - Else if data_path ends with .jsonl or not in known keys, use local JSONL path
        # - Else fall back to jsonl default
        known = {"glaive", "gsm8k", "math", "jsonl"}
        md = (self.mix_dataset or "").strip().lower()
        dp = (self.data_path or "").strip()
        if md in known and md != "jsonl":
            data_cfg = {"dataset_name": md, "limit_train": int(self.data_limit), "split": "train", "streaming": bool(self.streaming)}
        elif dp.lower().endswith('.jsonl') or dp not in known:
            data_cfg = {"dataset_name": "jsonl", "jsonl": dp or "data/train.jsonl", "limit_train": int(self.data_limit)}
        else:
            data_cfg = {"dataset_name": "jsonl", "jsonl": "data/train.jsonl", "limit_train": int(self.data_limit)}

        cfg: dict[str, Any] = {
            "model": {"base": self.model_path or "model/Base"},
            "trainer": {
                "mode": "supervised",
                "save_interval": int(self.save_interval),
                "eval_interval": int(self.eval_interval),
                "dashboard": True,
            },
            # Duplicate at top level for hooks used in runner
            "save_interval": int(self.save_interval),
            "eval_interval": int(self.eval_interval),
            "sample_every": int(self.sample_every),
            "budget_cap": int(self.budget_cap),
            "data": data_cfg,
            "metacog": {
                "num_experts": int(self.num_experts),
                "feedback": bool(self.feedback),
                "feedback_dim": (int(self.feedback_dim) if self.feedback_dim.strip().isdigit() else None),
                "linked_all_layers": bool(self.link_all),
                "agg": (self.agg or "attn"),
                "var_reg": float(self.var_reg or 0.0),
                "dump_per_layer": bool(self.dump_per_layer),
            },
            "lm_adaptation": {
                "enabled": bool(self.lm_adapt),
                "mode": (self.adapt_mode or "ln_only"),
                "last_k_layers": int(self.adapt_layers),
                "apply_to_all_layers": bool(self.apply_all_layers),
                "layer_indices": [int(x) for x in self.adapt_layers_list.split(',') if x.strip().isdigit()],
            },
        }
        try:
            cfg_path = Path(f"{self.run_name or 'run'}.json")
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            self.status.write_line("Starting training… (streaming logs)")

            # Streaming queue and writer
            q: "queue.Queue[str]" = queue.Queue()
            writer = _QueueWriter(q)

            def _runner(path: str, steps: int) -> None:
                # Ensure repo root on sys.path for import
                root = _P(__file__).resolve().parents[1]
                if str(root) not in _sys.path:
                    _sys.path.insert(0, str(root))
                # Redirect stdout inside this thread
                old = _sys.stdout
                try:
                    _sys.stdout = writer  # type: ignore[assignment]
                    from train.runner import run_from_config as _run
                    _run(path, steps=int(steps))
                except Exception as _e:
                    # Capture full stack trace and ship it to the log queue
                    try:
                        import traceback as _tb
                        trace_lines = _tb.format_exc().splitlines()
                        record = {"error": f"{type(_e).__name__}: {_e}", "trace": trace_lines}
                        q.put_nowait(json.dumps(record))
                    except Exception:
                        # As a last resort, send the repr
                        try:
                            q.put_nowait(f"[EXC] {type(_e).__name__}: {_e}")
                        except Exception:
                            pass
                finally:
                    _sys.stdout = old

            # Start background runner
            thr = threading.Thread(target=_runner, args=(str(cfg_path), int(self.steps)), daemon=True)
            thr.start()

            # Drainer thread: consume stdout and update status live
            def _drain() -> None:
                while thr.is_alive() or not q.empty():
                    try:
                        line = q.get(timeout=0.25)
                    except Exception:
                        continue
                    # Parse JSON if possible
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            # Known structured lines
                            if "train_step" in rec:
                                msg = f"step={rec.get('train_step')}"
                                self.app.call_from_thread(self.status.write_line, msg)
                                continue
                            if "onpolicy" in rec:
                                msg = f"onpolicy: {rec['onpolicy']}"
                                self.app.call_from_thread(self.status.write_line, msg)
                                continue
                            if "manifest" in rec:
                                self.app.call_from_thread(self.status.write_line, "manifest written")
                                continue
                            if "error" in rec:
                                # Print error then full stack trace if present
                                self.app.call_from_thread(self.status.write_line, f"Error: {rec['error']}")
                                tr = rec.get("trace")
                                if isinstance(tr, list) and tr:
                                    for tl in tr:
                                        self.app.call_from_thread(self.status.write_line, tl)
                                else:
                                    # If trace is a string, split lines
                                    if isinstance(tr, str):
                                        for tl in tr.splitlines():
                                            self.app.call_from_thread(self.status.write_line, tl)
                                continue
                            # Default: echo JSON compactly
                            msg = json.dumps(rec)
                        else:
                            msg = str(rec)
                    except Exception:
                        msg = line
                    # Update UI from thread
                    try:
                        self.app.call_from_thread(self.status.write_line, msg)
                    except Exception:
                        pass
                try:
                    self.app.call_from_thread(self.status.write_line, "Training finished.")
                except Exception:
                    pass

            dthr = threading.Thread(target=_drain, daemon=True)
            dthr.start()
            # Return immediately; background threads will keep updating the UI
        except Exception as e:  # surface any error in the UI
            self.status.write_line(f"Error: {type(e).__name__}: {e}")


class TrainingLauncherApp(App):
    CSS = """
    Screen { background: black; color: cyan; }
    #status { height: auto; color: green; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield RunConfigForm()
        yield Footer()


if __name__ == "__main__":
    app = TrainingLauncherApp()
    app.run()
