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
    )
    from textual.containers import Horizontal, Vertical
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
    num_experts = reactive(1)
    feedback = reactive(False)
    run_name = reactive("run1")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Model Path:")
            yield Input(id="model_path", placeholder="e.g. models/Qwen1.5B or model/Base", value=self.model_path)
            yield Label("Data Path (HF or local):")
            yield Input(id="data_path", placeholder="e.g. glaive or data/train.jsonl", value=self.data_path)
            yield Label("Steps:")
            yield Input(id="steps", value=str(self.steps))
            yield Label("Dataset Limit (#examples):")
            yield Input(id="data_limit", value=str(self.data_limit))
            yield Label("Save Interval:")
            yield Input(id="save_interval", value=str(self.save_interval))
            yield Label("Eval Interval:")
            yield Input(id="eval_interval", value=str(self.eval_interval))
            yield Label("Number of Experts:")
            yield Input(id="num_experts", value=str(self.num_experts))
            yield Checkbox(label="Enable Feedback", value=self.feedback, id="feedback_chk")
            yield Label("Run Name:")
            yield Input(id="run_name", value=self.run_name)
            yield Button("Start Training", id="start_btn", variant="success")
            self.status = Static("Ready", id="status")
            yield self.status

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
            elif event.input.id == "num_experts":
                self.num_experts = int(event.value)
            elif event.input.id == "model_path":
                self.model_path = event.value.strip()
            elif event.input.id == "data_path":
                self.data_path = event.value.strip()
            elif event.input.id == "run_name":
                self.run_name = event.value.strip()
        except ValueError:
            # Ignore non-integer input but keep previous state
            pass

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:  # type: ignore[override]
        if event.checkbox.id == "feedback_chk":
            self.feedback = event.value

    async def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
        if event.button.id != "start_btn":
            return
        # Build a config dict and run training asynchronously
        # If data_path equals a known dataset name, use it; else treat as jsonl path
        dp = (self.data_path or "").strip()
        known = {"glaive", "gsm8k", "math", "jsonl"}
        if dp in known and dp != "jsonl":
            data_cfg = {"dataset_name": dp, "limit_train": int(self.data_limit), "split": "train", "streaming": False}
        else:
            data_cfg = {"dataset_name": "jsonl", "jsonl": dp or "data/train.jsonl", "limit_train": int(self.data_limit)}

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
            "data": data_cfg,
            "metacog": {
                "num_experts": int(self.num_experts),
                "feedback": bool(self.feedback),
                "feedback_dim": None,
                "linked_all_layers": True,
                "agg": "attn",
                "var_reg": 0.0,
            },
        }
        try:
            cfg_path = Path(f"{self.run_name or 'run'}.json")
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            self.status.update("Starting training… (streaming logs)")

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
                    try:
                        q.put_nowait(json.dumps({"error": f"{type(_e).__name__}: {_e}"}))
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
                            if "train_step" in rec:
                                msg = f"step={rec.get('train_step')}"
                            elif "onpolicy" in rec:
                                msg = f"onpolicy: {rec['onpolicy']}"
                            elif "manifest" in rec:
                                msg = "manifest written"
                            elif "error" in rec:
                                msg = f"Error: {rec['error']}"
                            else:
                                msg = json.dumps(rec)
                        else:
                            msg = str(rec)
                    except Exception:
                        msg = line
                    # Update UI from thread
                    try:
                        self.app.call_from_thread(self.status.update, msg)
                    except Exception:
                        pass
                try:
                    self.app.call_from_thread(self.status.update, "Training finished.")
                except Exception:
                    pass

            dthr = threading.Thread(target=_drain, daemon=True)
            dthr.start()
            # Return immediately; background threads will keep updating the UI
        except Exception as e:  # surface any error in the UI
            self.status.update(f"Error: {type(e).__name__}: {e}")


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
