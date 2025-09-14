import os
import io
import datetime as _dt
from typing import Any


def _default_log_path() -> str:
    """Compute the default log file path.

    Uses env var PYTEST_OUTPUT_LOG if set; otherwise writes to logs/pytest-q.log.
    """
    env_path = os.getenv("PYTEST_OUTPUT_LOG")
    if env_path:
        return env_path
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "pytest-q.log")


class _TeeStream:
    """A minimal text stream that tees writes to an underlying stream and a log file.

    - Preserves terminal behavior (isatty, encoding) by delegating to the underlying stream.
    - Only duplicates write/flush into the log file; it never closes the underlying stream.
    """

    def __init__(self, main_stream: io.TextIOBase, log_stream: io.TextIOBase) -> None:
        self._main = main_stream
        self._log = log_stream

    # --- core text I/O methods ---
    def write(self, s: str) -> int:  # type: ignore[override]
        n = self._main.write(s)
        self._log.write(s)
        return n

    def writelines(self, lines: Any) -> None:  # pragma: no cover - rarely used by pytest
        self._main.writelines(lines)
        self._log.writelines(lines)

    def flush(self) -> None:
        self._main.flush()
        self._log.flush()

    # --- compatibility helpers ---
    def isatty(self) -> bool:  # preserve terminal color/width behavior
        try:
            return bool(self._main.isatty())
        except Exception:
            return False

    @property
    def encoding(self) -> str:
        return getattr(self._main, "encoding", "utf-8")

    def fileno(self) -> int:  # pragma: no cover - seldom needed
        return self._main.fileno()

    # Never close the real stdout/stderr; only close our log when asked
    def close(self) -> None:
        try:
            self._log.close()
        finally:
            pass

    # Delegate any other attributes to the underlying stream for maximum compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self._main, name)


def pytest_configure(config) -> None:  # noqa: D401 - pytest hook
    """Install a tee on the terminal reporter so all console output is logged.

    Keeps the console output unchanged and writes identical content to a log file.
    """
    # Allow opt-out via env var
    if os.getenv("PYTEST_DISABLE_OUTPUT_LOG"):
        return

    tr = config.pluginmanager.getplugin("terminalreporter")
    if tr is None:
        return  # no terminal output (e.g., -p no:terminal)

    tw = getattr(tr, "_tw", None)
    if tw is None:
        return

    # Determine and open the log file (text mode to match terminal writes)
    log_path = _default_log_path()
    try:
        log_fh = open(log_path, "w", encoding="utf-8", newline="")
    except OSError:
        # If logs dir not writable, fall back to cwd
        fallback = os.path.join(os.getcwd(), "pytest-q.log")
        log_fh = open(fallback, "w", encoding="utf-8", newline="")
        log_path = fallback

    # Resolve the underlying stream attribute used by TerminalWriter
    # Prefer private _file when present (pytest >= 7), else fall back to file/stream.
    underlying = None
    for attr in ("_file", "file", "stream", "out"):
        if hasattr(tw, attr):
            underlying = getattr(tw, attr)
            break
    if underlying is None:
        # As a last resort, give up quietly
        log_fh.close()
        return

    tee = _TeeStream(underlying, log_fh)

    # Store bookkeeping so we can restore on unconfigure
    config._pytest_output_log = {  # type: ignore[attr-defined]
        "log_fh": log_fh,
        "log_path": log_path,
        "tw": tw,
        "underlying_attr": attr,
        "underlying_stream": underlying,
        "tee": tee,
    }

    # Attach tee to the terminal writer without changing console behavior
    if hasattr(tw, "_file"):
        tw._file = tee  # type: ignore[attr-defined]
    elif hasattr(tw, "file"):
        try:
            setattr(tw, "file", tee)  # type: ignore[misc]
        except Exception:
            # Fallback: monkeypatch write method if assignment is not allowed
            _install_write_wrapper(tw, log_fh)
    else:
        _install_write_wrapper(tw, log_fh)


def _install_write_wrapper(tw, log_fh) -> None:
    """Fallback: wrap tw.write to duplicate text into the log file."""
    orig_write = getattr(tw, "write", None)
    if orig_write is None:
        return

    def _wrapped_write(s: str, *args, **kwargs):
        # Call original writer first to keep console timing/format intact
        res = orig_write(s, *args, **kwargs)
        try:
            log_fh.write(s)
        except Exception:
            # Best-effort logging only; never interfere with pytest output
            pass
        return res

    tw.write = _wrapped_write  # type: ignore[assignment]


def pytest_unconfigure(config) -> None:  # noqa: D401 - pytest hook
    """Restore terminal writer and close the log file."""
    state = getattr(config, "_pytest_output_log", None)
    if not state:
        return

    tw = state.get("tw")
    underlying_attr = state.get("underlying_attr")
    underlying_stream = state.get("underlying_stream")
    log_fh = state.get("log_fh")

    try:
        if tw is not None and underlying_attr in ("_file", "file", "stream", "out"):
            try:
                setattr(tw, underlying_attr, underlying_stream)
            except Exception:
                pass
    finally:
        try:
            if log_fh is not None:
                log_fh.flush()
                log_fh.close()
        except Exception:
            pass

