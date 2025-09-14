import os
import io
import datetime as _dt
from typing import Any
import sys
import threading
import os as _os


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


def _ensure_logging(config) -> None:
    """Idempotently ensure tee logging is installed."""
    if os.getenv("PYTEST_DISABLE_OUTPUT_LOG"):
        return

    state = getattr(config, "_pytest_output_log", None)
    if state is None:
        # Determine and open the log file (text mode to match terminal writes)
        log_path = _default_log_path()
        try:
            log_fh = open(log_path, "w", encoding="utf-8", newline="")
        except OSError:
            # If logs dir not writable, fall back to cwd
            fallback = os.path.join(os.getcwd(), "pytest-q.log")
            log_fh = open(fallback, "w", encoding="utf-8", newline="")
            log_path = fallback

        # Debug breadcrumb file
        try:
            dbg_dir = os.path.dirname(log_path)
            os.makedirs(dbg_dir, exist_ok=True)
            with open(os.path.join(dbg_dir, "pytest-tee-debug.txt"), "a", encoding="utf-8") as _dbg:
                _dbg.write("opened log at %s\n" % log_path)
        except Exception:
            pass

        state = {
            "log_fh": log_fh,
            "log_path": log_path,
            "tw": None,
            "orig_write": None,
            "orig_fullwrite": None,
            "gtw_module": None,
            "gtw_orig": None,
            "sys_stdout_orig": None,
            "sys_stderr_orig": None,
            "installed": False,
        }
        config._pytest_output_log = state  # type: ignore[attr-defined]

        # Install FD-level tee to capture very-early and low-level writes
        try:
            _install_fd_tee(state)
        except Exception:
            # Best-effort only
            pass

        # Patch global get_terminal_writer ASAP so any ad-hoc writers are wrapped
        try:
            try:
                import _pytest._io.terminalwriter as _twmod  # type: ignore
            except Exception:  # pragma: no cover - legacy fallback
                import py._io.terminalwriter as _twmod  # type: ignore

            gtw_orig = getattr(_twmod, "get_terminal_writer", None)
            if gtw_orig is not None:
                def _gtw_wrapper(file=None):  # type: ignore[override]
                    tw2 = gtw_orig(file)
                    try:
                        _install_write_wrapper(tw2, state["log_fh"])  # type: ignore[index]
                    except Exception:
                        pass
                    try:
                        with open(os.path.join(state["log_path"].rsplit(os.sep, 1)[0], "pytest-tee-debug.txt"), "a", encoding="utf-8") as _dbg:  # type: ignore[index]
                            _dbg.write("wrapped get_terminal_writer\n")
                    except Exception:
                        pass
                    return tw2

                state["gtw_module"] = _twmod
                state["gtw_orig"] = gtw_orig
                _twmod.get_terminal_writer = _gtw_wrapper  # type: ignore[assignment]
        except Exception:
            pass

    # Attach to TerminalReporter if present and not yet installed
    tr = config.pluginmanager.getplugin("terminalreporter")
    if tr is None:
        return
    tw = getattr(tr, "_tw", None)
    if tw is None:
        return
    if state.get("installed"):
        return
    state["tw"] = tw
    state["orig_write"] = getattr(tw, "write", None)
    state["orig_fullwrite"] = getattr(tw, "fullwrite", None)
    _install_write_wrapper(tw, state["log_fh"])  # type: ignore[index]
    state["installed"] = True

    # As a last line of defense, also tee sys.stdout and sys.stderr
    # This helps capture very-early messages and error printing paths
    if state.get("sys_stdout_orig") is None:
        state["sys_stdout_orig"] = sys.stdout
        sys.stdout = _TeeStream(sys.stdout, state["log_fh"])  # type: ignore[assignment,index]
    if state.get("sys_stderr_orig") is None:
        state["sys_stderr_orig"] = sys.stderr
        sys.stderr = _TeeStream(sys.stderr, state["log_fh"])  # type: ignore[assignment,index]


def pytest_configure(config) -> None:  # noqa: D401 - pytest hook
    _ensure_logging(config)


def pytest_sessionstart(session) -> None:  # noqa: D401 - pytest hook
    _ensure_logging(session.config)


def _install_write_wrapper(tw, log_fh) -> None:
    """Wrap tw.write and tw.fullwrite to duplicate text into the log file."""
    orig_write = getattr(tw, "write", None)
    if orig_write is not None:
        def _wrapped_write(s, *args, **kwargs):  # type: ignore[no-redef]
            res = orig_write(s, *args, **kwargs)
            try:
                log_fh.write(s)
            except Exception:
                pass
            return res
        tw.write = _wrapped_write  # type: ignore[assignment]

    orig_fullwrite = getattr(tw, "fullwrite", None)
    if orig_fullwrite is not None:
        def _wrapped_fullwrite(s, *args, **kwargs):  # type: ignore[no-redef]
            res = orig_fullwrite(s, *args, **kwargs)
            try:
                log_fh.write(s)
            except Exception:
                pass
            return res
        tw.fullwrite = _wrapped_fullwrite  # type: ignore[assignment]


def pytest_unconfigure(config) -> None:  # noqa: D401 - pytest hook
    """Restore streams and close the log at very end of pytest lifecycle."""
    state = getattr(config, "_pytest_output_log", None)
    if not state:
        return

    tw = state.get("tw")
    log_fh = state.get("log_fh")
    orig_write = state.get("orig_write")
    orig_fullwrite = state.get("orig_fullwrite")
    gtw_module = state.get("gtw_module")
    gtw_orig = state.get("gtw_orig")
    sys_stdout_orig = state.get("sys_stdout_orig")
    sys_stderr_orig = state.get("sys_stderr_orig")

    try:
        if tw is not None:
            if orig_write is not None:
                try:
                    tw.write = orig_write  # type: ignore[assignment]
                except Exception:
                    pass
            if orig_fullwrite is not None:
                try:
                    tw.fullwrite = orig_fullwrite  # type: ignore[assignment]
                except Exception:
                    pass
        if gtw_module is not None and gtw_orig is not None:
            try:
                gtw_module.get_terminal_writer = gtw_orig  # type: ignore[assignment]
            except Exception:
                pass
        # Restore sys streams
        if sys_stdout_orig is not None:
            try:
                sys.stdout = sys_stdout_orig  # type: ignore[assignment]
            except Exception:
                pass
        if sys_stderr_orig is not None:
            try:
                sys.stderr = sys_stderr_orig  # type: ignore[assignment]
            except Exception:
                pass
        # Tear down FD-level tee
        try:
            _teardown_fd_tee(state)
        except Exception:
            pass
    finally:
        try:
            if log_fh is not None:
                log_fh.flush()
                log_fh.close()
        except Exception:
            pass


def _install_fd_tee(state: dict) -> None:
    """Duplicate writes to FDs 1/2 into the log file using a background pump.

    This captures output paths that bypass Python (e.g., low-level writes during collection).

    Stores resources in state for later teardown.
    """
    log_fh = state["log_fh"]

    # Save originals
    orig_out = _os.dup(1)
    orig_err = _os.dup(2)

    # Create pipes
    out_r, out_w = _os.pipe()
    err_r, err_w = _os.pipe()

    # Redirect stdout/stderr to the pipe writers
    _os.dup2(out_w, 1)
    _os.dup2(err_w, 2)

    # Close the write ends in our process (the FDs 1/2 now point to them)
    _os.close(out_w)
    _os.close(err_w)

    # Pump threads: read from the pipes; write to original FD and log
    def _pump(src_fd: int, dest_fd: int) -> None:
        while True:
            try:
                chunk = _os.read(src_fd, 8192)
            except Exception:
                break
            if not chunk:
                break
            try:
                # mirror to original terminal
                _os.write(dest_fd, chunk)
            except Exception:
                pass
            try:
                # write to log as text
                log_fh.write(chunk.decode("utf-8", errors="replace"))
                log_fh.flush()
            except Exception:
                pass
        try:
            _os.close(src_fd)
        except Exception:
            pass

    t_out = threading.Thread(target=_pump, args=(out_r, orig_out), daemon=True)
    t_err = threading.Thread(target=_pump, args=(err_r, orig_err), daemon=True)
    t_out.start()
    t_err.start()

    state["fdtee"] = {
        "orig_out": orig_out,
        "orig_err": orig_err,
        "out_r": out_r,
        "err_r": err_r,
    }


def _teardown_fd_tee(state: dict) -> None:
    info = state.get("fdtee")
    if not info:
        return
    try:
        # Restore original FDs to stdout/stderr
        _os.dup2(info["orig_out"], 1)
        _os.dup2(info["orig_err"], 2)
    finally:
        for key in ("orig_out", "orig_err", "out_r", "err_r"):
            try:
                _os.close(info[key])
            except Exception:
                pass
