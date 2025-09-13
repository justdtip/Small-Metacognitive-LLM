from __future__ import annotations
from typing import Any, Tuple
import ast, multiprocessing as mp


class SafeExecError(Exception):
    pass


# Strictly whitelisted builtins; no I/O, imports, or reflection
SAFE_BUILTINS = {
    'abs': abs,
    'pow': pow,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'sorted': sorted,
    'map': map,
    'filter': filter,
    'all': all,
    'any': any,
    'list': list,
    'tuple': tuple,
    'dict': dict,
    'set': set,
}

_FORBIDDEN_CALLS = {
    'open', 'eval', 'exec', '__import__', 'compile', 'input', 'getattr', 'setattr', 'delattr', 'vars', 'globals', 'locals',
}

_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.With, ast.Raise, ast.Try, ast.Global, ast.Nonlocal, ast.Lambda,
)


def check_ast_safety(tree: ast.AST) -> None:
    """Traverse AST and raise SafeExecError on unsafe constructs."""
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise SafeExecError(f"unsafe AST node: {type(node).__name__}")
        # Forbid attribute access entirely to avoid escaping via __class__ etc.
        if isinstance(node, ast.Attribute):
            raise SafeExecError("attribute access is forbidden")
        # Forbid calling prohibited names
        if isinstance(node, ast.Call):
            # name calls
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise SafeExecError(f"forbidden call: {node.func.id}")
        # Assignment to __builtins__
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == '__builtins__':
                    raise SafeExecError("assignment to __builtins__ is forbidden")


def _worker_exec(code: str, inputs: Any, q: mp.Queue) -> None:
    try:
        # Restrict builtins completely; expose SAFE_BUILTINS as globals
        ns = {"__builtins__": {}}
        ns.update(SAFE_BUILTINS)
        loc = {}
        exec(code, ns, loc)
        fn = loc.get('solve') or ns.get('solve') or loc.get('f') or ns.get('f')
        if fn is None:
            raise SafeExecError("no callable 'solve' or 'f' defined")
        out = fn(*inputs) if isinstance(inputs, (list, tuple)) else fn(inputs)
        q.put((True, out))
    except Exception:
        q.put((False, None))


def execute_code(program: str, inputs: Any, timeout_s: float = 2.0) -> Any:
    """
    Parse, validate, and execute code in a subprocess and return the result or raise SafeExecError.
    """
    try:
        tree = ast.parse(program)
        check_ast_safety(tree)
        code = compile(tree, filename='<safe>', mode='exec')
    except Exception as e:
        raise SafeExecError(f"parse/validate failed: {type(e).__name__}") from e
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, inputs, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate(); p.join()
        raise SafeExecError("timeout")
    try:
        ok, out = q.get_nowait()
    except Exception as e:
        raise SafeExecError("no result") from e
    if not ok:
        raise SafeExecError("execution failed")
    return out


def validate_and_run(program: str, inputs: Any, timeout_s: float = 2.0) -> Tuple[bool, Any]:
    try:
        out1 = execute_code(program, inputs, timeout_s=timeout_s)
        out2 = execute_code(program, inputs, timeout_s=timeout_s)
        return (out1 == out2, out1 if out1 == out2 else None)
    except SafeExecError:
        return False, None


# Backward-compat wrappers ---------------------------------------------------------
def execute(program: str, inputs: Any, timeout_s: float = 2.0) -> Tuple[bool, Any]:
    try:
        out = execute_code(program, inputs, timeout_s=timeout_s)
        return True, out
    except SafeExecError:
        return False, None


def validate_task(program: str, inputs: Any, outputs: Any, timeout_s: float = 2.0) -> bool:
    ok, out = validate_and_run(program, inputs, timeout_s=timeout_s)
    return bool(ok and out == outputs)
