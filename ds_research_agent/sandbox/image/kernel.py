"""Persistent Python kernel; runs as uid 1000 under strace (untrusted side).

Reads one JSON request per line on the request pipe (fd 0 at start), runs
the cell in a persistent namespace with fds 1 and 2 pointed at per-cell
files in /out, and writes one JSON status line on the response pipe (fd 1
at start). Between cells fds 0-2 point at /dev/null, so cell code cannot
read requests from stdin. Everything this process writes is untrusted: the
bridge, not the kernel, reports reads, timing, and output files.
"""

import ast
import json
import os
import signal
import sys
import traceback
import types

REQ = os.fdopen(os.dup(0), "rb", buffering=0)
RESP = os.fdopen(os.dup(1), "wb", buffering=0)
_null = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    os.dup2(_null, fd)
del _null

# SIGINT interrupts a running cell only; between cells it is ignored, so a
# late interrupt cannot kill the kernel.
signal.signal(signal.SIGINT, signal.SIG_IGN)

# Cells run in a fresh __main__ module, so functions and classes they define
# pickle (multiprocessing, joblib), and the kernel's own globals stay out of
# the cell namespace.
_main = types.ModuleType("__main__")
_main.__dict__["__builtins__"] = __builtins__
sys.modules["__main__"] = _main
namespace = _main.__dict__


def _redirect(cell: int) -> None:
    for fd, name in ((1, "stdout"), (2, "stderr")):
        f = os.open(f"/out/{cell}.{name}", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(f, fd)
        os.close(f)


def _restore() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.close(null)


def _run(code: str, filename: str) -> None:
    """Exec the cell; if it ends in an expression, print its repr."""
    tree = ast.parse(code, filename, "exec")
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        tree.body.pop()
    exec(compile(tree, filename, "exec"), namespace)
    if isinstance(last, ast.Expr):
        value = eval(compile(ast.Expression(last.value), filename, "eval"), namespace)
        if value is not None:
            print(repr(value))


def _traceback(exc: BaseException, filename: str) -> str:
    """The traceback from the first frame in the cell onward."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != filename:
        tb = tb.tb_next
    if isinstance(exc, SyntaxError):
        tb = None
    return "".join(traceback.format_exception(type(exc), exc, tb))


while line := REQ.readline():
    req = json.loads(line)
    cell = int(req["cell"])
    filename = f"<cell {cell}>"
    status, tb = "ok", None
    _redirect(cell)
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        _run(req["code"], filename)
    except KeyboardInterrupt as e:
        status, tb = "interrupted", _traceback(e, filename)
    except BaseException as e:  # SystemExit included: the kernel keeps running
        status, tb = "error", _traceback(e, filename)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _restore()
    RESP.write(json.dumps({"cell": cell, "status": status, "traceback": tb}).encode() + b"\n")
