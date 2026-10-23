"""Loaded via PYTHONPATH by weekly.sh. pyarrow's thread pool can deadlock in its C++ destructor at process
exit on this Mac (seen in arrow::internal::ThreadPool::Shutdown), hanging an unattended run forever.
After Python's own cleanup, flush and hard-exit with the real exit code instead of running native finalizers."""
import atexit
import os
import sys

_code = [0]
_exit, _hook = sys.exit, sys.excepthook


def _record_exit(c=0):
    _code[0] = 0 if c is None else (c if isinstance(c, int) else 1)
    _exit(c)


def _record_exc(*a):
    _code[0] = 1
    _hook(*a)


sys.exit, sys.excepthook = _record_exit, _record_exc


@atexit.register  # registered first, so it runs last
def _hard_exit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code[0])
