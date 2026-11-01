"""One process per experiment folder (macOS/Linux); prevents quota ledger races."""

from contextlib import contextmanager
from pathlib import Path
import fcntl


@contextmanager
def run_lock(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".agentlab.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another process is using this experiment directory"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
