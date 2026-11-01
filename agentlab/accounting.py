"""Context-local request ownership. No arithmetic on shared provider counters."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


class RunStopped(RuntimeError):
    pass


@dataclass
class RequestScope:
    task_key: str
    task_id: str
    attempt_id: str | None = None
    store: object | None = None
    stop_event: object | None = None
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
            "usage_missing": 0,
        }
    )


_current = ContextVar("agentlab_request_scope", default=None)


def current_scope():
    return _current.get()


@contextmanager
def request_scope(scope):
    token = _current.set(scope)
    try:
        yield scope
    finally:
        _current.reset(token)
