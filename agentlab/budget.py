"""Durable pre-request quota. Requests are hard capped; token reservations estimated."""

import json
from pathlib import Path
from .storage import write_json


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path, max_requests=150, max_tokens=500000):
        if max_requests < 1 or max_tokens < 1:
            raise ValueError("Budget must be positive")
        self.path = Path(path)
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "requests": 0,
                "reserved_tokens": 0,
                "reported_tokens": 0,
                "missing_usage": 0,
                "by_role": {},
            }
        )
        self.state.pop("limits", None)
        self.state.pop("note", None)
        self.save()

    def save(self):
        write_json(
            self.path,
            {
                **self.state,
                "limits": {
                    "requests": self.max_requests,
                    "estimated_reserved_tokens": self.max_tokens,
                },
                "note": "Request cap is hard. Token reservation uses serialized UTF-8 bytes plus output cap, not a billing guarantee. Failed requests may have unknown charges.",
            },
        )

    def reserve(self, payload, max_output, role):
        # Conservative engineering estimate; tokenization/pricing is provider-specific.
        estimate = len(json.dumps(payload, ensure_ascii=False).encode()) + max_output
        if (
            self.state["requests"] >= self.max_requests
            or self.state["reserved_tokens"] + estimate > self.max_tokens
        ):
            self.save()
            raise BudgetExceeded(
                "Global quota reached before next API attempt; completed episodes retained"
            )
        self.state["requests"] += 1
        self.state["reserved_tokens"] += estimate
        self.state["by_role"].setdefault(role, {"requests": 0, "reported_tokens": 0})[
            "requests"
        ] += 1
        self.save()

    def record(self, tokens, role):
        if tokens is None:
            self.state["missing_usage"] += 1
        else:
            self.state["reported_tokens"] += tokens
            self.state["by_role"][role]["reported_tokens"] += tokens
        self.save()
