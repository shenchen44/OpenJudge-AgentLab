"""Budgeted transport with request-level durable ownership, including retries."""

import asyncio
import json
import os
from urllib.parse import urlparse
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)
from .accounting import current_scope, RunStopped
from .storage import EpisodeCache
from .local_config import load_local_env


class ProviderFailure(RuntimeError):
    def __init__(self, kind, status=None):
        self.kind = kind
        self.status = status
        super().__init__(f"{kind}" + (f" HTTP {status}" if status else ""))


class Provider:
    def __init__(self, model=None, budget=None, role="actor", json_output=False):
        load_local_env()
        key = os.environ.get("AGENTLAB_API_KEY")
        if not key:
            raise ValueError(
                "请在项目 .env 文件填写 AGENTLAB_API_KEY，或设置同名环境变量；尚未发送 API 请求"
            )
        if budget is None:
            raise ValueError("Live Provider requires a shared Budget")
        self.model = model or os.environ.get("AGENTLAB_MODEL", "deepseek-flash")
        self.endpoint = os.environ.get("AGENTLAB_BASE_URL", "https://api.deepseek.com")
        self.request_options = {}
        if urlparse(
            self.endpoint
        ).hostname == "api.deepseek.com" and self.model.startswith("deepseek-"):
            self.request_options = {"extra_body": {"thinking": {"type": "disabled"}}}
        if json_output:
            self.request_options["response_format"] = {"type": "json_object"}
        self.client = AsyncOpenAI(
            api_key=key, base_url=self.endpoint, timeout=45, max_retries=0
        )
        self.budget = budget
        self.role = role
        self.journal = EpisodeCache(budget.path.parent / "episodes.sqlite")
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
            "usage_missing": 0,
        }

    def add_usage(self, scope, **values):
        for k, v in values.items():
            self.usage[k] += v
            if scope is not None:
                scope.usage[k] += v

    async def chat(self, messages, tools=None, max_tokens=900):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            **self.request_options,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        scope = current_scope()
        store = (
            scope.store
            if scope is not None and scope.store is not None
            else self.journal
        )
        for attempt in range(2):
            if (
                scope is not None
                and scope.stop_event is not None
                and scope.stop_event.is_set()
            ):
                raise RunStopped(
                    "Another worker stopped the batch; no new request sent"
                )
            self.budget.reserve(kwargs, max_tokens, self.role)
            rid = store.start_request(
                scope.task_key if scope else None,
                scope.task_id if scope else self.role,
                scope.attempt_id if scope else None,
                self.role,
                self.model,
            )
            self.add_usage(scope, requests=1)
            try:
                response = await self.client.chat.completions.create(**kwargs)
            except asyncio.CancelledError:
                self.budget.record(None, self.role)
                self.add_usage(scope, usage_missing=1)
                store.end_request(rid, "unknown", error="cancelled_in_flight")
                raise
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                self.budget.record(None, self.role)
                self.add_usage(scope, usage_missing=1)
                store.end_request(
                    rid,
                    "failed",
                    status=getattr(exc, "status_code", None),
                    error=type(exc).__name__,
                )
                if attempt == 1:
                    raise ProviderFailure(type(exc).__name__) from None
                await asyncio.sleep(0.5)
                continue
            except APIStatusError as exc:
                self.budget.record(None, self.role)
                self.add_usage(scope, usage_missing=1)
                store.end_request(
                    rid, "failed", status=exc.status_code, error="http_error"
                )
                if exc.status_code < 500 or attempt == 1:
                    raise ProviderFailure("http_error", exc.status_code) from None
                await asyncio.sleep(0.5)
                continue
            except Exception as exc:
                self.budget.record(None, self.role)
                self.add_usage(scope, usage_missing=1)
                store.end_request(rid, "failed", error=type(exc).__name__)
                raise ProviderFailure(type(exc).__name__) from None
            if response.usage:
                pt, ct = response.usage.prompt_tokens, response.usage.completion_tokens
                self.add_usage(scope, prompt_tokens=pt, completion_tokens=ct)
                self.budget.record(pt + ct, self.role)
                store.end_request(rid, "completed", prompt=pt, completion=ct)
            else:
                self.add_usage(scope, usage_missing=1)
                self.budget.record(None, self.role)
                store.end_request(rid, "completed", error="usage_not_returned")
            if not response.choices:
                raise ProviderFailure("empty_choices")
            return response.choices[0].message, response.choices[0].finish_reason

    async def json(self, messages):
        msg, finish = await self.chat(messages, max_tokens=2200)
        if finish == "length":
            raise ValueError("Optimizer output truncated; candidate rejected")
        text = (msg.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result

    async def close(self):
        try:
            await self.client.close()
        finally:
            self.journal.close()
