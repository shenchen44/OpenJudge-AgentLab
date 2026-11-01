"""Deterministic coordinator with isolated coroutine workers. No router LLM calls."""

import asyncio
import time
from .accounting import RequestScope, request_scope, RunStopped
from .provider import ProviderFailure
from .storage import digest


class WorkerTimeout(RuntimeError):
    pass


async def execute_batch(
    entries, cache, metadata, runner, concurrency=1, task_timeout=300, progress=None
):
    if not 1 <= concurrency <= 8 or task_timeout <= 0:
        raise ValueError("concurrency must be 1..8; task_timeout must be positive")
    # A duplicate submission must not change the denominator or create another attempt.
    unique = list({key: (key, task, repeat) for key, task, repeat in entries}.values())
    batch_id = digest({"keys": [key for key, _, _ in unique], "metadata": metadata})
    cache.enqueue(batch_id, unique, metadata)
    queue = asyncio.Queue()
    for index, entry in enumerate(unique):
        queue.put_nowait((index, entry))
    stop = asyncio.Event()
    rows = {}
    errors = []
    active = 0
    peak = 0
    completed = 0
    start = time.perf_counter()

    async def worker():
        nonlocal active, peak, completed
        while not stop.is_set():
            try:
                index, (key, task, repeat) = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            attempt_id = None
            try:
                episode = cache.get(key)
                if episode is not None:
                    episode["cache_hit"] = True
                else:
                    attempt_id = cache.claim(key)
                    active += 1
                    peak = max(peak, active)
                    scope = RequestScope(key, task["id"], attempt_id, cache, stop)
                    try:
                        with request_scope(scope):
                            async with asyncio.timeout(task_timeout):
                                episode = await runner(task)
                    finally:
                        active -= 1
                    cache.complete(key, attempt_id, episode)
                    completed += 1
                rows[index] = {
                    "task": task,
                    "repeat": repeat,
                    "episode": episode,
                    "task_key": key,
                    "batch_id": batch_id,
                }
                if progress:
                    progress(len(rows), len(unique), task["id"], episode)
            except BaseException as exc:
                stop.set()
                if attempt_id:
                    reason = type(exc).__name__
                    state = (
                        "blocked"
                        if isinstance(exc, (ProviderFailure, TimeoutError))
                        else "interrupted"
                    )
                    cache.interrupt(key, attempt_id, reason, state)
                if isinstance(exc, TimeoutError):
                    exc = WorkerTimeout(
                        f"Task exceeded {task_timeout}s; inspect checkpoint before retry"
                    )
                errors.append(exc)
                return
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(concurrency, max(1, len(unique))))
    ]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        stop.set()
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    finally:
        elapsed = time.perf_counter() - start
        report = {
            "batch_id": batch_id,
            "submitted": len(entries),
            "unique_tasks": len(unique),
            "duplicates_dropped": len(entries) - len(unique),
            "completed_this_invocation": completed,
            "cache_hits": sum(r["episode"]["cache_hit"] for r in rows.values()),
            "requested_concurrency": concurrency,
            "peak_active_workers": peak,
            "elapsed_seconds": elapsed,
            "uncached_tasks_per_second": completed / elapsed if elapsed else None,
            "state": "paused" if errors else "completed",
            "note": "Current-invocation wall time includes scheduling and local storage. Resumed/cached runs are not full task speedup evidence.",
        }
        cache.finish_batch(batch_id, report)
    if errors:
        # Preserve the original error rather than a secondary cooperative stop.
        first = next((e for e in errors if not isinstance(e, RunStopped)), errors[0])
        raise first
    output = [rows[i] for i in sorted(rows)]
    for row in output:
        row["batch_report"] = report
    return output
