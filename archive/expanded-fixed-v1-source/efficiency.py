"""One fixed short-prompt candidate. Reuse paid controls and the original quota."""

import json
from pathlib import Path
from . import __version__
from .data import load_split
from .experiment import evaluate, summarize, identity, bound_config, progress
from .prompts import BASELINE, MANUAL
from .storage import EpisodeCache, digest, write_json

SHORT_PROMPT = """按现行政策分诊工单。先查工单，缺类别立即追问；否则查客户和政策，缺等级立即追问。复用已知信息，目标已满足则不写入、不复查。仅临时未提交错误重试一次。
只输出JSON：{"action":"complete|clarify|failed","message":"一句话结果或追问","missing_fields":[]}。缺失字段仅用category、customer_tier。"""


def prepare(
    data_dir,
    out,
    pilot_dir,
    max_requests=150,
    max_tokens=1000000,
    max_steps=8,
    concurrency=1,
    task_timeout=300,
):
    out = Path(out)
    pilot_dir = Path(pilot_dir)
    if out.resolve() != pilot_dir.resolve():
        raise ValueError(
            "Efficiency check must reuse the original --pilot directory as --out"
        )
    pilot = json.loads((out / "pilot.json").read_text())
    cfg = pilot["config"]
    if (
        cfg["engine"] != __version__
        or cfg["mode"] != "live"
        or cfg["baseline"] != digest(BASELINE)
        or cfg["manual"] != digest(MANUAL)
    ):
        raise ValueError(
            "Pilot engine/prompts do not match; do not rerun paid controls"
        )
    if (
        cfg["max_steps"] != max_steps
        or cfg["concurrency"] != concurrency
        or cfg["task_timeout"] != task_timeout
    ):
        raise ValueError(
            "Use identical steps, concurrency and timeout as the original pilot"
        )
    manifest = digest(json.loads((Path(data_dir) / "manifest.json").read_text()))
    tasks = load_split(data_dir, "pilot")
    if manifest != cfg["data_manifest"] or digest(tasks) != cfg["pilot_hash"]:
        raise ValueError("Pilot data changed")
    controls = {}
    for label in ("baseline", "manual"):
        rows = json.loads((out / f"pilot_{label}.json").read_text())
        if len(rows) != len(tasks) or {r["task"]["id"] for r in rows} != {
            t["id"] for t in tasks
        }:
            raise ValueError("Incomplete control traces")
        expected = {t["id"]: digest(t) for t in tasks}
        if any(
            digest(r["task"]) != expected[r["task"]["id"]]
            or r["episode"]["mode"] != "live"
            for r in rows
        ):
            raise ValueError("Control traces do not match the pilot dataset")
        controls[label] = rows
    ledger = json.loads((out / "budget.json").read_text())
    if (
        max_requests > ledger["limits"]["requests"]
        or max_tokens > ledger["limits"]["estimated_reserved_tokens"]
    ):
        raise ValueError(
            "This command cannot increase the existing quota; keep the original limits"
        )
    config = {
        "engine": __version__,
        "pilot_sha256": digest(pilot),
        "candidate_sha256": digest(SHORT_PROMPT),
        "control_sha256": {k: digest(v) for k, v in controls.items()},
        "criterion": "candidate preserves all manual successes and reduces baseline total tokens by at least 5%; exploratory only",
    }
    return (
        tasks,
        controls,
        config,
        {
            "candidate": SHORT_PROMPT,
            "candidate_characters": len(SHORT_PROMPT),
            "manual_characters": len(MANUAL),
            "new_episodes": len(tasks),
            "reused_control_episodes": sum(map(len, controls.values())),
            "requests_already_used": ledger["requests"],
            "request_cap": max_requests,
            "requests_remaining": max(0, max_requests - ledger["requests"]),
            "estimated_token_reservation_remaining": max(
                0, max_tokens - ledger["reserved_tokens"]
            ),
            "note": "Only one fixed candidate; no optimizer model, no test split, no automatic quota increase.",
        },
    )


async def efficiency_check(
    data_dir, out, pilot_dir, provider, max_steps=8, concurrency=1, task_timeout=300
):
    tasks, controls, config, plan = prepare(
        data_dir,
        out,
        pilot_dir,
        provider.budget.max_requests,
        provider.budget.max_tokens,
        max_steps,
        concurrency,
        task_timeout,
    )
    out = Path(out)
    pilot = json.loads((out / "pilot.json").read_text())
    if identity(provider) != pilot["config"]["provider"]:
        raise ValueError("Use the same model, endpoint and thinking mode as the pilot")
    bound_config(out, config, "efficiency_config.json")
    result_path = out / "efficiency.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    write_json(
        out / "efficiency_prompt.json",
        {"prompt": SHORT_PROMPT, "sha256": digest(SHORT_PROMPT)},
    )
    cache = EpisodeCache(out / "episodes.sqlite")
    try:
        rows = await evaluate(
            tasks,
            SHORT_PROMPT,
            "live",
            cache,
            provider,
            max_steps=max_steps,
            concurrency=concurrency,
            task_timeout=task_timeout,
            progress=progress,
        )
        write_json(out / "efficiency_candidate.json", rows)
    finally:
        cache.close()
    metrics = {k: summarize(v) for k, v in controls.items()}
    metrics["short"] = summarize(rows)
    new = {r["task"]["id"]: r for r in rows}
    regressions = {
        k: [
            r["task"]["id"]
            for r in v
            if r["grades"]["outcome"]["score"]
            > new[r["task"]["id"]]["grades"]["outcome"]["score"]
        ]
        for k, v in controls.items()
    }
    deltas = {}
    for label in ("baseline", "manual"):
        before = metrics[label]
        after = metrics["short"]
        deltas[label] = {
            "success_delta_pp": 100 * (after["success_rate"] - before["success_rate"]),
            "token_reduction_pct": 100
            * (1 - after["episode_tokens"] / before["episode_tokens"])
            if before["episode_tokens"]
            else None,
            "request_reduction": sum(
                r["episode"]["usage"]["requests"] for r in controls[label]
            )
            - sum(r["episode"]["usage"]["requests"] for r in rows),
        }
    unknown = any(m["usage_missing"] for m in metrics.values())
    if regressions["manual"]:
        decision = "correctness_regression_stop"
    elif unknown:
        decision = "usage_unknown_stop"
    elif (deltas["baseline"]["token_reduction_pct"] or 0) >= 5:
        decision = "promising_efficiency_needs_independent_validation"
    else:
        decision = "insufficient_savings_stop"
    result = {
        "kind": "efficiency_pilot",
        "config": config,
        "plan": plan,
        "metrics": metrics,
        "comparison": deltas,
        "regressed_tasks": regressions,
        "decision": decision,
        "budget": provider.budget.state,
        "note": "Hand-designed candidate evaluated on an already inspected pilot; NOT automatic tuning or independent final-test evidence.",
    }
    write_json(result_path, result)
    return result
