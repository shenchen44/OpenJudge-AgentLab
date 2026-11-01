import argparse
import asyncio
import json
import os
from pathlib import Path
from .data import build_dataset, load_split, stratified
from .experiment import pilot, optimize, finalize, evaluate, summarize, progress
from .prompts import BASELINE, MANUAL
from .provider import Provider, ProviderFailure
from .budget import Budget, BudgetExceeded
from .report import render_report
from .storage import EpisodeCache, write_json
from .locking import run_lock
from .coordinator import WorkerTimeout
from .accounting import RunStopped


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Must be >=1")
    return n


def plan(args):
    m = json.loads((Path(args.data) / "manifest.json").read_text())
    counts = {k: v["count"] for k, v in m["splits"].items()}
    episodes = {
        "pilot": 2 * counts["pilot"],
        "search": (
            counts["validation"] + args.rounds * (counts["dev"] + counts["validation"])
        )
        * args.repeats,
        "finalize": 3 * counts["test"] * args.repeats,
    }
    return {
        "dataset": m["version"],
        "counts": counts,
        "runtime": {"concurrency": args.concurrency, "task_timeout": args.task_timeout},
        "episodes_before_cache": episodes,
        "max_request_attempts_before_retries": {
            k: v * args.max_steps + (args.rounds if k == "search" else 0)
            for k, v in episodes.items()
        },
        "note": "No API calls. Real requests depend on tool batching/cache/stopping. Each phase still obeys its durable hard request cap; token budget is an estimate, not a currency cap.",
    }


async def execute(args):
    data = Path(args.data)
    out = Path(args.out)
    if args.command == "protocol-diagnose":
        from .protocol import diagnose
        result = await diagnose(out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command in ("tasks", "cancel", "retry"):
        if not (out / "episodes.sqlite").exists():
            raise ValueError("No execution store in this output directory")
        store = EpisodeCache(out / "episodes.sqlite")
        try:
            if args.command == "cancel":
                store.cancel(args.task_key)
            if args.command == "retry":
                store.retry(args.task_key)
            print(json.dumps(store.tasks(), ensure_ascii=False, indent=2))
        finally:
            store.close()
        return
    if args.command in ("efficiency", "efficiency-plan"):
        from .efficiency import prepare

        if not args.pilot:
            raise ValueError("Specify --pilot and identical --out")
        _, _, _, efficiency_plan = prepare(
            data,
            out,
            args.pilot,
            args.max_api_calls,
            args.max_reserved_tokens,
            args.max_steps,
            args.concurrency,
            args.task_timeout,
        )
        if args.command == "efficiency-plan":
            print(json.dumps(efficiency_plan, ensure_ascii=False, indent=2))
            return
    if args.command == "generate":
        print(json.dumps(build_dataset(data), ensure_ascii=False, indent=2))
        return
    if args.command == "plan":
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return
    if args.command == "report":
        print(render_report(out))
        return
    if args.command == "demo" and not (data / "manifest.json").exists():
        build_dataset(data)
    if args.command == "calibrate":
        from .calibration import calibrate, fixture_rows

        labels = Path(args.labels) if args.labels else out / "fixtures.json"
        if not args.labels:
            write_json(labels, await fixture_rows(data))
        result = await calibrate(labels, out / "calibration.json")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "cases"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    mode = "demo" if args.command == "demo" else args.mode
    clients = []
    if mode == "live":
        # Refuse incomplete/gated runs before any provider call.
        if args.command == "optimize" and not args.pilot:
            raise ValueError("First run pilot; optimize requires --pilot")
        if args.command == "evaluate" and args.limit is None:
            raise ValueError(
                "Live evaluate requires --limit to avoid accidental bulk spending"
            )
        budget = Budget(
            out / "budget.json", args.max_api_calls, args.max_reserved_tokens
        )
        actor = Provider(budget=budget, json_output=args.json_output)
        clients.append(actor)
    else:
        actor = None
    try:
        if args.command == "efficiency":
            if mode != "live":
                raise ValueError("Efficiency check uses a completed live pilot")
            from .efficiency import efficiency_check

            result = await efficiency_check(
                data,
                out,
                args.pilot,
                actor,
                args.max_steps,
                args.concurrency,
                args.task_timeout,
            )
        elif args.command == "pilot":
            result = await pilot(
                data,
                out,
                mode,
                actor,
                args.max_steps,
                args.concurrency,
                args.task_timeout,
            )
        elif args.command in ("optimize", "demo"):
            optimizer = (
                Provider(
                    os.environ.get("AGENTLAB_OPTIMIZER_MODEL"),
                    actor.budget,
                    "optimizer",
                    json_output=args.json_output,
                )
                if actor
                else None
            )
            if optimizer:
                clients.append(optimizer)
            result = await optimize(
                data,
                out,
                mode,
                actor,
                optimizer,
                args.rounds,
                args.repeats,
                args.max_steps,
                args.pilot,
                args.concurrency,
                args.task_timeout,
            )
            if args.command == "demo":
                result = await finalize(
                    data,
                    out,
                    mode,
                    actor,
                    args.max_steps,
                    args.repeats,
                    args.concurrency,
                    args.task_timeout,
                )
                render_report(out)
        elif args.command == "finalize":
            result = await finalize(
                data,
                out,
                mode,
                actor,
                args.max_steps,
                args.repeats,
                args.concurrency,
                args.task_timeout,
            )
            render_report(out)
        elif args.command == "evaluate":
            tasks = stratified(load_split(data, args.split), args.limit)
            cache = EpisodeCache(out / "episodes.sqlite")
            try:
                rows = await evaluate(
                    tasks,
                    MANUAL if args.variant == "manual" else BASELINE,
                    mode,
                    cache,
                    actor,
                    args.repeats,
                    args.max_steps,
                    progress,
                    args.concurrency,
                    args.task_timeout,
                )
                result = summarize(rows)
                write_json(out / "episodes.json", rows)
                write_json(out / "metrics.json", result)
            finally:
                cache.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        write_json(out / "status.json", {"status": "completed", "phase": args.command})
    except (BudgetExceeded, ProviderFailure, WorkerTimeout, RunStopped) as exc:
        write_json(
            out / "status.json",
            {
                "status": "paused",
                "phase": args.command,
                "reason": str(exc),
                "http_status": getattr(exc, "status", None),
            },
        )
        raise
    except asyncio.CancelledError:
        write_json(
            out / "status.json",
            {
                "status": "interrupted",
                "phase": args.command,
                "reason": "cancelled; inspect task attempts and checkpoints",
            },
        )
        raise
    finally:
        for client in clients:
            await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="OpenJudge-AgentLab v3: persistent isolated workers and request accounting"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "protocol-diagnose",
        "efficiency",
        "efficiency-plan",
        "generate",
        "plan",
        "demo",
        "pilot",
        "optimize",
        "finalize",
        "evaluate",
        "calibrate",
        "report",
        "tasks",
        "cancel",
        "retry",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--data", default="data_v2")
        sub.add_argument("--out", default=f"runs/v3-{name}")
        sub.add_argument("--json-output", action="store_true",
                         help="Opt in to JSON object response format for every provider request; changes experiment identity")
        sub.add_argument("--mode", choices=("live", "demo"), default="live")
        sub.add_argument("--rounds", type=positive, default=1)
        sub.add_argument("--repeats", type=positive, default=1)
        sub.add_argument("--max-steps", type=positive, default=8)
        sub.add_argument("--concurrency", type=positive, choices=range(1, 9), default=1)
        sub.add_argument("--task-timeout", type=positive, default=300)
        if name in ("cancel", "retry"):
            sub.add_argument("--task-key", required=True)
        sub.add_argument(
            "--max-api-calls",
            type=positive,
            default=150
            if name in ("pilot", "evaluate", "efficiency", "efficiency-plan")
            else 800,
        )
        sub.add_argument(
            "--max-reserved-tokens",
            type=positive,
            default=1000000
            if name in ("pilot", "evaluate", "efficiency", "efficiency-plan")
            else 4000000,
        )
        sub.add_argument("--pilot")
        if name == "evaluate":
            sub.add_argument("--limit", type=positive)
            sub.add_argument(
                "--split", choices=("pilot", "dev", "validation"), default="pilot"
            )
            sub.add_argument(
                "--variant", choices=("baseline", "manual"), default="baseline"
            )
        if name == "calibrate":
            sub.add_argument("--labels")
    args = parser.parse_args()
    try:
        if args.command in (
            "pilot",
            "optimize",
            "finalize",
            "evaluate",
            "demo",
            "cancel",
            "retry",
            "efficiency",
        ):
            with run_lock(args.out):
                asyncio.run(execute(args))
        else:
            asyncio.run(execute(args))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"agentlab: {exc}\n")


if __name__ == "__main__":
    main()
