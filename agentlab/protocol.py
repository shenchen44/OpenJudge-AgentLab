"""Offline, counterfactual protocol diagnostics. Never modifies original scores."""
import copy
import hashlib
import json
from pathlib import Path
from .graders import grade_rows


def extract_for_diagnosis(raw):
    """Require one trailing JSON object; reject ambiguous braces and trailing prose."""
    if not isinstance(raw, str):
        raise ValueError("No text")
    start = raw.find("{")
    if start < 0:
        raise ValueError("No object")
    obj, end = json.JSONDecoder().raw_decode(raw[start:])
    if raw[start + end:].strip():
        raise ValueError("Trailing text or multiple objects")
    if (not isinstance(obj, dict)
        or obj.get("action") not in ("complete", "clarify", "failed")
        or not isinstance(obj.get("message"), str)
        or not isinstance(obj.get("missing_fields"), list)
        or not all(isinstance(v, str) for v in obj["missing_fields"])):
        raise ValueError("Invalid schema")
    return obj


async def diagnose(directory):
    directory = Path(directory)
    variants = {"baseline": "pilot_baseline.json", "manual": "pilot_manual.json",
                "short": "efficiency_candidate.json"}
    report = {"kind": "offline_protocol_diagnostic", "api_calls": 0,
              "note": "Counterfactual extraction of archived output, not a rerun or deployment improvement. Original strict scores remain authoritative.",
              "variants": {}}
    task_ids = None
    for label, filename in variants.items():
        source = (directory / filename).read_bytes()
        original = json.loads(source)
        ids = [r["task"]["id"] for r in original]
        if len(ids) != len(set(ids)) or (task_ids is not None and set(ids) != task_ids):
            raise ValueError("Controls must contain identical unique tasks")
        task_ids = set(ids)
        rows = copy.deepcopy(original)
        extracted = []
        # Apply the same rule to all rows; only format errors may be cleared.
        for row in rows:
            ep = row["episode"]
            try:
                output = extract_for_diagnosis(ep.get("raw_final"))
            except (ValueError, TypeError):
                continue
            if ep.get("error") == "invalid_final_json":
                ep["output"] = output
                ep["error"] = None
                extracted.append(row["task"]["id"])
        await grade_rows(rows)
        strict = sum(r["grades"]["outcome"]["score"] for r in original)
        counterfactual = sum(r["grades"]["outcome"]["score"] for r in rows)
        repaired = [r["task"]["id"] for r, old in zip(rows, original)
                    if r["grades"]["outcome"]["score"] > old["grades"]["outcome"]["score"]]
        report["variants"][label] = {
            "source": filename, "source_sha256": hashlib.sha256(source).hexdigest(),
            "n": len(rows), "strict_successes": int(strict),
            "diagnostic_successes_after_extraction": int(counterfactual),
            "format_error_count": sum(r["episode"].get("error") == "invalid_final_json" for r in original),
            "extracted_task_ids": extracted, "format_only_failure_task_ids": repaired,
            "remaining_failure_task_ids": [r["task"]["id"] for r in rows if not r["grades"]["outcome"]["score"]]}
    return report
