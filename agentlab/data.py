"""Small mechanism-oriented synthetic benchmark; no claim of real-world generalization."""

import json
from copy import deepcopy
from pathlib import Path
from .storage import write_json, digest

FAMILIES = (
    "standard",
    "vip",
    "sla",
    "boundary",
    "routing",
    "retry",
    "missing",
    "missing_customer",
    "already_done",
    "transient_read",
    "combined",
    "policy_exception",
)
TEMPLATES = {
    "pilot": ("请处理工单 {id}，按现行政策完成分诊；信息不足时明确追问。",),
    "dev": (
        "请接手工单 {id}，核实服务承诺和必要信息后完成分诊。",
        "需要为工单 {id} 确定优先级和团队，完成必要更新，缺信息先问。",
    ),
    "validation": ("请对 {id} 做工单分流。依据当前规则核验、处理并说明结果。",),
    "test": (
        "工单 {id} 待处理，请按服务规则落实分级和派单，缺失信息请说明。",
        "接到工单 {id}，请查询其情况和适用政策，完成合适的处理动作。",
    ),
}


def oracle(task):
    w = task["world"]
    ticket, customer, policy = w["ticket"], w["customer"], w["policy"]
    missing = (
        ["category"]
        if ticket["category"] is None
        else (["customer_tier"] if customer["tier"] is None else [])
    )
    if missing:
        return {
            "action": "clarify",
            "missing_fields": missing,
            "state": deepcopy(ticket),
            "writes": 0,
            "required_tools": ["get_ticket"]
            + (["get_customer"] if missing == ["customer_tier"] else []),
        }
    priority = policy["category_priority"].get(ticket["category"])
    if priority is None:
        priority = (
            "P1"
            if customer["tier"] in policy["priority_tiers"]
            or ticket["age_minutes"] > policy["sla_minutes"][customer["tier"]]
            else "P2"
        )
    target = {
        **ticket,
        "priority": priority,
        "team": policy["routing"][ticket["category"]],
        "status": "in_progress",
    }
    return {
        "action": "complete",
        "missing_fields": [],
        "state": target,
        "writes": int(target != ticket),
        "required_tools": ["get_ticket", "get_customer", "get_policy"],
    }


def build_dataset(root):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError(
            "Dataset already exists; use a new directory, do not overwrite an experiment dataset"
        )
    manifest = {
        "version": "tickets-v2.0",
        "synthetic": True,
        "splits": {},
        "limitation": "Controlled mechanisms with held-out compositions/values; not production or unseen-domain evidence",
    }
    for split, variants in (("pilot", 1), ("dev", 2), ("validation", 1), ("test", 2)):
        tasks = []
        for family in FAMILIES:
            for i in range(variants):
                seed = f"{split}:{family}:{i}"
                tid = "TK-" + digest(seed)[:10]
                variant = {"pilot": 0, "dev": 1, "validation": 3, "test": 5}[split] + i
                limit = 45 + variant * 13
                category = "billing" if variant % 2 else "technical"
                customer = {
                    "id": "C-" + digest(seed + "customer")[:10],
                    "tier": "standard",
                }
                ticket = {
                    "id": tid,
                    "customer_id": customer["id"],
                    "category": category,
                    "age_minutes": max(1, limit - 7),
                    "priority": "P3",
                    "team": None,
                    "status": "new",
                }
                policy = {
                    "sla_minutes": {"standard": limit, "vip": limit + 20},
                    "priority_tiers": ["vip"],
                    "routing": {
                        c: "TEAM-" + digest(f"{split}:{i}:{c}")[:6]
                        for c in ("billing", "technical", "outage")
                    },
                    "category_priority": {},
                    "target_status": "in_progress",
                    "instructions": "类别缺失先追问category；否则查询客户，等级缺失追问customer_tier且不写入。类别优先级覆盖优先级等级规则；否则priority_tiers中的客户或等待分钟数严格超过本等级SLA为P1，其余P2。按routing分配并设in_progress。目标字段已一致时不要重复写入。临时未提交错误最多重试一次。",
                }
                faults = {}
                if family == "vip":
                    customer["tier"] = "vip"
                    ticket["age_minutes"] = max(0, limit - 10)
                elif family == "sla":
                    ticket["age_minutes"] = limit + 1 + variant
                elif family == "boundary":
                    ticket["age_minutes"] = limit
                elif family == "routing":
                    policy["routing"][category] = (
                        "TEAM-" + digest(seed + "newroute")[:6]
                    )
                elif family == "retry":
                    faults = {"update_ticket": 1}
                elif family == "missing":
                    ticket["category"] = None
                elif family == "missing_customer":
                    customer["tier"] = None
                elif family == "transient_read":
                    faults = {"get_ticket": 1}
                elif family == "combined":
                    customer["tier"] = (
                        "vip" if split in ("pilot", "dev") else "standard"
                    )
                    ticket["age_minutes"] = limit + 30
                    faults = (
                        {"update_ticket": 1}
                        if split in ("pilot", "dev")
                        else {"get_policy": 1}
                    )
                elif family == "policy_exception":
                    ticket["category"] = "outage"
                    policy["category_priority"] = {"outage": "P1"}
                    if split in ("validation", "test"):
                        customer["tier"] = "vip"
                        policy["category_priority"] = {"outage": "P2"}
                world = {"ticket": ticket, "customer": customer, "policy": policy}
                task = {
                    "id": tid,
                    "split": split,
                    "family": family,
                    "template_group": f"{split}:{i}",
                    "query": TEMPLATES[split][i].format(id=tid),
                    "world": world,
                    "faults": faults,
                }
                if family == "already_done":
                    task["world"]["ticket"] = oracle(task)["state"]
                task["expected"] = oracle(task)
                tasks.append(task)
        write_json(root / f"{split}.json", tasks)
        manifest["splits"][split] = {"count": len(tasks), "sha256": digest(tasks)}
    write_json(root / "manifest.json", manifest)
    return manifest


def load_split(root, split):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != "tickets-v2.0":
        raise ValueError("v2 commands require v2 data; generate into data_v2")
    if split not in manifest["splits"]:
        raise ValueError("Unknown split")
    rows = json.loads((root / f"{split}.json").read_text())
    if digest(rows) != manifest["splits"][split]["sha256"]:
        raise ValueError("Dataset hash mismatch")
    if not rows or any(t["split"] != split for t in rows):
        raise ValueError("Empty or contaminated split")
    return rows


def stratified(tasks, count):
    if count is None or count >= len(tasks):
        return tasks
    if count < 1:
        raise ValueError("count must be positive")
    groups = {f: [t for t in tasks if t["family"] == f] for f in FAMILIES}
    output = []
    while len(output) < count and any(groups.values()):
        for family in FAMILIES:
            if groups[family] and len(output) < count:
                output.append(groups[family].pop(0))
    return output
