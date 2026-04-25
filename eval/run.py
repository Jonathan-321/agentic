import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
sys.path.append(str(ROOT))

from agent.loop import build_agent
from eval.generate_needle_tasks import generate_tasks as gen_needle
from eval.generate_long_horizon_tasks import generate_tasks as gen_long

RUNS_DIR = Path("runs")
REPORT_DIR = Path("report")


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_tasks():
    needle_path = BASE_DIR / "needle_tasks.jsonl"
    long_path = BASE_DIR / "long_horizon_tasks.jsonl"
    if not needle_path.exists():
        gen_needle()
    if not long_path.exists():
        gen_long()
    return needle_path, long_path


def score_task(task, output: str) -> bool:
    ttype = task["type"]
    if ttype == "needle":
        return task["answer"].strip() == (output or "").strip()
    if ttype == "long_horizon":
        try:
            expected = json.loads(task["answer"])
            got = json.loads(output)
            return expected == got
        except Exception:  # noqa: BLE001
            return False
    raise ValueError(f"unknown task type {ttype}")


def bucket_key(task):
    if task["type"] != "needle":
        return "n/a"
    length = task.get("length_words", 0)
    if length <= 800:
        return "<=800"
    if length <= 2500:
        return "<=2500"
    if length <= 5000:
        return "<=5000"
    return ">5000"


def run_eval(condition: str = "baseline", memory_mode: str = "none"):
    needle_path, long_path = ensure_tasks()
    tasks = list(load_jsonl(needle_path)) + list(load_jsonl(long_path))

    run_id = f"{condition}-{int(time.time())}"
    log_path = RUNS_DIR / f"{run_id}.jsonl"
    RUNS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    agent = build_agent(str(log_path), use_memory=(memory_mode != "none"))
    agent.memory_mode = memory_mode

    results = []
    for task in tasks:
        output, meta = agent.run_task(task, run_id=run_id)
        ok = score_task(task, output)
        results.append(
            {
                "task_id": task["id"],
                "type": task["type"],
                "bucket": bucket_key(task),
                "passed": ok,
                "steps": meta.get("steps", 0),
            }
        )

    # aggregate
    summary = defaultdict(lambda: {"total": 0, "passed": 0, "steps": []})
    for r in results:
        key = (r["type"], r["bucket"])
        summary[key]["total"] += 1
        summary[key]["passed"] += int(r["passed"])
        summary[key]["steps"].append(r["steps"])

    table_lines = []
    for key, agg in sorted(summary.items()):
        ttype, bucket = key
        total = agg["total"]
        passed = agg["passed"]
        avg_steps = mean(agg["steps"]) if agg["steps"] else 0
        table_lines.append(
            {
                "task_type": ttype,
                "bucket": bucket,
                "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
                "avg_steps": round(avg_steps, 2),
                "n": total,
            }
        )

    # write report json
    report_path = REPORT_DIR / f"results_{run_id}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"condition": condition, "results": table_lines}, f, indent=2)

    # log summary line to run log
    agent.logger.log_summary({"condition": condition, "table": table_lines})

    print("Run ID:", run_id)
    print("Log:", log_path)
    print("Report:", report_path)
    print("\nResults table:")
    for row in table_lines:
        print(
            f"{row['task_type']:<13} bucket={row['bucket']:<6} pass_rate={row['pass_rate']:>5}%  avg_steps={row['avg_steps']:<4} n={row['n']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="baseline", help="label for this run")
    parser.add_argument("--memory", choices=["none", "summary", "retrieval", "both"], default="none", help="memory mode")
    args = parser.parse_args()
    run_eval(condition=args.condition, memory_mode=args.memory)
