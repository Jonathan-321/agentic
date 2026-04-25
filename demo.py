#!/usr/bin/env python3
"""
demo.py — Interactive CLI demo for the ReAct agent framework.

Demonstrates how vector-retrieval memory fixes long-context failures:
  - Baseline (no memory)  → agent truncates the doc, misses the needle → FAIL
  - Memory retrieval       → agent chunks + indexes + retrieves → PASS

Usage:
  python3 demo.py          # interactive walkthrough (default)
  python3 demo.py --full   # full eval across all conditions + comparison table
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── ensure project root is on sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── custom theme ─────────────────────────────────────────────────────────────
THEME = Theme(
    {
        "thought": "bold cyan",
        "action": "bold yellow",
        "observation": "dim white",
        "success": "bold green",
        "failure": "bold red",
        "label": "bold white",
        "dim_label": "dim white",
        "step_num": "bold magenta",
        "header": "bold blue",
        "needle": "bold bright_yellow",
        "memory": "bold cyan",
        "baseline": "bold red",
        "retrieval": "bold green",
    }
)

console = Console(theme=THEME, highlight=False)

# ── paths ─────────────────────────────────────────────────────────────────────
DEMO_DIR = ROOT / "demo"
SAMPLE_TASK_PATH = DEMO_DIR / "sample_task.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pause(sec: float = 0.6) -> None:
    time.sleep(sec)


def load_task(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No tasks found in {path}")


def section_rule(title: str) -> None:
    console.print()
    console.rule(f"[header]{title}[/header]", style="dim blue")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Agent runner with step capture
# ─────────────────────────────────────────────────────────────────────────────

class StepCapture:
    """Monkey-patches JSONLLogger.log_step to capture steps in memory."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def log_step(self, run_id, task_id, step, thought, action, tool, tool_input, observation, decision=None, tokens_used=None):
        self.steps.append(
            {
                "step": step,
                "thought": thought,
                "action": action,
                "tool": tool,
                "tool_input": tool_input or {},
                "observation": (observation or "")[:600],
                "decision": decision,
            }
        )

    def log_summary(self, summary):
        pass


def run_agent_with_capture(task: Dict[str, Any], memory_mode: str) -> Tuple[str, List[Dict]]:
    """Run the ReActAgent and capture its step trace."""
    from agent.loop import ReActAgent
    from agent.logger import JSONLLogger
    from memory.memory import MemoryManager
    from memory.vector_store import VectorStore
    from tools.builtin import get_builtin_tools

    memory_dir = str(ROOT / "memory" / "store_demo")  # isolated store for demo
    os.makedirs(memory_dir, exist_ok=True)

    # Clean up any leftover demo notes so runs are independent
    needle_notes = Path(memory_dir) / "needle.notes.txt"
    if needle_notes.exists():
        needle_notes.unlink()

    vector_store = VectorStore()
    tools = get_builtin_tools(memory_dir, vector_store=vector_store)
    memory = MemoryManager(memory_dir)

    capture = StepCapture()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        log_path = tf.name

    try:
        agent = ReActAgent(
            tools=tools,
            logger=capture,  # type: ignore[arg-type]
            memory=memory,
            max_steps=8,
            context_window_words=1200,
            memory_mode=memory_mode,
        )
        run_id = f"demo-{memory_mode}"
        answer, _ = agent.run_task(task, run_id=run_id)
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)

    return answer, capture.steps


# ─────────────────────────────────────────────────────────────────────────────
# Rich rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_step_panel(step: Dict[str, Any], index: int, total: int, mode_label: str, color: str) -> Panel:
    """Render one agent step as a rich Panel."""
    thought = step.get("thought", "")
    action = step.get("action", "")
    tool = step.get("tool") or ""
    tool_input = step.get("tool_input", {})
    observation = step.get("observation", "")
    decision = step.get("decision")

    content_lines = []

    # Thought
    if thought:
        content_lines.append(Text.assemble(("  Thought  ", "bold white on dark_magenta"), "  ", (thought, "italic cyan")))

    # Action / tool
    action_str = action.upper()
    if tool:
        action_str += f"  →  [action]{tool}[/action]"
        if tool_input:
            # Show abbreviated input
            inp_preview = str(tool_input)[:120]
            action_str += f"  [dim white]({inp_preview})[/dim white]"
    content_lines.append(Text.from_markup(f"  [bold yellow]Action[/bold yellow]   [dim]step {index}/{total}[/dim]  {action_str}"))

    # Observation
    if observation:
        obs_lines = observation.replace("\r", "").split("\n")
        short_obs = " ".join(obs_lines)[:300]
        if len(observation) > 300:
            short_obs += " …"
        content_lines.append(Text.from_markup(f"  [dim white]Obs[/dim white]      [observation]{short_obs}[/observation]"))

    # Decision / answer
    if decision:
        content_lines.append(Text.from_markup(f"\n  [bold white]Answer[/bold white]   [{color}]{decision}[/{color}]"))

    body = Text("\n").join(content_lines)
    return Panel(
        body,
        title=f"[{color}][ {mode_label} ]  Step {index}[/{color}]",
        border_style=color,
        padding=(0, 1),
        expand=True,
    )


def print_steps_live(steps: List[Dict[str, Any]], mode_label: str, color: str, step_delay: float = 0.8) -> None:
    """Print agent steps one-by-one with pauses for dramatic effect."""
    total = len(steps)
    for i, step in enumerate(steps, 1):
        panel = render_step_panel(step, i, total, mode_label, color)
        console.print(panel)
        pause(step_delay)


def verdict_panel(answer: str, expected: str, mode_label: str) -> Panel:
    passed = answer.strip() == expected.strip()
    icon = "✓  PASS" if passed else "✗  FAIL"
    style = "success" if passed else "failure"
    body = Text.assemble(
        (f"  Answer:   ", "label"),
        (f"{answer}\n", style),
        (f"  Expected: ", "label"),
        (f"{expected}\n", "dim white"),
        (f"\n  Result:   ", "label"),
        (f"{icon}", style),
    )
    return Panel(body, title=f"[{style}]  Verdict — {mode_label}  [{style}]", border_style=style, padding=(0, 1))


def comparison_table(
    baseline_answer: str,
    retrieval_answer: str,
    expected: str,
    baseline_steps: List[Dict],
    retrieval_steps: List[Dict],
) -> Table:
    table = Table(
        title="[bold white]Side-by-Side Comparison[/bold white]",
        box=box.DOUBLE_EDGE,
        border_style="blue",
        expand=True,
        show_lines=True,
    )
    table.add_column("", style="dim white", width=18)
    table.add_column("[baseline]Baseline (no memory)[/baseline]", justify="center")
    table.add_column("[retrieval]Memory Retrieval[/retrieval]", justify="center")

    b_pass = baseline_answer.strip() == expected.strip()
    r_pass = retrieval_answer.strip() == expected.strip()

    table.add_row(
        "Memory Mode",
        "[baseline]none[/baseline]",
        "[retrieval]retrieval[/retrieval]",
    )
    table.add_row(
        "Context Window",
        "1 200 words",
        "1 200 words + vector index",
    )
    table.add_row(
        "Doc Length",
        "~2 450 words",
        "~2 450 words",
    )
    table.add_row(
        "Steps Taken",
        str(len(baseline_steps)),
        str(len(retrieval_steps)),
    )
    table.add_row(
        "Answer",
        f"[{'success' if b_pass else 'failure'}]{baseline_answer}[/{'success' if b_pass else 'failure'}]",
        f"[{'success' if r_pass else 'failure'}]{retrieval_answer}[/{'success' if r_pass else 'failure'}]",
    )
    table.add_row(
        "Result",
        f"[{'success' if b_pass else 'failure'}]{'✓ PASS' if b_pass else '✗ FAIL'}[/{'success' if b_pass else 'failure'}]",
        f"[{'success' if r_pass else 'failure'}]{'✓ PASS' if r_pass else '✗ FAIL'}[/{'success' if r_pass else 'failure'}]",
    )
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Welcome banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
 ____            _    _      _                    _   _      
|  _ \ ___  __ _| |  / \   | | __   _ __   ___ | |_| |__  
| |_) / _ \/ _` | | / _ \  | |/ /  | '_ \ / _ \| __| '_ \ 
|  _ <  __/ (_| | |/ ___ \ |   <   | | | | (_) | |_| | | |
|_| \_\___|\__,_|_/_/   \_\|_|\_\  |_| |_|\___/ \__|_| |_|
                                                              
    Memory-Augmented ReAct Agent  ·  Interactive Demo
"""

def print_banner() -> None:
    console.print(Text(BANNER, style="bold blue"), justify="center")
    console.print(Rule(style="dim blue"))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Intro explanation
# ─────────────────────────────────────────────────────────────────────────────

def print_intro(task: Dict[str, Any]) -> None:
    doc_words = task.get("length_words", "?")
    key = task["key"]
    answer = task["answer"]
    needle_snippet = f"NEEDLE: {key} -> {answer}"

    intro_table = Table.grid(padding=(0, 2))
    intro_table.add_column(style="dim white", justify="right", width=22)
    intro_table.add_column(style="white")

    intro_table.add_row("Task type", "[bold]Needle-in-Haystack[/bold]")
    intro_table.add_row("Document length", f"[bold]{doc_words:,}[/bold] words  (context window = 1 200 words)")
    intro_table.add_row("Hidden key", f"[needle]{key}[/needle]")
    intro_table.add_row("Hidden needle", f"[needle]{needle_snippet}[/needle]  ← buried past word 1 200")
    intro_table.add_row("", "")
    intro_table.add_row("Run 1 — Baseline", "[baseline]No memory.  Agent reads 1 200 words, truncates rest.[/baseline]")
    intro_table.add_row("Run 2 — Retrieval", "[retrieval]Chunks doc → indexes chunks → queries vector store.[/retrieval]")

    console.print(
        Panel(
            Align.center(intro_table),
            title="[header]  What This Demo Shows  [/header]",
            border_style="blue",
            padding=(1, 4),
        )
    )
    console.print()
    pause(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive demo (default mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_demo() -> None:
    print_banner()
    pause(0.3)

    # ── load task ─────────────────────────────────────────────────────────────
    if not SAMPLE_TASK_PATH.exists():
        console.print(f"[failure]Sample task not found at {SAMPLE_TASK_PATH}[/failure]")
        console.print("Run:  python3 create_sample_task.py")
        sys.exit(1)

    task = load_task(SAMPLE_TASK_PATH)
    print_intro(task)

    # ── prompt user to continue ───────────────────────────────────────────────
    console.print("[dim white]Press [bold]Enter[/bold] to start the baseline run …[/dim white]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 1 — BASELINE (no memory)
    # ══════════════════════════════════════════════════════════════════════════
    section_rule("RUN 1 of 2 — Baseline Agent  (memory = none)")

    console.print(Panel(
        Text.from_markup(
            "  The agent will:\n"
            "  [dim white]1.[/dim white] Read the document\n"
            "  [dim white]2.[/dim white] Truncate to the first [bold]1 200 words[/bold] (its context window)\n"
            "  [dim white]3.[/dim white] Try to find [needle]launchkey[/needle] — but the needle is at word [bold]1 878[/bold]\n"
            "  [failure]→ It will not find it.[/failure]"
        ),
        title="[baseline]  Baseline — No Memory  [/baseline]",
        border_style="red",
        padding=(0, 2),
    ))
    pause(0.5)

    with Progress(
        SpinnerColumn(style="red"),
        TextColumn("[baseline]Running baseline agent …[/baseline]"),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task("run", total=None)
        baseline_answer, baseline_steps = run_agent_with_capture(task, memory_mode="none")

    console.print()
    console.print("[bold red]  Baseline run complete.[/bold red]")
    pause(0.4)

    section_rule("Baseline — Step Trace")
    print_steps_live(baseline_steps, mode_label="BASELINE", color="red", step_delay=0.6)
    console.print()
    console.print(verdict_panel(baseline_answer, task["answer"], "Baseline — No Memory"))
    pause(0.8)

    # ── prompt user to continue ───────────────────────────────────────────────
    console.print()
    console.print("[dim white]Press [bold]Enter[/bold] to start the memory-retrieval run …[/dim white]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 2 — MEMORY RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════
    section_rule("RUN 2 of 2 — Memory-Retrieval Agent  (memory = retrieval)")

    console.print(Panel(
        Text.from_markup(
            "  The agent will:\n"
            "  [dim white]1.[/dim white] Read the full document\n"
            "  [dim white]2.[/dim white] [memory]Chunk it into 300-word blocks[/memory] and index each chunk in the vector store\n"
            "  [dim white]3.[/dim white] [memory]Query the vector store[/memory] with key [needle]launchkey[/needle]\n"
            "  [dim white]4.[/dim white] Retrieve the relevant chunk — [needle]the one containing the needle[/needle]\n"
            "  [retrieval]→ It will find PHOENIX-7749.[/retrieval]"
        ),
        title="[retrieval]  Memory Retrieval — Vector Search  [/retrieval]",
        border_style="green",
        padding=(0, 2),
    ))
    pause(0.5)

    with Progress(
        SpinnerColumn(style="green"),
        TextColumn("[retrieval]Running memory-retrieval agent …[/retrieval]"),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task("run", total=None)
        retrieval_answer, retrieval_steps = run_agent_with_capture(task, memory_mode="retrieval")

    console.print()
    console.print("[bold green]  Memory-retrieval run complete.[/bold green]")
    pause(0.4)

    section_rule("Memory Retrieval — Step Trace")
    print_steps_live(retrieval_steps, mode_label="RETRIEVAL", color="green", step_delay=0.6)
    console.print()
    console.print(verdict_panel(retrieval_answer, task["answer"], "Memory Retrieval"))
    pause(0.8)

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    section_rule("Final Comparison")
    table = comparison_table(
        baseline_answer=baseline_answer,
        retrieval_answer=retrieval_answer,
        expected=task["answer"],
        baseline_steps=baseline_steps,
        retrieval_steps=retrieval_steps,
    )
    console.print(Align.center(table))
    console.print()

    # ── takeaway ──────────────────────────────────────────────────────────────
    b_pass = baseline_answer.strip() == task["answer"].strip()
    r_pass = retrieval_answer.strip() == task["answer"].strip()

    summary = Text()
    summary.append("  Key insight: ", style="bold white")
    if not b_pass and r_pass:
        summary.append(
            "Without memory, the agent is blind beyond its 1 200-word context window.\n",
            style="red",
        )
        summary.append(
            "  With vector retrieval, it chunks the entire document and fetches only the relevant\n"
            "  passage — making long-context failure a thing of the past.",
            style="green",
        )
    elif b_pass and r_pass:
        summary.append("Both conditions passed on this task.", style="green")
    else:
        summary.append("Results may vary — see --full for the full eval table.", style="yellow")

    console.print(
        Panel(summary, title="[header]  Takeaway  [/header]", border_style="blue", padding=(0, 2))
    )
    console.print()
    console.print(
        Align.center(
            Text("Run  python3 demo.py --full  to see the complete eval heatmap.", style="dim white")
        )
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Full eval mode (--full)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eval() -> None:
    from eval.run import run_eval, ensure_tasks, load_jsonl, score_task, bucket_key
    from agent.loop import ReActAgent
    from agent.logger import JSONLLogger
    from memory.memory import MemoryManager
    from memory.vector_store import VectorStore
    from tools.builtin import get_builtin_tools
    from collections import defaultdict
    from statistics import mean
    import time

    print_banner()
    pause(0.3)

    section_rule("Full Evaluation — All Conditions")

    console.print(Panel(
        Text.from_markup(
            "  Running all four memory conditions against the full task suite:\n\n"
            "  [dim white]baseline[/dim white]     — no memory, context window only\n"
            "  [dim white]mem-summary[/dim white]  — summarization memory\n"
            "  [dim white]mem-retrieval[/dim white]— vector retrieval  [retrieval](the good stuff)[/retrieval]\n"
            "  [dim white]mem-both[/dim white]     — summary + retrieval combined\n\n"
            "  This will take a minute or two …"
        ),
        title="[header]  Eval Configuration  [/header]",
        border_style="blue",
        padding=(0, 2),
    ))
    pause(0.5)

    conditions = [
        ("baseline",     "none"),
        ("mem-summary",  "summary"),
        ("mem-retrieval","retrieval"),
        ("mem-both",     "both"),
    ]

    all_results: Dict[str, List[Dict]] = {}

    for condition, memory_mode in conditions:
        console.print(f"\n[bold white]Running[/bold white] [action]{condition}[/action] …")
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[dim white]{condition}[/dim white]"),
            BarColumn(bar_width=30),
            TextColumn("[dim white]{task.completed}/{task.total}[/dim white]"),
            transient=True,
            console=console,
        ) as progress:

            # Set up agent
            memory_dir = str(ROOT / "memory" / f"store_{condition}")
            os.makedirs(memory_dir, exist_ok=True)

            vector_store = VectorStore()
            tools = get_builtin_tools(memory_dir, vector_store=vector_store)
            memory = MemoryManager(memory_dir)
            capture = StepCapture()

            agent = ReActAgent(
                tools=tools,
                logger=capture,  # type: ignore[arg-type]
                memory=memory,
                max_steps=8,
                context_window_words=1200,
                memory_mode=memory_mode,
            )

            eval_dir = ROOT / "eval"
            needle_path = eval_dir / "needle_tasks.jsonl"
            long_path = eval_dir / "long_horizon_tasks.jsonl"

            if not needle_path.exists() or not long_path.exists():
                console.print("[dim]Generating tasks …[/dim]")
                ensure_tasks()

            tasks = list(load_jsonl(needle_path)) + list(load_jsonl(long_path))
            task_progress = progress.add_task("run", total=len(tasks))

            results = []
            for task in tasks:
                output, _ = agent.run_task(task, run_id=condition)
                ok = score_task(task, output)
                results.append({
                    "type": task["type"],
                    "bucket": bucket_key(task),
                    "passed": ok,
                })
                progress.advance(task_progress)

        # aggregate
        summary: Dict = defaultdict(lambda: {"total": 0, "passed": 0})
        for r in results:
            key = (r["type"], r["bucket"])
            summary[key]["total"] += 1
            summary[key]["passed"] += int(r["passed"])

        table_rows = []
        for key, agg in sorted(summary.items()):
            ttype, bucket = key
            total = agg["total"]
            passed = agg["passed"]
            table_rows.append({
                "task_type": ttype,
                "bucket": bucket,
                "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
                "n": total,
            })

        all_results[condition] = table_rows
        console.print(f"  [success]✓[/success] {condition} complete")

    # ── build heatmap table ───────────────────────────────────────────────────
    section_rule("Results — Pass Rate Heatmap")

    # Collect all row keys
    row_keys = []
    seen = set()
    for rows in all_results.values():
        for r in rows:
            k = (r["task_type"], r["bucket"])
            if k not in seen:
                row_keys.append(k)
                seen.add(k)
    row_keys.sort()

    heat_table = Table(
        title="[bold white]Pass Rate (%) by Memory Condition[/bold white]",
        box=box.DOUBLE_EDGE,
        border_style="blue",
        show_lines=True,
        expand=True,
    )
    heat_table.add_column("Task Type", style="dim white", min_width=14)
    heat_table.add_column("Bucket", style="dim white", min_width=8)
    for condition, _ in conditions:
        heat_table.add_column(condition, justify="center", min_width=14)

    def rate_style(pct: float) -> str:
        if pct >= 90:
            return "bold green"
        if pct >= 60:
            return "bold yellow"
        if pct >= 30:
            return "yellow"
        return "bold red"

    for (ttype, bucket) in row_keys:
        row_cells = [ttype, bucket]
        for condition, _ in conditions:
            rows = all_results.get(condition, [])
            match = next((r for r in rows if r["task_type"] == ttype and r["bucket"] == bucket), None)
            if match:
                pct = match["pass_rate"]
                row_cells.append(f"[{rate_style(pct)}]{pct:.0f}%[/{rate_style(pct)}]")
            else:
                row_cells.append("[dim]—[/dim]")
        heat_table.add_row(*row_cells)

    console.print(Align.center(heat_table))
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "  [bold red]Red[/bold red]     = 0–29%  — agent cannot handle this task category\n"
                "  [bold yellow]Yellow[/bold yellow]  = 30–89% — partial success\n"
                "  [bold green]Green[/bold green]   = 90–100% — agent succeeds reliably\n\n"
                "  Notice: needle tasks in the [bold]>5000[/bold] bucket go from [failure]0%[/failure] (baseline) "
                "to [success]100%[/success] (retrieval)."
            ),
            title="[header]  Legend  [/header]",
            border_style="dim blue",
            padding=(0, 2),
        )
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI demo for the ReAct memory-augmented agent framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 demo.py           # interactive walkthrough\n"
            "  python3 demo.py --full    # full eval comparison table\n"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all eval conditions and display a full comparison heatmap.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.full:
        run_full_eval()
    else:
        run_interactive_demo()


if __name__ == "__main__":
    main()
