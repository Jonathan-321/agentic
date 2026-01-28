# Agentic Long-Runner

Tool-using ReAct agent that can stay on track across long tasks by mixing logging, memory, and evals. Comes with synthetic benchmarks and ablations you can run in minutes.

## Why it exists
- Long docs and long multi-step instructions make agents lose context; we need memory + compression, not just bigger prompts.
- This repo is a minimal, inspectable lab: you can tweak memory, rerun evals, and see exactly where the agent wins or fails.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# run one condition (memory modes: none | summary | retrieval | both)
python3 eval/run.py --condition baseline --memory none
```
Artifacts land in `runs/` (JSONL traces) and `report/` (result tables).

## What’s inside
- **agent/loop.py** – ReAct loop for two task types (needle, long-horizon) with memory modes.
- **agent/context.py** – Rolling context + auto-summarization when over word budget.
- **memory/** – episodic text store, heuristic summarizer, bag-of-words vector store.
- **tools/builtin.py** – python exec, read/write file, append/search memory (vector-backed).
- **eval/** – task generators + runner; produces ground-truth-labeled tasks on demand.
- **runs/** – stepwise logs: thought, action, tool, observation, timestamps.
- **report/** – per-run JSON tables; `report/latest_table.md` shows the headline numbers.

## How the system works (at a glance)
1) `eval/run.py` builds tasks → spins up agent with a chosen memory mode.  
2) For **needle tasks**: read doc → (optionally) summarize + chunk/index → retrieve by key → extract value.  
3) For **long-horizon tasks**: optional memory lookup → execute deterministic “recipe” via python tool → redact banned token → return JSON.  
4) Context manager trims history into summaries when over budget; notes/chunks go into vector store for retrieval.  
5) Every step is logged; scorer computes pass/fail and aggregates.

## Current results (Jan 27, 2026)
Command pattern: `python3 eval/run.py --condition <label> --memory {none,summary,retrieval,both}`

| Condition | Task type    | Context bucket | Pass rate | Avg steps | N |
|-----------|--------------|----------------|-----------|-----------|---|
| baseline  | long_horizon | n/a            | 100%      | 2.0       | 20 |
| baseline  | needle       | <=800          | 100%      | 3.0       | 8 |
| baseline  | needle       | <=2500         | 37.5%     | 3.0       | 8 |
| baseline  | needle       | <=5000         | 25.0%     | 3.0       | 8 |
| baseline  | needle       | >5000          | 0%        | 3.0       | 8 |
| summary   | needle       | >5000          | 0%        | 3.0       | 8 |
| retrieval | needle       | >5000          | 100%      | 3.0       | 8 |
| both      | needle       | >5000          | 100%      | 3.0       | 8 |

Takeaway: chunked vector retrieval fixes long-doc failures; summarization alone doesn’t yet help.

## How to inspect a run
- Logs: `runs/<run_id>.jsonl` (one JSON object per step).  
- Reports: `report/results_<run_id>.json` (aggregated metrics).  
- Quick glance: `report/latest_table.md` (side-by-side conditions).

## Roadmap (next iterations)
- Better summarization + salience scoring; cap what we store.
- Prompt builder that mixes latest turns, retrieved notes, and summaries with a strict token budget.
- Reflexion loop: on failure, write “why + fix” notes and optionally retry.
- Add a small chart + 1–2 page “results & lessons” write-up and a short demo clip.
