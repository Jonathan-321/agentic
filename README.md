# Long-Runner Agent (Day 1 baseline)

A lightweight ReAct-style tool-using agent with logging and synthetic evals for long-context behavior. Day 1 delivers the scaffold, eval harness, and baseline numbers (no memory, limited context window).

## Repo layout
- `agent/` – agent loop, logging, builder.
- `tools/` – builtin tools (`python_exec`, `read_file`, `write_file`, `append_note`, `search_memory`).
- `memory/` – simple episodic store (text files) used by tools.
- `eval/` – task generators and eval runner; auto-writes tasks to `eval/data/`.
- `runs/` – JSONL traces per run.
- `report/` – result tables per run.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 eval/run.py --condition baseline
```
Optional memory flag (stub for Day 2 work):
```bash
python3 eval/run.py --condition mem --memory
```

## What the agent does (today)
- ReAct loop with stepwise logging (thought, action, tool, observation, timestamp).
- Context window cap for needle tasks to simulate overflow pressure.
- Tools: sandboxed `python_exec`, file read/write, episodic note append/search.
- Long-horizon tasks solved via structured “recipe” + python tool execution for transparency.

## Eval suite (synthetic)
- **Needle-in-haystack**: long docs (0.5k–8k words) hiding `NEEDLE: key -> value` tokens.
- **Long-horizon instructions**: constrained JSON outputs with banned tokens.
- 52 tasks total (32 needle, 20 long-horizon) generated on demand.

## Latest results (Jan 27, 2026)
`python3 eval/run.py --condition <label> --memory {none,summary,retrieval,both}`

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

Takeaway: vector retrieval of chunked docs restores 100% accuracy on the largest bucket; simple summaries alone are insufficient.

## Notes / next steps
- Add summary + retrieval memory to lift long-doc buckets.
- Implement context manager that mixes recent turns + retrieved notes.
- Add reflexion logging for failures and retry subset.
- Ship a short demo clip (pending) and expand report with ablation chart.
