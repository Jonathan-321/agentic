import json
import os
import time
from typing import Any, Dict, Optional


class JSONLLogger:
    """Append-only JSONL logger for agent traces."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.start_time = time.time()

    def log_step(
        self,
        run_id: str,
        task_id: str,
        step: int,
        thought: str,
        action: str,
        tool: Optional[str],
        tool_input: Optional[Dict[str, Any]],
        observation: Optional[str],
        decision: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> None:
        entry = {
            "ts": time.time(),
            "elapsed": time.time() - self.start_time,
            "run_id": run_id,
            "task_id": task_id,
            "step": step,
            "thought": thought,
            "action": action,
            "tool": tool,
            "tool_input": tool_input,
            "observation": (observation or "")[:500],
            "decision": decision,
            "tokens_used": tokens_used,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_summary(self, summary: Dict[str, Any]) -> None:
        summary_entry = {"type": "summary", **summary}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_entry) + "\n")
