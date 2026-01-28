import os
from typing import List


class MemoryManager:
    """Lightweight episodic memory backed by plain text files."""

    def __init__(self, memory_dir: str = "memory/store"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

    def append(self, note: str, session: str = "default") -> str:
        path = os.path.join(self.memory_dir, f"{session}.notes.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(note.strip() + "\n")
        return path

    def search(self, query: str, session: str = "default", top_k: int = 3) -> List[str]:
        path = os.path.join(self.memory_dir, f"{session}.notes.txt")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        matches = [ln for ln in lines if query.lower() in ln.lower()]
        return matches[:top_k]
