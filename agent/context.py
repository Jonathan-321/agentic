from typing import List, Dict
from memory.summary import summarize_text


class ContextManager:
    """Maintains rolling context with opportunistic summarization to fit word budget."""

    def __init__(self, max_words: int = 1200):
        self.max_words = max_words
        self.history: List[Dict[str, str]] = []
        self.summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self._maybe_summarize()

    def _word_count(self) -> int:
        words = sum(len(item["content"].split()) for item in self.history)
        words += len(self.summary.split())
        return words

    def _maybe_summarize(self) -> None:
        if self._word_count() <= self.max_words:
            return
        # summarize oldest half of the history
        keep = self.history[len(self.history) // 2 :]
        discard = self.history[: len(self.history) // 2]
        discard_text = " \n".join(item["content"] for item in discard)
        compressed = summarize_text(discard_text, max_words=120)
        self.summary = f"{self.summary}\n{compressed}".strip()
        self.history = keep

    def build_context(self) -> str:
        parts: List[str] = []
        if self.summary:
            parts.append(f"[summary]\n{self.summary}")
        for item in self.history:
            parts.append(f"[{item['role']}] {item['content']}")
        return "\n".join(parts)
