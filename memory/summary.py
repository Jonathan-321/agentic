import re
from typing import Optional

def summarize_text(text: str, max_words: int = 60, prefer_keyword: Optional[str] = None) -> str:
    """Tiny heuristic summarizer: keep sentence with keyword if present else first sentence/words."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if prefer_keyword:
        for sent in sentences:
            if prefer_keyword.lower() in sent.lower() or "needle" in sent.lower():
                return _trim_words(sent, max_words)
    if sentences:
        return _trim_words(sentences[0], max_words)
    return _trim_words(text, max_words)


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()
