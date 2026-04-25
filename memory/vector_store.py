import math
import uuid
from collections import Counter
from typing import Dict, List, Tuple


class VectorStore:
    """Lightweight bag-of-words vector store with cosine similarity."""

    def __init__(self):
        self.items: Dict[str, Dict] = {}

    def _vectorize(self, text: str) -> Counter:
        tokens = [t.lower() for t in text.split() if t.isalpha() or t.isalnum()]
        return Counter(tokens)

    def add(self, text: str, metadata: Dict | None = None) -> str:
        vid = str(uuid.uuid4())
        vec = self._vectorize(text)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        self.items[vid] = {"text": text, "vec": vec, "norm": norm, "meta": metadata or {}}
        return vid

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        qvec = self._vectorize(query)
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scores = []
        for vid, item in self.items.items():
            score = self._cosine(qvec, qnorm, item["vec"], item["norm"])
            scores.append((vid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_text(self, vid: str) -> str:
        return self.items.get(vid, {}).get("text", "")

    def _cosine(self, qvec: Counter, qnorm: float, dvec: Counter, dnorm: float) -> float:
        dot = sum(val * dvec.get(tok, 0) for tok, val in qvec.items())
        return dot / (qnorm * dnorm) if qnorm and dnorm else 0.0
