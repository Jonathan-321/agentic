import json
import os
import random
import string
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "data" / "needle"
DOC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = BASE_DIR / "needle_tasks.jsonl"

FILLER_WORDS = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".split()


def make_filler(words: int) -> str:
    population = FILLER_WORDS + ["analysis", "context", "information", "detail", "background"]
    return " ".join(random.choice(population) for _ in range(words))


def create_doc(word_count: int, key: str, value: str) -> str:
    insert_at = random.randint(word_count // 5, word_count - 10)
    words = make_filler(word_count).split()
    needle_sentence = f"NEEDLE: {key} -> {value}"
    words.insert(insert_at, needle_sentence)
    return " ".join(words)


def generate_tasks(n_per_bucket=8) -> None:
    random.seed(42)
    lengths = [500, 2000, 4000, 8000]  # approximate token buckets
    tasks = []
    for length in lengths:
        for _ in range(n_per_bucket):
            key = random.choice(string.ascii_lowercase) + str(random.randint(100, 999))
            value = random.choice(["alpha", "beta", "gamma", "delta", "omega"]) + str(random.randint(10, 99))
            doc_text = create_doc(length, key, value)
            task_id = f"needle-{uuid.uuid4().hex[:8]}"
            doc_path = DOC_DIR / f"{task_id}.txt"
            doc_path.write_text(doc_text)
            tasks.append(
                {
                    "id": task_id,
                    "type": "needle",
                    "doc_path": str(doc_path),
                    "key": key,
                    "answer": value,
                    "length_words": length,
                }
            )
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"wrote {len(tasks)} needle tasks to {OUTPUT}")


if __name__ == "__main__":
    generate_tasks()
