import json
import os
import random
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "long_horizon_tasks.jsonl"
TOPICS = ["astronomy", "gardening", "finance", "basketball", "mycology", "cinema"]
BANNED = ["forbidden", "skip", "omit", "ban"]


def build_recipe() -> dict:
    words = ["comet", "ferns", "ledger", "pivot", "truffle", "camera", "nebula", "orchid"]
    random.shuffle(words)
    fields = {
        "summary": {"op": "concat", "args": words[:3]},
        "vowel_count": {"op": "count_vowels", "args": words[3:4]},
        "backwards": {"op": "reverse", "args": words[4:5]},
        "shout": {"op": "uppercase", "args": words[5:6]},
    }
    return {
        "topic": random.choice(TOPICS),
        "banned_word": random.choice(BANNED),
        "fields": fields,
    }


def compute_expected(recipe: dict) -> str:
    def count_vowels(text: str) -> int:
        return sum(1 for ch in text.lower() if ch in "aeiou")

    result = {}
    for field, spec in recipe["fields"].items():
        if spec["op"] == "concat":
            result[field] = "-".join(spec["args"])
        elif spec["op"] == "count_vowels":
            result[field] = count_vowels(spec["args"][0])
        elif spec["op"] == "reverse":
            result[field] = spec["args"][0][::-1]
        elif spec["op"] == "uppercase":
            result[field] = spec["args"][0].upper()
    return json.dumps(result, sort_keys=True)


def recipe_to_instruction(recipe: dict) -> str:
    parts = [
        "Follow all constraints carefully:",
        "1) Produce a JSON object only, no narration.",
        "2) Keys must be summary, vowel_count, backwards, shout.",
        "3) summary = join the first three tokens with hyphens in the order given.",
        "4) vowel_count = number of vowels in the fourth token.",
        "5) backwards = reverse the fifth token.",
        "6) shout = uppercase the sixth token.",
        f"7) Never include the word '{recipe['banned_word']}' anywhere.",
    ]
    tokens = list(recipe["fields"].values())
    # tokens list of specs in order - ensure consistent order for text
    ordered_tokens = [
        recipe["fields"]["summary"]["args"],
        recipe["fields"]["vowel_count"]["args"],
        recipe["fields"]["backwards"]["args"],
        recipe["fields"]["shout"]["args"],
    ]
    flat_tokens = [item for sublist in ordered_tokens for item in sublist]
    parts.append(f"Tokens (in order): {' '.join(flat_tokens)}")
    parts.append("Return JSON only.")
    return "\n".join(parts)


def generate_tasks(n_tasks: int = 20) -> None:
    random.seed(123)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for _ in range(n_tasks):
            recipe = build_recipe()
            expected = compute_expected(recipe)
            instructions = recipe_to_instruction(recipe)
            task = {
                "id": f"long-{uuid.uuid4().hex[:8]}",
                "type": "long_horizon",
                "topic": recipe["topic"],
                "instructions": instructions,
                "recipe": recipe,
                "answer": expected,
            }
            f.write(json.dumps(task) + "\n")
    print(f"wrote {n_tasks} long-horizon tasks to {OUTPUT}")


if __name__ == "__main__":
    generate_tasks()
