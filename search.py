"""Entry point. This is the only interface the evaluation depends on.

`evaluation.py` imports this module, calls `warmup()` once, then calls
`search()` for every query. 

What is here now is a placeholder that picks recipes at random. It exists so
the pipeline runs end to end before you have written anything. Replace
both functions.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

RECIPES = Path(__file__).resolve().parent / "data" / "recipes.parquet"

MAX_K = 100

_recipe_ids: list[str] = []


def warmup() -> None:
    """Load or build whatever `search()` needs. Called once, before any query.

    REPLACE THE BODY of this function with your own set-up: load your embedding
    model, build or open your index, read the catalogue. Keep the signature.

    Embedding 30,000 recipes takes minutes, so do it here rather than on the
    first query -- otherwise the first `search()` pays the whole cost.
    """
    # Placeholder set-up: just the list of ids to pick from. Replace.
    global _recipe_ids
    frame = pd.read_parquet(RECIPES, columns=["recipe_id"])
    _recipe_ids = frame["recipe_id"].astype(str).tolist()


def search(query: str, k: int = 10) -> list[dict]:
    """Return up to k results for a natural-language request, best first.

    Each result is a dict:

        recipe_id    str    must match a recipe_id in data/recipes.parquet
        score        float  your relevance score; only the ordering is scored
        explanation  str    one line on why this recipe fits the request

    REPLACE THE BODY of this function with your own retrieval: embed the
    query, search your index, rank the results, explain them. Keep the
    signature and the return shape -- that is all the evaluation depends on.
    The validation below is worth keeping.

    Returning fewer than k is fine. Duplicate recipe_ids are not: only the
    first occurrence counts, so a repeat wastes a slot.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(k, int) or isinstance(k, bool) or not (1 <= k <= MAX_K):
        raise ValueError(f"k must be an integer in 1..{MAX_K}")

    if not _recipe_ids:
        warmup()

    # ---- PLACEHOLDER: delete everything below and return your own results --
    chosen = random.Random(query).sample(_recipe_ids, min(k, len(_recipe_ids)))
    return [
        {
            "recipe_id": recipe_id,
            "score": 1.0 - position / k,
            "explanation": "placeholder result -- search() has not been implemented yet",
        }
        for position, recipe_id in enumerate(chosen)
    ]


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "something vegetarian and quick that doesn't need an oven"
    warmup()
    for position, hit in enumerate(search(text), start=1):
        print(f"{position:2d}. [{hit['score']:.2f}] {hit['recipe_id']}  {hit['explanation']}")
