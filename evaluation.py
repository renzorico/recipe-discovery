"""Offline evaluation: recall@10 and ndcg@10 over a judged query set.

    python evaluation.py dev                    # data/queries_dev.json
    python evaluation.py test                   # data/queries_test.json
    python evaluation.py test --path other.json --k 20

`evaluate_dev()` is for iterating locally. `evaluate_test()` is the same
measurement against a held-out set that is handed over at the technical
interview -- the dev judgements are visible while building, so a score on them
flatters whatever was tuned against them, and only the held-out number says
whether any of it generalised.

The two metric functions below are the definitions everything is scored with.
Treat them as fixed.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEV_QUERIES = ROOT / "data" / "queries_dev.json"
TEST_QUERIES = ROOT / "data" / "queries_test.json"


# --------------------------------------------------------------------------
# Metrics. Relevance is graded: 2 = strongly relevant, 1 = relevant, absent = not.
# --------------------------------------------------------------------------


def _dedupe_keep_first(ranked: list[str]) -> list[str]:
    """Drop repeats, keeping the first position.

    A repeated id must not be able to score twice.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for recipe_id in ranked:
        if recipe_id not in seen:
            seen.add(recipe_id)
            unique.append(recipe_id)
    return unique


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Share of the reachable relevant recipes found in the top k.

    Normalised by min(k, len(relevant)) rather than len(relevant): queries here
    have hundreds of relevant recipes, so dividing by the total would cap
    recall@10 near 0.01 for everyone and make the metric incomparable across
    queries.
    """
    if not relevant:
        return 0.0
    retrieved = set(_dedupe_keep_first(ranked)[:k])
    hits = sum(1 for recipe_id in relevant if recipe_id in retrieved)
    return hits / min(k, len(relevant))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain over graded relevance."""
    if not relevant:
        return 0.0

    gain = sum(
        (2 ** relevant.get(recipe_id, 0) - 1) / math.log2(position + 2)
        for position, recipe_id in enumerate(_dedupe_keep_first(ranked)[:k])
    )

    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    ideal = sum(
        (2**grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(ideal_grades)
    )

    return gain / ideal if ideal else 0.0


# --------------------------------------------------------------------------
# Running a query set through search()
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    query: str
    recall: float
    ndcg: float
    elapsed_ms: float


@dataclass(frozen=True)
class Report:
    name: str
    rows: list[Scored]
    warmup_seconds: float
    k: int

    @property
    def mean_recall(self) -> float:
        return (
            sum(row.recall for row in self.rows) / len(self.rows) if self.rows else 0.0
        )

    @property
    def mean_ndcg(self) -> float:
        return sum(row.ndcg for row in self.rows) / len(self.rows) if self.rows else 0.0

    def render(self) -> str:
        width = 79
        lines = [
            f"{self.name}: {len(self.rows)} queries, warmup {self.warmup_seconds:.1f}s",
            "",
            f"{'query':<50}{'recall@' + str(self.k):>11}{'ndcg@' + str(self.k):>10}{'ms':>8}",
            "-" * width,
        ]
        for row in self.rows:
            lines.append(
                f"{row.query[:48]:<50}{row.recall:>11.3f}{row.ndcg:>10.3f}{row.elapsed_ms:>8.0f}"
            )
        lines.append("-" * width)
        lines.append(f"{'MEAN':<50}{self.mean_recall:>11.3f}{self.mean_ndcg:>10.3f}")
        return "\n".join(lines)


def load_queries(path: Path) -> list[dict]:
    """Read a judged query set, failing with something actionable.

    The test set does not exist until it is handed over, so a missing file is
    an expected state rather than a bug.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No query set at {path}.\n"
            f"queries_test.json is provided at the technical interview; "
            f"until then use: python evaluation.py dev"
        )

    queries = json.loads(path.read_text())
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path} should hold a non-empty list of queries")
    for query in queries:
        if "text" not in query or "relevant" not in query:
            raise ValueError(f"{path}: every query needs 'text' and 'relevant' keys")
    return queries


def evaluate(queries: list[dict], k: int = 10, name: str = "queries") -> Report:
    """Run every query through search() and score the ranking."""
    import search as search_module

    started = time.perf_counter()
    search_module.warmup()
    warmup_seconds = time.perf_counter() - started

    rows = []
    for query in queries:
        call_started = time.perf_counter()
        hits = search_module.search(query["text"], k=k)
        elapsed_ms = (time.perf_counter() - call_started) * 1000

        ranked = [hit["recipe_id"] for hit in hits]
        rows.append(
            Scored(
                query=query["text"],
                recall=recall_at_k(ranked, query["relevant"], k),
                ndcg=ndcg_at_k(ranked, query["relevant"], k),
                elapsed_ms=elapsed_ms,
            )
        )
    return Report(name=name, rows=rows, warmup_seconds=warmup_seconds, k=k)


def evaluate_dev(k: int = 10, path: Path = DEV_QUERIES) -> Report:
    """Score against the dev judgements, for local iteration."""
    return evaluate(load_queries(path), k=k, name="dev")


def evaluate_test(k: int = 10, path: Path = TEST_QUERIES) -> Report:
    """Score against the held-out judgements handed over at the interview.

    Same measurement as evaluate_dev, different queries. Nothing here has seen
    them, which is the whole point.
    """
    return evaluate(load_queries(path), k=k, name="held-out test")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("split", choices=["dev", "test"], nargs="?", default="dev")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--path", type=Path, help="override the query-set path")
    args = parser.parse_args(argv)

    runner = evaluate_dev if args.split == "dev" else evaluate_test
    kwargs = {"k": args.k}
    if args.path is not None:
        kwargs["path"] = args.path

    try:
        report = runner(**kwargs)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
