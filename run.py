from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lego import (
    ExpertGraphRetriever,
    ProvisionGraph,
    answer,
    answer_without_provisions,
    load_articles,
    query_text,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score(predicted: set[str], gold: set[str]) -> tuple[int, float]:
    overlap = len(predicted & gold)
    if not overlap:
        return 0, 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return int(predicted == gold), 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--articles", help="required unless --method zeroshot")
    parser.add_argument("--method", choices=("lego", "zeroshot"), default="lego")
    parser.add_argument("--graph", default="data/graph")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    items = read_jsonl(Path(args.items))
    zeroshot = args.method == "zeroshot"
    if zeroshot:
        print(f"{len(items)} items, closed book.")
        articles, by_no, retriever = [], {}, None
    else:
        if not args.articles:
            parser.error("--articles is required unless --method zeroshot")
        articles = load_articles(args.articles)
        by_no = {a.no: a for a in articles}
        print(f"{len(items)} items, {len(articles)} articles; loading graph...")
        retriever = ExpertGraphRetriever(ProvisionGraph.load(args.graph), articles)

    def solve(indexed: tuple[int, dict]) -> dict:
        index, item = indexed
        case, question = item.get("case", ""), item.get("question", "")
        options = item.get("options", "")
        if zeroshot:
            selected, routes = [], []
            prediction = answer_without_provisions(case, question, options)
        else:
            selected, routes = retriever.retrieve(
                query_text(case, question, options), top_k=args.top_k
            )
            prediction = answer(
                case, question, options, [by_no[n] for n in selected if n in by_no]
            )

        row = {
            "index": index,
            "id": item.get("id"),
            "method": args.method,
            "predicted": "".join(sorted(prediction.answer)),
            "articles": selected,
            "concepts": [r.node for r in routes],
            "response": prediction.response,
        }
        gold = set(re.findall(r"[A-D]", str(item.get("answer", "")).upper()))
        if gold and prediction.answer:
            row["exact_match"], row["f1"] = score(prediction.answer, gold)
        elif gold:
            row["exact_match"], row["f1"] = 0, 0.0
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = sorted(pool.map(solve, enumerate(items)), key=lambda r: r["index"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = [r for r in results if "exact_match" in r]
    if scored:
        accuracy = sum(r["exact_match"] for r in scored) / len(scored)
        macro_f1 = sum(r["f1"] for r in scored) / len(scored)
        print(f"accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}  ({len(scored)} scored)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
