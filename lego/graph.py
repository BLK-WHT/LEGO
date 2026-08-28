from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .client import embed

CARD_FIELDS = (
    "title",
    "issue_frames",
    "positive_conditions",
    "negative_conditions",
    "legal_effects",
    "distinguish_from",
    "priority_or_exception",
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def card_text(card: dict) -> str:
    parts = []
    for name in CARD_FIELDS:
        value = card.get(name)
        parts.append(value if isinstance(value, str) else " ".join(value or []))
    return "\n".join(p for p in parts if p)


@dataclass
class Node:
    id: str
    title: str


@dataclass
class Edge:
    node: str
    article: int
    weight: float
    confidence: str


class ProvisionGraph:
    def __init__(self, nodes: list[Node], edges: list[Edge], vectors: np.ndarray):
        self.nodes = nodes
        self.edges = edges
        self.vectors = vectors
        self._by_node: dict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            self._by_node[edge.node].append(edge)

    def edges_for(self, node_id: str) -> list[Edge]:
        return self._by_node.get(node_id, [])

    @classmethod
    def load(cls, graph_dir: str | Path) -> "ProvisionGraph":
        graph_dir = Path(graph_dir)
        rows = _read_jsonl(graph_dir / "nodes.jsonl")

        nodes = [Node(id=row["id"], title=row["card"]["title"]) for row in rows]
        edges = [
            Edge(
                node=row["node"],
                article=int(row["article"]),
                weight=float(row["weight"]),
                confidence=row["confidence"],
            )
            for row in _read_jsonl(graph_dir / "edges.jsonl")
        ]
        return cls(nodes, edges, embed([card_text(row["card"]) for row in rows]))
