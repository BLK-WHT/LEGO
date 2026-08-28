from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .client import embed
from .corpus import Article
from .graph import Edge, ProvisionGraph

DENSE_POOL = 50
GRAPH_POOL = 30
ROUTE_TOP_K = 8
ARTICLE_BUDGET = 8
ALIGN_WEIGHT = 0.55
COVERAGE_WEIGHT = 0.18
REDUNDANCY_WEIGHT = 0.06
HIGH_CONFIDENCE_BOOST = 1.15


@dataclass
class Route:
    node: str
    title: str
    score: float
    weight: float


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = query / (np.linalg.norm(query) + 1e-12)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return matrix @ query


def _rescale(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - low) / (high - low) for k, v in scores.items()}


def confidence_scale(edge: Edge) -> float:
    return HIGH_CONFIDENCE_BOOST if edge.confidence == "high" else 1.0


class ExpertGraphRetriever:
    def __init__(self, graph: ProvisionGraph, articles: list[Article]):
        self.graph = graph
        self.articles = articles
        self.vectors = embed([a.as_passage() for a in articles])
        self.nos = [a.no for a in articles]
        self.known = set(self.nos)

    def route(self, query_vector: np.ndarray, top_k: int = ROUTE_TOP_K) -> list[Route]:
        sims = _cosine(query_vector, self.graph.vectors)

        best: dict[str, tuple[float, int]] = {}
        for i, sim in enumerate(sims):
            node = self.graph.nodes[i].id
            if node not in best or sim > best[node][0]:
                best[node] = (float(sim), i)

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top_k]
        mass = sum(max(s, 0.0) for _, (s, _) in ranked) + 1e-12
        return [
            Route(node=node, title=self.graph.nodes[i].title, score=s, weight=max(s, 0.0) / mass)
            for node, (s, i) in ranked
        ]

    def _support(self, routes: list[Route]) -> tuple[dict[int, float], dict[int, set[str]]]:
        g: dict[int, float] = defaultdict(float)
        gamma: dict[int, set[str]] = defaultdict(set)
        for route in routes:
            for edge in self.graph.edges_for(route.node):
                score = route.weight * edge.weight * confidence_scale(edge)
                g[edge.article] = max(g[edge.article], score)
                gamma[edge.article].add(route.node)
        return dict(g), dict(gamma)

    def _select(self, candidates, dense, support, covers, k) -> list[int]:
        d, g = _rescale(dense), _rescale(support)
        selected: list[int] = []
        covered: set[str] = set()
        remaining = set(candidates)

        def delta(no: int) -> float:
            gamma = covers.get(no, set())
            return (
                d.get(no, 0.0)
                + ALIGN_WEIGHT * g.get(no, 0.0)
                + COVERAGE_WEIGHT * len(gamma - covered)
                - REDUNDANCY_WEIGHT * len(gamma & covered)
            )

        while remaining and len(selected) < k:
            best = max(remaining, key=delta)
            selected.append(best)
            covered |= covers.get(best, set())
            remaining.discard(best)
        return selected

    def retrieve(self, query: str, top_k: int = ARTICLE_BUDGET) -> tuple[list[int], list[Route]]:
        query_vector = embed([query])[0]

        sims = _cosine(query_vector, self.vectors)
        order = np.argsort(-sims)[:DENSE_POOL]
        dense = {self.nos[int(i)]: float(sims[int(i)]) for i in order}

        routes = self.route(query_vector)
        support, covers = self._support(routes)
        supported = sorted(
            (no for no in support if no in self.known), key=lambda no: -support[no]
        )[:GRAPH_POOL]

        candidates = set(dense) | set(supported)
        return self._select(candidates, dense, support, covers, top_k), routes
