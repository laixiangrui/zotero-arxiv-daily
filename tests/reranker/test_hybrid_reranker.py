from datetime import datetime, timedelta
import numpy as np
from zotero_arxiv_daily.protocol import Paper, CorpusPaper
from zotero_arxiv_daily.reranker.base import BaseReranker


class DummyReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        return np.array([
            [0.30, 0.20],
            [0.05, 0.05],
        ])


def test_hybrid_rerank_prefers_local_neighbor_and_lexical_overlap(config):
    reranker = DummyReranker(config)
    now = datetime.utcnow()
    corpus = [
        CorpusPaper(
            title="Diffusion transformers",
            abstract="Diffusion transformer for image generation with scalable sampling.",
            added_date=now,
            paths=["2026/survey/diffusion"],
        ),
        CorpusPaper(
            title="Graph neural network theory",
            abstract="Generalization theory for graph neural networks.",
            added_date=now - timedelta(days=30),
            paths=["2026/survey/gnn"],
        ),
    ]
    candidates = [
        Paper(
            source="arxiv",
            title="Image generation with diffusion transformers",
            authors=["A"],
            abstract="A diffusion transformer is proposed for image generation.",
            url="https://example.com/1",
        ),
        Paper(
            source="arxiv",
            title="Unrelated control benchmark",
            authors=["B"],
            abstract="A benchmark for classical control systems.",
            url="https://example.com/2",
        ),
    ]

    reranked = reranker.rerank(candidates, corpus)
    assert reranked[0].title == "Image generation with diffusion transformers"
    assert reranked[0].score > reranked[1].score
