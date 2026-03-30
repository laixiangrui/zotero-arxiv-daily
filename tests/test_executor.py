from datetime import datetime
from types import SimpleNamespace

from omegaconf import open_dict

import zotero_arxiv_daily.executor as executor_module
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


class _SuccessfulRetriever:
    def __init__(self, papers):
        self._papers = papers

    def retrieve_papers(self):
        return self._papers


class _FailingRetriever:
    def retrieve_papers(self):
        raise RuntimeError("403 Client Error: Forbidden")


class _PassthroughReranker:
    def rerank(self, papers, corpus):
        return papers


def test_executor_continues_when_one_source_fails(config, monkeypatch):
    with open_dict(config.executor):
        config.executor.source = ["arxiv", "ieee"]
        config.executor.max_paper_num = 20
        config.executor.send_empty = False

    corpus = [
        CorpusPaper(
            title="Wireless sensing survey",
            abstract="Survey of wireless sensing systems.",
            added_date=datetime(2026, 3, 30),
            paths=["2 AI Sensing"],
        )
    ]
    arxiv_paper = Paper(
        source="arxiv",
        title="Robust arXiv paper",
        authors=["Alice"],
        abstract="A robust paper summary.",
        url="https://arxiv.org/abs/1234.5678",
    )

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.include_path_patterns = None
    executor.retrievers = {
        "arxiv": _SuccessfulRetriever([arxiv_paper]),
        "ieee": _FailingRetriever(),
    }
    executor.reranker = _PassthroughReranker()
    executor.openai_client = SimpleNamespace()
    executor.fetch_zotero_corpus = lambda: corpus
    executor.filter_corpus = lambda items: items

    email_payload: dict[str, str] = {}

    monkeypatch.setattr(
        executor_module,
        "render_email",
        lambda papers: "\n".join(paper.title for paper in papers),
    )
    monkeypatch.setattr(
        executor_module,
        "send_email",
        lambda cfg, content: email_payload.setdefault("content", content),
    )
    monkeypatch.setattr(
        Paper,
        "generate_tldr",
        lambda self, client, llm: setattr(self, "tldr", self.abstract) or self.abstract,
    )
    monkeypatch.setattr(
        Paper,
        "generate_affiliations",
        lambda self, client, llm: None,
    )

    executor.run()

    assert email_payload["content"] == "Robust arXiv paper"
