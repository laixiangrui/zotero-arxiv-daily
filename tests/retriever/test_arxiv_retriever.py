import time
from types import SimpleNamespace
import feedparser
import io
from urllib.error import HTTPError

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
from zotero_arxiv_daily.retriever.base import BaseRetriever, register_retriever
from zotero_arxiv_daily.protocol import Paper
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever
import zotero_arxiv_daily.retriever.base as base_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value



def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def _build_fake_arxiv_results(parsed_result, include_cross_list: bool):
    allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
    fake_results = []
    for entry in parsed_result.entries:
        if entry.get("arxiv_announce_type", "new") not in allowed_announce_types:
            continue
        summary = (
            entry.get("summary")
            or entry.get("description")
            or ""
        )
        raw_authors = entry.get("authors") or []
        authors = []
        for author in raw_authors:
            author_name = getattr(author, "name", None)
            if author_name is None and isinstance(author, dict):
                author_name = author.get("name")
            if author_name:
                authors.append(SimpleNamespace(name=author_name))
        link = entry.get("link") or ""
        fake_results.append(
            SimpleNamespace(
                title=entry.title,
                summary=summary,
                authors=authors,
                pdf_url=link.replace("/abs/", "/pdf/") if link else None,
                entry_id=link,
                source_url=lambda: None,
            )
        )
    return fake_results


def _mock_arxiv_network(parsed_result, include_cross_list: bool, monkeypatch):
    fake_results = _build_fake_arxiv_results(parsed_result, include_cross_list)
    def mock_retrieve_raw_papers(self):
        raw_papers = fake_results
        keywords = self.config.source.arxiv.get("keywords")
        if keywords:
            raw_papers = [paper for paper in raw_papers if self._match_keywords(paper)]
        return raw_papers

    monkeypatch.setattr(ArxivRetriever, "_retrieve_raw_papers", mock_retrieve_raw_papers)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: "Mock HTML full text")
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)
    monkeypatch.setattr(base_retriever, "sleep", lambda _: None)



def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    _mock_arxiv_network(parsed_result, include_cross_list=False, monkeypatch=monkeypatch)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type", "new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)


def test_arxiv_retriever_keyword_filter_any(config, monkeypatch):
    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse

    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)

    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    with open_dict(config.source.arxiv):
        config.source.arxiv.include_cross_list = True
        config.source.arxiv.keywords = ["time series forecasting", "multimodal"]
        config.source.arxiv.keyword_match = "any"
    _mock_arxiv_network(parsed_result, include_cross_list=True, monkeypatch=monkeypatch)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title.startswith("EventTSF: Event-Aware Non-Stationary Time Series Forecasting")


def test_arxiv_retriever_keyword_filter_all(config, monkeypatch):
    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse

    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)

    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    with open_dict(config.source.arxiv):
        config.source.arxiv.include_cross_list = True
        config.source.arxiv.keywords = ["time series forecasting", "multimodal"]
        config.source.arxiv.keyword_match = "all"
    _mock_arxiv_network(parsed_result, include_cross_list=True, monkeypatch=monkeypatch)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert "EventTSF" in papers[0].title


def test_arxiv_retriever_falls_back_to_rss_when_api_is_rate_limited(config, monkeypatch):
    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse

    def mock_feedparser_parse(url):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url)

    class FailingArxivClient:
        def __init__(self, *args, **kwargs):
            pass

        def results(self, search):
            raise RuntimeError("HTTP 429")

    with open_dict(config.source.arxiv):
        config.source.arxiv.include_cross_list = False
        config.source.arxiv.keywords = None
        config.source.arxiv.keyword_match = "any"

    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FailingArxivClient)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)
    monkeypatch.setattr(base_retriever, "sleep", lambda _: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type", "new") == "new"]
    assert len(papers) == len(parsed_results)
    assert {paper.title for paper in papers} == {entry.title for entry in parsed_results}
    assert all(not paper.abstract.startswith("arXiv:") for paper in papers if paper.abstract)


@register_retriever("failing_test")
class FailingTestRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "good paper", "mode": "ok"},
            {"title": "bad paper", "mode": "fail"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper | None:
        if raw_paper["mode"] == "fail":
            raise HTTPError(
                url="https://example.com/paper.pdf",
                code=404,
                msg="not found",
                hdrs=None,
                fp=io.BufferedReader(io.BytesIO(b"missing")),
            )
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['mode']}",
        )


@register_retriever("serial_test")
class SerialTestRetriever(BaseRetriever):
    def __init__(self, config, seen_titles: list[str]):
        super().__init__(config)
        self.seen_titles = seen_titles

    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "paper 1"},
            {"title": "paper 2"},
            {"title": "paper 3"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper:
        self.seen_titles.append(raw_paper["title"])
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['title']}",
        )



def test_retrieve_papers_skips_conversion_errors(config):
    with open_dict(config.source):
        config.source.failing_test = {}

    retriever = FailingTestRetriever(config)

    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == ["good paper"]


def test_retrieve_papers_runs_serially(config):
    with open_dict(config.source):
        config.source.serial_test = {}

    seen_titles: list[str] = []
    retriever = SerialTestRetriever(config, seen_titles)

    papers = retriever.retrieve_papers()

    expected_titles = ["paper 1", "paper 2", "paper 3"]
    assert seen_titles == expected_titles
    assert [paper.title for paper in papers] == expected_titles



def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 0.01),
        timeout=1,
        operation="test operation",
        paper_title="paper",
    )

    assert result == "done"



def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 1.0),
        timeout=0.01,
        operation="test operation",
        paper_title="paper",
    )

    assert result is None
    assert warnings == ["test operation timed out for paper after 0.01 seconds"]



def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    result = _run_with_hard_timeout(
        _raise_runtime_error,
        (),
        timeout=1,
        operation="test operation",
        paper_title="paper",
    )

    assert result is None
    assert warnings == ["test operation failed for paper: RuntimeError: boom"]
