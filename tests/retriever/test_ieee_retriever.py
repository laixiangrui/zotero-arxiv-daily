import pytest
import requests
from omegaconf import open_dict

import zotero_arxiv_daily.retriever.base as base_retriever
import zotero_arxiv_daily.retriever.ieee_retriever as ieee_retriever
from zotero_arxiv_daily.retriever.ieee_retriever import IEEERetriever


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_ieee_retriever_retrieves_and_converts_metadata(config, monkeypatch):
    with open_dict(config.source):
        config.source.ieee.api_key = "ieee-test-key"
        config.source.ieee.querytext = "(ISAC OR \"integrated sensing and communication\") AND wireless"
        config.source.ieee.content_type = ["Journals", "Conferences"]
        config.source.ieee.publisher = "IEEE"
        config.source.ieee.open_access = True
        config.source.ieee.start_date = "20260329"
        config.source.ieee.end_date = "20260330"
        config.source.ieee.max_records = 5

    captured_params = {}

    def mock_get(url, params, timeout):
        captured_params.update(params)
        assert url == ieee_retriever.IEEE_API_URL
        assert timeout == ieee_retriever.REQUEST_TIMEOUT
        return _MockResponse(
            {
                "total_records": 1,
                "articles": [
                    {
                        "article_number": "123456",
                        "title": "Joint ISAC waveform design with sparse sensing priors",
                        "abstract": "A practical ISAC design is studied for sparse target sensing.",
                        "html_url": "https://ieeexplore.ieee.org/document/123456",
                        "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=123456",
                        "authors": {
                            "authors": [
                                {
                                    "full_name": "Alice Smith",
                                    "affiliation": "Tsinghua University",
                                },
                                {
                                    "full_name": "Bob Lee",
                                    "affiliation": "Peking University",
                                },
                            ]
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(ieee_retriever.requests, "get", mock_get)
    monkeypatch.setattr(base_retriever, "sleep", lambda _: None)

    retriever = IEEERetriever(config)
    papers = retriever.retrieve_papers()

    assert captured_params["apikey"] == "ieee-test-key"
    assert captured_params["querytext"] == config.source.ieee.querytext
    assert captured_params["start_date"] == "20260329"
    assert captured_params["end_date"] == "20260330"
    assert captured_params["content_type"] == "Journals,Conferences"
    assert captured_params["publisher"] == "IEEE"
    assert captured_params["open_access"] == "true"

    assert len(papers) == 1
    paper = papers[0]
    assert paper.source == "ieee"
    assert paper.title.startswith("Joint ISAC waveform design")
    assert paper.authors == ["Alice Smith", "Bob Lee"]
    assert paper.affiliations == ["Tsinghua University", "Peking University"]
    assert paper.url == "https://ieeexplore.ieee.org/document/123456"
    assert paper.pdf_url.endswith("123456")
    assert paper.full_text is None


def test_ieee_retriever_uses_article_affiliation_and_url_fallback(config, monkeypatch):
    with open_dict(config.source):
        config.source.ieee.api_key = "ieee-test-key"
        config.source.ieee.querytext = "wireless sensing"
        config.source.ieee.start_date = "20260329"
        config.source.ieee.end_date = "20260330"
        config.source.ieee.max_records = 1

    def mock_get(url, params, timeout):
        return _MockResponse(
            {
                "total_records": 1,
                "articles": [
                    {
                        "article_number": "7890",
                        "title": "IEEE early access sensing benchmark",
                        "abstract": "This paper studies a sensing benchmark for early-access systems.",
                        "abstract_url": "https://ieeexplore.ieee.org/abstract/document/7890",
                        "authors": {"authors": [{"full_name": "Carol Wang"}]},
                        "affiliation": "Shanghai Jiao Tong University",
                    }
                ],
            }
        )

    monkeypatch.setattr(ieee_retriever.requests, "get", mock_get)
    monkeypatch.setattr(base_retriever, "sleep", lambda _: None)

    retriever = IEEERetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    paper = papers[0]
    assert paper.authors == ["Carol Wang"]
    assert paper.affiliations == ["Shanghai Jiao Tong University"]
    assert paper.url == "https://ieeexplore.ieee.org/abstract/document/7890"
    assert paper.pdf_url == paper.url


def test_ieee_retriever_requires_api_key_and_querytext(config):
    with open_dict(config.source):
        config.source.ieee.api_key = None
        config.source.ieee.querytext = None

    with pytest.raises(ValueError, match="api_key must be specified for ieee"):
        IEEERetriever(config)


def test_ieee_retriever_does_not_retry_on_forbidden(config, monkeypatch):
    with open_dict(config.source):
        config.source.ieee.api_key = "ieee-test-key"
        config.source.ieee.querytext = "wireless sensing"
        config.source.ieee.start_date = "20260329"
        config.source.ieee.end_date = "20260330"
        config.source.ieee.max_records = 1

    call_count = 0

    class _ForbiddenResponse:
        status_code = 403

        def raise_for_status(self) -> None:
            raise requests.HTTPError("403 Client Error: Forbidden", response=self)

    def mock_get(url, params, timeout):
        nonlocal call_count
        call_count += 1
        return _ForbiddenResponse()

    monkeypatch.setattr(ieee_retriever.requests, "get", mock_get)

    retriever = IEEERetriever(config)
    with pytest.raises(requests.HTTPError, match="403 Client Error"):
        retriever._request_page(start_record=1)

    assert call_count == 1
