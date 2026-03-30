from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

import requests
from loguru import logger
from omegaconf import ListConfig

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
REQUEST_TIMEOUT = (10, 60)
MAX_RECORDS_PER_REQUEST = 200


def _normalize_bool(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    lowered = str(value).strip().lower()
    if lowered in {"true", "false"}:
        return lowered
    raise ValueError(f"Invalid boolean-like value: {value}")


def _coerce_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ";" in text:
            items = [item.strip() for item in text.split(";") if item.strip()]
            return items or None
        return [text]
    if isinstance(value, (list, tuple, ListConfig)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    raise TypeError(f"Expected string or list of strings, got {type(value).__name__}")


@register_retriever("ieee")
class IEEERetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if not self.retriever_config.get("api_key"):
            raise ValueError("api_key must be specified for ieee.")
        if not self.retriever_config.get("querytext"):
            raise ValueError("querytext must be specified for ieee.")

    def _resolve_insert_date_range(self) -> tuple[str, str]:
        start_date = self.retriever_config.get("start_date")
        end_date = self.retriever_config.get("end_date")
        if start_date and end_date:
            return str(start_date), str(end_date)

        days_back = int(self.retriever_config.get("days_back", 1))
        if days_back < 1:
            raise ValueError("source.ieee.days_back must be a positive integer.")

        today_utc = datetime.now(timezone.utc).date()
        end_day = today_utc - timedelta(days=1)
        start_day = today_utc - timedelta(days=days_back)
        return start_day.strftime("%Y%m%d"), end_day.strftime("%Y%m%d")

    def _build_request_params(self, start_record: int) -> dict[str, str]:
        retriever_cfg = self.retriever_config
        start_date, end_date = self._resolve_insert_date_range()

        params: dict[str, str] = {
            "apikey": str(retriever_cfg.api_key),
            "querytext": str(retriever_cfg.querytext),
            "start_date": start_date,
            "end_date": end_date,
            "start_record": str(start_record),
            "max_records": str(
                min(int(retriever_cfg.get("max_records", 100)), MAX_RECORDS_PER_REQUEST)
            ),
            "publisher": str(retriever_cfg.get("publisher", "IEEE")),
        }

        open_access = _normalize_bool(retriever_cfg.get("open_access"))
        if open_access is not None:
            params["open_access"] = open_access

        content_types = _coerce_string_list(retriever_cfg.get("content_type"))
        if content_types:
            params["content_type"] = ",".join(content_types)

        publication_title = retriever_cfg.get("publication_title")
        if publication_title:
            params["publication_title"] = str(publication_title)

        author = retriever_cfg.get("author")
        if author:
            params["author"] = str(author)

        affiliation = retriever_cfg.get("affiliation")
        if affiliation:
            params["affiliation"] = str(affiliation)

        return params

    def _request_page(self, start_record: int) -> dict[str, Any]:
        retry_num = 5
        delay_time = 5
        last_exc: Exception | None = None
        params = self._build_request_params(start_record)
        for attempt in range(retry_num):
            try:
                response = requests.get(IEEE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt == retry_num - 1:
                    raise
                logger.warning(
                    f"Failed to retrieve IEEE papers at start_record={start_record}: {exc}. "
                    f"Retry in {delay_time} seconds."
                )
                sleep(delay_time)
        raise RuntimeError(f"Failed to retrieve IEEE papers: {last_exc}")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        requested_max_records = int(self.retriever_config.get("max_records", 100))
        target_total = max(1, requested_max_records)
        all_articles: list[dict[str, Any]] = []
        seen_article_numbers: set[str] = set()
        start_record = 1

        while len(all_articles) < target_total:
            payload = self._request_page(start_record)
            articles = payload.get("articles") or []
            if not isinstance(articles, list):
                raise ValueError("Unexpected IEEE response shape: 'articles' is not a list.")
            if len(articles) == 0:
                break

            for article in articles:
                article_number = str(article.get("article_number") or article.get("doi") or "")
                if article_number and article_number in seen_article_numbers:
                    continue
                if article_number:
                    seen_article_numbers.add(article_number)
                all_articles.append(article)
                if len(all_articles) >= target_total:
                    break

            total_records = int(payload.get("total_records", len(all_articles)) or len(all_articles))
            page_size = len(articles)
            if page_size == 0 or start_record + page_size > total_records:
                break
            start_record += page_size

            if self.config.executor.debug and len(all_articles) >= 10:
                break

        if self.config.executor.debug:
            all_articles = all_articles[:10]
        return all_articles[:target_total]

    @staticmethod
    def _extract_authors(article: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        authors_block = article.get("authors")
        if isinstance(authors_block, dict):
            author_entries = authors_block.get("authors") or []
        else:
            author_entries = authors_block or []

        if isinstance(author_entries, dict):
            author_entries = [author_entries]

        authors: list[str] = []
        affiliations: list[str] = []
        seen_affiliations: set[str] = set()

        for entry in author_entries:
            if not isinstance(entry, dict):
                text = str(entry).strip()
                if text:
                    authors.append(text)
                continue

            full_name = str(entry.get("full_name") or entry.get("name") or "").strip()
            if full_name:
                authors.append(full_name)

            affiliation = str(entry.get("affiliation") or "").strip()
            if affiliation and affiliation not in seen_affiliations:
                seen_affiliations.add(affiliation)
                affiliations.append(affiliation)

        article_affiliation = article.get("affiliation")
        if article_affiliation and not affiliations:
            raw_affiliations = _coerce_string_list(article_affiliation) or []
            for affiliation in raw_affiliations:
                if affiliation not in seen_affiliations:
                    seen_affiliations.add(affiliation)
                    affiliations.append(affiliation)

        return authors, (affiliations or None)

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = str(raw_paper.get("title") or "").strip()
        abstract = str(raw_paper.get("abstract") or "").strip()
        if not title or not abstract:
            logger.warning("Skipping IEEE paper without title or abstract.")
            return None

        authors, affiliations = self._extract_authors(raw_paper)
        article_number = str(raw_paper.get("article_number") or "").strip()
        url = (
            raw_paper.get("html_url")
            or raw_paper.get("abstract_url")
            or raw_paper.get("pdf_url")
            or (f"https://ieeexplore.ieee.org/document/{article_number}" if article_number else "")
            or ""
        )
        pdf_url = raw_paper.get("pdf_url") or url
        full_text = None
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=str(url),
            pdf_url=str(pdf_url) if pdf_url else None,
            full_text=full_text,
            affiliations=affiliations,
        )
