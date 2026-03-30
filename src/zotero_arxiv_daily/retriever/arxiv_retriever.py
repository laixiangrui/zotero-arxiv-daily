from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import re

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


@dataclass
class ArxivFeedPaper:
    title: str
    summary: str
    authors: list[str]
    entry_id: str
    pdf_url: str | None = None

    def source_url(self) -> str | None:
        return None


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _normalize_feed_summary(summary: str | None) -> str:
    if not summary:
        return ""
    normalized = re.sub(r"\s+", " ", summary).strip()
    match = re.search(r"\bAbstract:\s*(.*)", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return normalized


def _extract_author_names(raw_authors: Any) -> list[str]:
    author_names: list[str] = []
    for author in raw_authors or []:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = str(author.get("name", "")).strip()
        else:
            name = str(getattr(author, "name", "")).strip()
        if name:
            author_names.append(name)
    return author_names


def _extract_feed_author_names(entry: Any) -> list[str]:
    author_names = _extract_author_names(entry.get("authors"))
    if author_names:
        return author_names
    raw_author_text = entry.get("author") or entry.get("dc_creator") or entry.get("creator") or ""
    if not isinstance(raw_author_text, str):
        return []
    return [item.strip() for item in raw_author_text.split(",") if item.strip()]


def _entry_to_feed_paper(entry: Any) -> ArxivFeedPaper:
    entry_id = str(entry.get("link") or "")
    if not entry_id:
        paper_id = str(entry.get("id") or "").removeprefix("oai:arXiv.org:")
        if paper_id:
            entry_id = f"https://arxiv.org/abs/{paper_id}"
    pdf_url = entry_id.replace("/abs/", "/pdf/") if entry_id else None
    return ArxivFeedPaper(
        title=str(entry.get("title") or ""),
        summary=_normalize_feed_summary(entry.get("summary") or entry.get("description")),
        authors=_extract_feed_author_names(entry),
        entry_id=entry_id,
        pdf_url=pdf_url,
    )


def _get_paper_id(raw_paper: ArxivResult | ArxivFeedPaper) -> str | None:
    entry_id = getattr(raw_paper, "entry_id", None) or getattr(raw_paper, "id", None)
    if not isinstance(entry_id, str):
        return None
    if "/abs/" in entry_id:
        return entry_id.rsplit("/abs/", maxsplit=1)[-1]
    return entry_id.removeprefix("oai:arXiv.org:")


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    @staticmethod
    def _normalize_keyword(keyword: str) -> str:
        return re.sub(r"\s+", " ", keyword.strip().lower())

    def _match_keywords(self, paper: ArxivResult | ArxivFeedPaper) -> bool:
        keywords = self.config.source.arxiv.get("keywords")
        if not keywords:
            return True
        text = f"{paper.title}\n{paper.summary}".lower()
        normalized_text = re.sub(r"\s+", " ", text)
        match_mode = self.config.source.arxiv.get("keyword_match", "any")
        normalized_keywords = [self._normalize_keyword(keyword) for keyword in keywords if keyword.strip()]
        if not normalized_keywords:
            return True
        if match_mode == "all":
            return all(keyword in normalized_text for keyword in normalized_keywords)
        return any(keyword in normalized_text for keyword in normalized_keywords)

    def _retrieve_raw_papers(self) -> list[ArxivResult | ArxivFeedPaper]:
        # RSS already gives the daily candidate set. arXiv API enrichment is best effort,
        # so fail fast and fall back to RSS metadata when the API is rate-limited.
        client = arxiv.Client(num_retries=0, delay_seconds=3)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        feed_entries = [
            entry
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        fallback_papers_by_id = {
            paper_id: _entry_to_feed_paper(entry)
            for entry in feed_entries
            if (paper_id := str(entry.get("id") or "").removeprefix("oai:arXiv.org:"))
        }
        all_paper_ids = list(fallback_papers_by_id.keys())
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0, len(all_paper_ids), 20):
            batch_ids = all_paper_ids[i:i + 20]
            search = arxiv.Search(id_list=batch_ids)
            try:
                batch = list(client.results(search))
                raw_papers.extend(batch)
                returned_ids = {_get_paper_id(paper) for paper in batch}
                missing_ids = [paper_id for paper_id in batch_ids if paper_id not in returned_ids]
                if missing_ids:
                    raw_papers.extend(
                        fallback_papers_by_id[paper_id]
                        for paper_id in missing_ids
                        if paper_id in fallback_papers_by_id
                    )
                    logger.warning(
                        f"arXiv API returned incomplete metadata for {len(missing_ids)} papers; "
                        "using RSS metadata fallback for the missing entries."
                    )
            except Exception as exc:
                fallback_batch = [
                    fallback_papers_by_id[paper_id]
                    for paper_id in batch_ids
                    if paper_id in fallback_papers_by_id
                ]
                raw_papers.extend(fallback_batch)
                logger.warning(
                    f"arXiv API enrichment failed for batch starting at index {i}; "
                    f"using RSS metadata fallback for {len(fallback_batch)} papers. Error: {exc}"
                )
            bar.update(len(batch_ids))
        bar.close()

        keywords = self.config.source.arxiv.get("keywords")
        if keywords:
            before_filter = len(raw_papers)
            raw_papers = [paper for paper in raw_papers if self._match_keywords(paper)]
            logger.info(
                f"Filtered arXiv papers by keywords {list(keywords)}: "
                f"{len(raw_papers)}/{before_filter} papers kept"
            )

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult | ArxivFeedPaper) -> Paper:
        title = raw_paper.title
        authors = _extract_author_names(getattr(raw_paper, "authors", []))
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        if full_text is None:
            full_text = extract_text_from_tar(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult | ArxivFeedPaper) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult | ArxivFeedPaper) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult | ArxivFeedPaper) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
