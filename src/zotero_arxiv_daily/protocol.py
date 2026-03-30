from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    @staticmethod
    def _truncate_text(text: str, max_tokens: int) -> str:
        enc = tiktoken.encoding_for_model("gpt-4o")
        text_tokens = enc.encode(text)
        text_tokens = text_tokens[:max_tokens]
        return enc.decode(text_tokens)

    def _build_tldr_context(self) -> str:
        context_parts = []
        if self.title:
            context_parts.append(f"Title:\n{self.title}")
        if self.abstract:
            context_parts.append(f"Abstract:\n{self.abstract}")

        if self.full_text:
            normalized_text = re.sub(r"\s+", " ", self.full_text)
            key_sentences = re.findall(
                r"([^.?!\n]{0,80}\b("
                r"we propose|we introduce|we present|we develop|we study|"
                r"our method|our framework|our approach|this paper|"
                r"results show|experiments show|we demonstrate|we show|"
                r"outperform|improves|improvement|contribution|contributions"
                r")\b[^.?!\n]{0,240}[.?!])",
                normalized_text,
                flags=re.IGNORECASE,
            )
            excerpt = " ".join(sentence for sentence, _ in key_sentences[:8]).strip()
            if not excerpt:
                excerpt = normalized_text[:4000]
            context_parts.append(
                "Focused excerpts from the paper body:\n"
                f"{self._truncate_text(excerpt, 1200)}"
            )
        return "\n\n".join(context_parts)

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = (
            f"Read the paper metadata and excerpts below, then produce a compact but information-rich summary in {lang}.\n\n"
            "Requirements:\n"
            "- Output exactly 3 lines.\n"
            "- Line 1 starts with 'Problem: ' and states the task or challenge.\n"
            "- Line 2 starts with 'Method: ' and states the core method or technical idea.\n"
            "- Line 3 starts with 'Finding: ' and states the main empirical result, practical takeaway, or intended benefit.\n"
            "- Each line should be concise but specific.\n"
            "- Avoid generic phrases such as 'this paper presents'.\n"
            "- Do not mention unavailable details or speculate.\n"
            "- Output only the final 3 lines.\n\n"
        )
        prompt += self._build_tldr_context()
        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        prompt = self._truncate_text(prompt, 4000)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize scientific papers for expert readers. "
                        f"Return exactly 3 lines in {lang} with the prefixes 'Problem:', 'Method:', and 'Finding:'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = (response.choices[0].message.content or "").strip()
        if not tldr:
            raise ValueError("Empty summary returned by LLM")
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.affiliations is not None:
            return self.affiliations
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
