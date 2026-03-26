from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
import re


class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        candidate_texts = [self._build_ranking_text(c.title, c.abstract) for c in candidates]
        corpus_texts = [self._build_ranking_text(c.title, c.abstract) for c in corpus]
        semantic_sim = self.get_similarity_score(candidate_texts, corpus_texts)
        lexical_sim = self._get_lexical_overlap_score(candidate_texts, corpus_texts)
        assert semantic_sim.shape == (len(candidates), len(corpus))
        assert lexical_sim.shape == (len(candidates), len(corpus))

        scoring_config = self.config.reranker.get("scoring", {})
        semantic_weight = scoring_config.get("semantic_weight", 0.85)
        lexical_weight = scoring_config.get("lexical_weight", 0.15)
        neighbor_top_k = min(scoring_config.get("neighbor_top_k", 8), len(corpus))
        top_k_weight = scoring_config.get("top_k_weight", 0.7)

        combined_sim = semantic_weight * semantic_sim + lexical_weight * lexical_sim
        global_score = (combined_sim * time_decay_weight).sum(axis=1)
        if neighbor_top_k > 0:
            top_k_indices = np.argpartition(combined_sim, -neighbor_top_k, axis=1)[:, -neighbor_top_k:]
            top_k_sim = np.take_along_axis(combined_sim, top_k_indices, axis=1)
            top_k_time_weight = time_decay_weight[top_k_indices]
            top_k_time_weight = top_k_time_weight / top_k_time_weight.sum(axis=1, keepdims=True)
            top_k_score = (top_k_sim * top_k_time_weight).sum(axis=1)
        else:
            top_k_score = global_score

        scores = (top_k_weight * top_k_score + (1 - top_k_weight) * global_score) * 10
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    @staticmethod
    def _build_ranking_text(title: str, abstract: str) -> str:
        title = (title or "").strip()
        abstract = (abstract or "").strip()
        if title and abstract:
            return f"Title: {title}\nAbstract: {abstract}"
        return title or abstract

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9][a-z0-9\\-]{1,}", text.lower())
            if len(token) >= 3
        }

    def _get_lexical_overlap_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        s1_tokens = [self._tokenize(text) for text in s1]
        s2_tokens = [self._tokenize(text) for text in s2]
        sim = np.zeros((len(s1_tokens), len(s2_tokens)))
        for i, tokens_1 in enumerate(s1_tokens):
            for j, tokens_2 in enumerate(s2_tokens):
                if not tokens_1 or not tokens_2:
                    continue
                intersection = len(tokens_1 & tokens_2)
                if intersection == 0:
                    continue
                sim[i, j] = 2 * intersection / (len(tokens_1) + len(tokens_2))
        return sim
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
