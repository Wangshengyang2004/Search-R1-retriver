from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder
import torch
import numpy as np

class BaseCrossEncoder:
    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)

    def _passage_to_string(self, doc_item):
        if "document" not in doc_item:
            content = doc_item['contents']
        else:
            content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])

        return f"(Title: {title}) {text}"

    def rerank(self, 
               queries: List[str], 
               documents: List[List[Dict]]):
        """
        Assume documents is a list of list of dicts, where each dict is a document with keys "id" and "contents".
        This assumption is made to be consistent with the output of the retrieval server.
        """ 
        assert len(queries) == len(documents)

        pairs = []
        qids = []
        for qid, query in enumerate(queries):
            for document in documents:
                for doc_item in document:
                    doc = self._passage_to_string(doc_item)
                    pairs.append((query, doc))
                    qids.append(qid)

        scores = self._predict(pairs)
        query_to_doc_scores = defaultdict(list)

        assert len(scores) == len(pairs) == len(qids)
        for i in range(len(pairs)):
            query, doc = pairs[i]
            score = scores[i] 
            qid = qids[i]
            query_to_doc_scores[qid].append((doc, score))

        sorted_query_to_doc_scores = {}
        for query, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return sorted_query_to_doc_scores

    def _predict(self, pairs: List[tuple[str, str]]):
        raise NotImplementedError 

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        raise NotImplementedError


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, model, batch_size=32, device="cuda"):
        super().__init__(model, batch_size, device)

    def _predict(self, pairs: List[tuple[str, str]]):
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = scores.tolist() if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray) else scores
        return scores

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        model = CrossEncoder(model_name_or_path)
        return cls(model, **kwargs)


@dataclass 
class RerankerConfig:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")


def get_reranker(config: RerankerConfig):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}") 