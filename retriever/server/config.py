from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""
    retrieval_method: str = field(default="e5")
    retrieval_topk: int = field(default=10)
    index_path: str = field(default="./index/e5_Flat.index")
    corpus_path: str = field(default="./data/corpus.jsonl")
    faiss_gpu: bool = field(default=True)
    retrieval_model_path: str = field(default="intfloat/e5-base-v2")
    retrieval_pooling_method: str = field(default="mean")
    retrieval_query_max_length: int = field(default=256)
    retrieval_use_fp16: bool = field(default=True)
    retrieval_batch_size: int = field(default=512)

@dataclass
class RerankerConfig:
    """Configuration for the reranker."""
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

@dataclass
class ServerConfig:
    """Configuration for the server."""
    host: str = field(default="0.0.0.0")
    port: int = field(default=8000)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: Optional[RerankerConfig] = field(default=None) 