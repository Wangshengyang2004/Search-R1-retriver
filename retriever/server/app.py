import argparse
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ..models.dense import DenseRetriever
from ..models.reranker import get_reranker
from .config import RetrieverConfig, RerankerConfig, ServerConfig

app = FastAPI(title="Search-R1 Retriever Service")

class QueryRequest(BaseModel):
    queries: List[str]
    topk_retrieval: Optional[int] = None
    topk_rerank: Optional[int] = None
    return_scores: bool = False

def convert_title_format(text):
    if '"' not in text.split("\n")[0]:
        title = text.split("\n")[0]
        content = "\n".join(text.split("\n")[1:])
        return f'"{title}"\n{content}'
    return text

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval with optional reranking.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk_retrieval": 10,  # Optional, defaults to config value
      "topk_rerank": 3,      # Optional, defaults to config value
      "return_scores": true   # Optional, defaults to false
    }
    """
    # Use request params or fall back to config defaults
    topk_retrieval = request.topk_retrieval or server_config.retriever.retrieval_topk
    
    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=topk_retrieval,
        return_score=True
    )
    
    # Perform reranking if configured
    if server_config.reranker:
        topk_rerank = request.topk_rerank or server_config.reranker.rerank_topk
        reranked = reranker.rerank(request.queries, results)
        
        # Format reranked response
        resp = []
        for i, doc_scores in reranked.items():
            doc_scores = doc_scores[:topk_rerank]
            if request.return_scores:
                combined = []
                for doc, score in doc_scores:
                    combined.append({"document": convert_title_format(doc), "score": score})
                resp.append(combined)
            else:
                resp.append([convert_title_format(doc) for doc, _ in doc_scores])
    else:
        # Format retrieval-only response
        resp = []
        for i, single_result in enumerate(results):
            if request.return_scores:
                combined = []
                for doc, score in zip(single_result, scores[i]):
                    combined.append({"document": convert_title_format(doc['contents']), "score": score})
                resp.append(combined)
            else:
                resp.append([convert_title_format(doc['contents']) for doc in single_result])
    
    return {"result": resp}

def main():
    parser = argparse.ArgumentParser(description="Launch the retriever service.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of retriever model")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path to retriever model")
    parser.add_argument("--retrieval_topk", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--reranker_model", type=str, help="Path to reranker model")
    parser.add_argument("--rerank_topk", type=int, default=3, help="Number of documents after reranking")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    # Create configurations
    retriever_config = RetrieverConfig(
        retrieval_method=args.retriever_name,
        retrieval_topk=args.retrieval_topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_model_path=args.retriever_model
    )

    reranker_config = RerankerConfig(
        rerank_model_name_or_path=args.reranker_model,
        rerank_topk=args.rerank_topk
    ) if args.reranker_model else None

    global server_config, retriever, reranker
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        retriever=retriever_config,
        reranker=reranker_config
    )

    # Initialize retriever and reranker
    retriever = DenseRetriever(server_config.retriever)
    reranker = get_reranker(server_config.reranker) if server_config.reranker else None

    # Start server
    uvicorn.run(app, host=server_config.host, port=server_config.port)

if __name__ == "__main__":
    main() 