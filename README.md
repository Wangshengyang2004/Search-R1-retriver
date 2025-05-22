# Search-R1-retriver

A fast GPU-accelerated semantic search service using FAISS and transformer models.

## Features
- GPU-accelerated FAISS indexing and search
- Support for multiple embedding models (E5, BGE, etc.)
- Optional reranking with cross-encoders
- Docker support with CUDA
- FastAPI server interface

## Directory Structure
```
Search-R1-retriver/
├── retriever/              # Core retriever implementation
│   ├── models/            # Model implementations
│   │   ├── dense.py      # Dense retriever (FAISS)
│   │   ├── encoder.py    # Text encoders
│   │   └── reranker.py   # Cross-encoder reranker
│   ├── server/           # Server implementations
│   │   ├── app.py        # FastAPI server
│   │   └── config.py     # Server configuration
│   └── utils/            # Utility functions
├── scripts/               # Helper scripts
│   ├── build_index.py    # Index building script
│   └── download.py       # Data download script
├── docker/               # Docker related files
│   ├── Dockerfile       
│   └── docker-compose.yml
└── examples/             # Example usage
```

## Quick Start

1. **Build and Run with Docker:**
```bash
# Build the Docker image
docker build -t retriever-gpu -f docker/Dockerfile .

# Run with GPU support
docker run --gpus all -v ~/sr1_save:/data/corpus -p 8000:8000 retriever-gpu
```

2. **Build Index:**
```bash
python scripts/build_index.py \
  --corpus_path /path/to/corpus.jsonl \
  --save_dir /path/to/save \
  --retriever_name e5 \
  --model_path intfloat/e5-base-v2 \
  --faiss_type Flat \
  --faiss_gpu
```

3. **Run Server:**
```bash
python -m retriever.server.app \
  --index_path /path/to/index \
  --corpus_path /path/to/corpus.jsonl \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --reranker_model cross-encoder/ms-marco-MiniLM-L12-v2
```

## API Usage

### Basic Retrieval
```python
import requests

response = requests.post("http://localhost:8000/retrieve", 
    json={
        "queries": ["What is Python?"],
        "topk": 3
    }
)
```

### Retrieval with Reranking
```python
response = requests.post("http://localhost:8000/retrieve", 
    json={
        "queries": ["What is Python?"],
        "topk_retrieval": 10,
        "topk_rerank": 3
    }
)
```

## Configuration

The service can be configured through environment variables or command line arguments:

- `RETRIEVER_TYPE`: Type of retriever (e5, bge, etc.)
- `DATA_DIR`: Directory containing corpus and index files
- `INDEX_FILE`: Name of the FAISS index file
- `RETRIEVER_MODEL`: HuggingFace model path for embeddings
- `RERANKER_MODEL`: Cross-encoder model for reranking

## License

Apache License 2.0 