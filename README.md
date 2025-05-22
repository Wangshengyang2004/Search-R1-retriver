# Search-R1-retriver

A fast GPU-accelerated semantic search service using FAISS and transformer models.

## Features
- GPU-accelerated FAISS indexing and search
- Support for multiple embedding models (E5, BGE, etc.)
- Optional reranking with cross-encoders
- Docker support with CUDA
- FastAPI server interface

## Prerequisites

Before building or running the service, run the pre-check script:

```bash
# Check prerequisites and download required data
./scripts/check_docker.sh
```

This will:
1. Check if Docker and NVIDIA drivers are installed
2. Verify Python environment
3. Download required data files if missing:
   - FAISS index (~4GB)
   - Wikipedia corpus (~2GB)

## Quick Start

1. **Build and Run with Docker:**
```bash
# Build the Docker image
docker build -t retriever-gpu -f docker/Dockerfile .

# Run with GPU support
docker run --gpus all -v ~/sr1_save:/data -p 8000:8000 retriever-gpu
```

2. **Use the API:**
```python
import requests

# Basic retrieval
response = requests.post("http://localhost:8000/retrieve", 
    json={
        "queries": ["What is Python?"],
        "topk_retrieval": 10
    }
)

# Retrieval with reranking
response = requests.post("http://localhost:8000/retrieve", 
    json={
        "queries": ["What is Python?"],
        "topk_retrieval": 10,
        "topk_rerank": 3,
        "return_scores": True
    }
)
```

## Directory Structure
```
Search-R1-retriver/
├── docker/               # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── retriever/           # Core package
│   ├── models/         # Model implementations
│   │   ├── dense.py   # FAISS retriever
│   │   ├── encoder.py # Text encoder
│   │   └── reranker.py # Cross-encoder reranker
│   ├── server/        # Server implementation
│   │   ├── app.py    # FastAPI server
│   │   └── config.py # Configuration
│   └── utils/        # Utility functions
└── scripts/           # Helper scripts
    ├── check_data.py  # Data verification
    └── check_docker.sh # Prerequisites check
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