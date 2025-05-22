FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY retriever/requirements_retriever.txt .
RUN pip install --no-cache-dir -r requirements_retriever.txt && \
    rm -rf ~/.cache/pip/*

# Copy source code
COPY retriever/ ./retriever/

# Create directories for data and models
RUN mkdir -p /data/index /data/corpus /data/models

# Set environment variables
ENV RETRIEVER_TYPE=ann \
    DATA_DIR=/data \
    INDEX_FILE=e5_Flat.index \
    CORPUS_FILE=wiki-18.jsonl \
    RETRIEVER_MODEL=intfloat/e5-base-v2 \
    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L12-v2 \
    PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "-m", "retriever.server.app", \
            "--index_path", "/data/index/e5_Flat.index", \
            "--corpus_path", "/data/corpus/wiki-18.jsonl", \
            "--retriever_model", "intfloat/e5-base-v2", \
            "--reranker_model", "cross-encoder/ms-marco-MiniLM-L12-v2"]
