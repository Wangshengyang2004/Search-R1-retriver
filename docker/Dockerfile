FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Create conda environment with specific Python version
RUN conda create -n retriever python=3.10 -y

# Set up shell to use conda environment by default
SHELL ["conda", "run", "-n", "retriever", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN conda install -y -c pytorch -c nvidia \
    pytorch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    pytorch-cuda=12.1

# Install FAISS-GPU
RUN conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0

# Install other Python packages
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    pyserini \
    uvicorn \
    fastapi \
    huggingface_hub

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Make sure scripts are executable
RUN chmod +x scripts/*.py

# Set the default command to activate conda environment
SHELL ["conda", "run", "-n", "retriever", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "retriever"]
CMD ["/bin/bash"]
