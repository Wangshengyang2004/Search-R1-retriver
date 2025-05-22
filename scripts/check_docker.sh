#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Checking Docker Prerequisites ===${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
else
    echo -e "${GREEN}✓ Docker is installed${NC}"
fi

# Check if nvidia-docker is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA drivers not found${NC}"
    echo "Please install NVIDIA drivers and nvidia-docker2"
    exit 1
else
    echo -e "${GREEN}✓ NVIDIA drivers found${NC}"
fi

# Check if required Python packages are installed
echo -e "\n${BLUE}=== Checking Python Environment ===${NC}"
if ! command -v pip &> /dev/null; then
    echo -e "${RED}✗ pip is not installed${NC}"
    echo "Please install Python and pip"
    exit 1
fi

# Install huggingface_hub if not present
if ! pip show huggingface_hub &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install --quiet huggingface_hub
fi

# Check data files
echo -e "\n${BLUE}=== Checking Data Files ===${NC}"
python scripts/check_data.py
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Data file check failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All prerequisites met!${NC}"
echo -e "${BLUE}You can now build the Docker image:${NC}"
echo "docker build -t retriever-gpu -f docker/Dockerfile ." 