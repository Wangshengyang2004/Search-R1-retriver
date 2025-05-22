#!/usr/bin/env python3
import os
import sys
import gzip
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# Set HuggingFace mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def print_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors[color]}{text}{colors['end']}")

def check_data_dir():
    """Check if data directories exist, create if not."""
    data_dir = Path.home() / "sr1_save"
    index_dir = data_dir / "index"
    corpus_dir = data_dir / "corpus"
    nq_dir = data_dir / "nq_data"
    
    for dir_path in [data_dir, index_dir, corpus_dir, nq_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print_color(f"Created directory: {dir_path}", "blue")
    return data_dir

def download_index(save_dir):
    """Download and combine index files."""
    print_color("\nDownloading index files...", "blue")
    repo_id = "PeterJinGo/wiki-18-e5-index"
    index_parts = []
    
    # Download parts
    for file in ["part_aa", "part_ab"]:
        part_path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset",
            local_dir=save_dir
        )
        index_parts.append(part_path)
        print_color(f"Downloaded {file}", "green")
    
    # Combine parts into index file
    print_color("\nMerging index parts...", "blue")
    index_path = Path(save_dir) / "index" / "e5_Flat.index"
    with open(index_path, 'wb') as outfile:
        for part in index_parts:
            with open(part, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    
    # Clean up parts
    for part in index_parts:
        os.remove(part)
    
    print_color(f"Created index file at: {index_path}", "green")
    return index_path

def download_corpus(save_dir):
    """Download and extract corpus file."""
    print_color("\nDownloading corpus file...", "blue")
    repo_id = "PeterJinGo/wiki-18-corpus"
    
    # Download gzipped file
    gz_path = hf_hub_download(
        repo_id=repo_id,
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=save_dir
    )
    
    # Extract file
    print_color("\nExtracting corpus file...", "blue")
    corpus_path = Path(save_dir) / "corpus" / "wiki-18.jsonl"
    with gzip.open(gz_path, 'rb') as f_in:
        with open(corpus_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Clean up gzip file
    os.remove(gz_path)
    
    print_color(f"Created corpus file at: {corpus_path}", "green")
    return corpus_path

def download_nq_dataset(save_dir):
    """Download and process Natural Questions dataset."""
    print_color("\nDownloading Natural Questions dataset...", "blue")
    
    # Download NQ dataset files
    nq_dir = Path(save_dir) / "nq_data"
    files_to_download = {
        "train": "nq-train.json",
        "dev": "nq-dev.json",
        "test": "nq-test.json"
    }
    
    for split, filename in files_to_download.items():
        try:
            file_path = hf_hub_download(
                repo_id="PeterJinGo/nq-dataset",
                filename=filename,
                repo_type="dataset",
                local_dir=nq_dir
            )
            print_color(f"Downloaded {filename}", "green")
        except Exception as e:
            print_color(f"Error downloading {filename}: {str(e)}", "red")
            return False
    
    return True

def check_files():
    """Check if required files exist and have correct sizes."""
    data_dir = Path.home() / "sr1_save"
    index_path = data_dir / "index" / "e5_Flat.index"
    corpus_path = data_dir / "corpus" / "wiki-18.jsonl"
    nq_dir = data_dir / "nq_data"
    
    files_ok = True
    
    # Check index file
    if not index_path.exists():
        print_color(f"Missing index file: {index_path}", "red")
        files_ok = False
    elif index_path.stat().st_size < 1_000_000:  # At least 1MB
        print_color(f"Index file seems too small: {index_path}", "red")
        files_ok = False
        
    # Check corpus file
    if not corpus_path.exists():
        print_color(f"Missing corpus file: {corpus_path}", "red")
        files_ok = False
    elif corpus_path.stat().st_size < 1_000_000:  # At least 1MB
        print_color(f"Corpus file seems too small: {corpus_path}", "red")
        files_ok = False
    
    # Check NQ dataset files
    nq_files = ["nq-train.json", "nq-dev.json", "nq-test.json"]
    for file in nq_files:
        file_path = nq_dir / file
        if not file_path.exists():
            print_color(f"Missing NQ dataset file: {file}", "red")
            files_ok = False
        
    return files_ok

def main():
    print_color("\n=== Checking Search-R1 Data Files ===", "blue")
    
    # Check/create directories
    data_dir = check_data_dir()
    
    # Check if files exist and are valid
    if check_files():
        print_color("\n✓ All required files are present!", "green")
        return 0
        
    print_color("\nSome required files are missing. Would you like to download them? [y/N]", "yellow")
    response = input().lower()
    
    if response != 'y':
        print_color("\nPlease download the required files manually:", "yellow")
        print("1. Index file should be at: ~/sr1_save/index/e5_Flat.index")
        print("2. Corpus file should be at: ~/sr1_save/corpus/wiki-18.jsonl")
        print("3. NQ dataset files should be in: ~/sr1_save/nq_data/")
        return 1
        
    try:
        # Download all required files
        download_index(data_dir)
        download_corpus(data_dir)
        download_nq_dataset(data_dir)
        
        if check_files():
            print_color("\n✓ All files downloaded successfully!", "green")
            print_color("\nNext steps:", "blue")
            print("1. Build the Docker image:")
            print("   docker build -t retriever-gpu -f docker/Dockerfile .")
            print("\n2. Process the NQ dataset:")
            print("   python scripts/process_nq.py")
            return 0
        else:
            print_color("\n✗ Some files failed to download correctly.", "red")
            return 1
            
    except Exception as e:
        print_color(f"\n✗ Error during download: {str(e)}", "red")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 