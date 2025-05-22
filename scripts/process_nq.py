#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import datasets
import torch
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_nq_dataset(data_dir):
    """Load NQ dataset from json files."""
    nq_dir = Path(data_dir) / "nq_data"
    datasets_dict = {}
    
    for split in ['train', 'dev', 'test']:
        file_path = nq_dir / f"nq-{split}.json"
        with open(file_path) as f:
            data = json.load(f)
        datasets_dict[split] = data
        logger.info(f"Loaded {split} set with {len(data)} examples")
    
    return datasets_dict

def load_wiki_corpus(data_dir):
    """Load Wikipedia corpus."""
    corpus_path = Path(data_dir) / "corpus" / "wiki-18.jsonl"
    corpus = datasets.load_dataset(
        'json',
        data_files=str(corpus_path),
        split='train'
    )
    logger.info(f"Loaded corpus with {len(corpus)} documents")
    return corpus

def setup_retriever(model_name="intfloat/e5-base-v2"):
    """Initialize E5 model for retrieval."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer

def process_nq_examples(nq_data, corpus, model, tokenizer, split):
    """Process NQ examples with retrieved context."""
    processed_examples = []
    
    for example in tqdm(nq_data, desc=f"Processing {split} set"):
        # Extract question and answer
        question = example['question']
        answer = example['answer']
        
        # Create retrieval query
        query = f"query: {question}"
        inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            query_emb = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        
        # TODO: Implement retrieval using FAISS index
        # For now, we'll just use a placeholder
        context = "Retrieved context will go here"
        
        processed_example = {
            'question': question,
            'answer': answer,
            'context': context
        }
        processed_examples.append(processed_example)
    
    return processed_examples

def save_processed_data(processed_data, output_dir, split):
    """Save processed examples to disk."""
    output_path = Path(output_dir) / f"nq_{split}_processed.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    logger.info(f"Saved processed {split} data to {output_path}")

def main():
    # Set up paths
    data_dir = Path.home() / "sr1_save"
    output_dir = data_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    nq_datasets = load_nq_dataset(data_dir)
    corpus = load_wiki_corpus(data_dir)
    
    # Set up retriever
    logger.info("Setting up retriever...")
    model, tokenizer = setup_retriever()
    
    # Process each split
    for split, data in nq_datasets.items():
        logger.info(f"Processing {split} split...")
        processed_data = process_nq_examples(data, corpus, model, tokenizer, split)
        save_processed_data(processed_data, output_dir, split)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main() 