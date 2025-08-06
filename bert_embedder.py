#!/usr/bin/env python3
"""
BERT Embedding Generator

A flexible command-line tool for generating BERT embeddings from text data.
Supports multiple BERT models, different pooling strategies, and various input formats.
"""

import json
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import os
import argparse
import pandas as pd
from typing import List, Dict, Any, Union
import sys

class BERTEmbedder:
    def __init__(self, model_name: str, pooling_strategy: str = "cls"):
        """
        Initialize the BERT embedder

        Args:
            model_name (str): Name of the BERT model to use
            pooling_strategy (str): Pooling strategy ('cls' or 'mean')
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        print(f"Pooling strategy: {pooling_strategy}")

        try:
            # Load model directly
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            sys.exit(1)

    def load_data(self, file_path: str, text_column: str, id_column: str = None, 
                  label_column: str = None) -> List[Dict[str, Any]]:
        """
        Load data from various file formats
        
        Args:
            file_path (str): Path to the input file
            text_column (str): Name of the column containing text
            id_column (str): Name of the column containing IDs (optional)
            label_column (str): Name of the column containing labels (optional)
            
        Returns:
            List[Dict]: List of dictionaries containing the data
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_extension == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif file_extension in ['.csv', '.tsv']:
                separator = ',' if file_extension == '.csv' else '\t'
                df = pd.read_csv(file_path, sep=separator)
                data = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
            sys.exit(1)

        # Validate required columns
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            if text_column not in sample:
                print(f"Error: Text column '{text_column}' not found in data.")
                print(f"Available columns: {list(sample.keys())}")
                sys.exit(1)

        print(f"Loaded {len(data)} samples from {file_path}")
        return data

    def get_sentence_embedding(self, sentence: str, max_length: int = 512) -> np.ndarray:
        """
        Get embedding for a single sentence using BERT

        Args:
            sentence (str): Input sentence
            max_length (int): Maximum sequence length

        Returns:
            np.ndarray: Sentence embedding
        """
        # Tokenize the sentence
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model.bert(**inputs)

            if self.pooling_strategy == "cls":
                # Use [CLS] token embedding as sentence representation
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.pooling_strategy == "mean":
                # Use mean pooling of all tokens
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return embedding.squeeze()

    def generate_embeddings(self, data: List[Dict], text_column: str, 
                          id_column: str = None, label_column: str = None,
                          batch_size: int = 8, max_length: int = 512) -> Dict[str, Any]:
        """
        Generate embeddings for all sentences in the dataset

        Args:
            data (List[Dict]): List of dictionaries containing the data
            text_column (str): Name of the column containing text
            id_column (str): Name of the column containing IDs
            label_column (str): Name of the column containing labels
            batch_size (int): Batch size for processing
            max_length (int): Maximum sequence length

        Returns:
            Dict: Dictionary containing embeddings, IDs, sentences, and labels
        """
        sentences = [item[text_column] for item in data]
        ids = [item.get(id_column, i) for i, item in enumerate(data)] if id_column else list(range(len(data)))
        labels = [item.get(label_column) for item in data] if label_column else [None] * len(data)

        embeddings = []

        print("Generating BERT embeddings...")

        # Process in batches
        for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
            batch_sentences = sentences[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_sentences,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model.bert(**inputs)

                if self.pooling_strategy == "cls":
                    # Use [CLS] token embeddings
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif self.pooling_strategy == "mean":
                    # Use mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        return {
            'embeddings': embeddings,
            'ids': ids,
            'sentences': sentences,
            'labels': labels,
            'model_name': self.model_name,
            'pooling_strategy': self.pooling_strategy,
            'text_column': text_column,
            'id_column': id_column,
            'label_column': label_column
        }

    def save_embeddings(self, embeddings_data: Dict[str, Any], output_path: str):
        """Save embeddings to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_data, f)

        print(f"Embeddings saved to: {output_path}")
        print(f"Embedding shape: {embeddings_data['embeddings'].shape}")
        print(f"Model used: {embeddings_data['model_name']}")
        print(f"Pooling strategy: {embeddings_data['pooling_strategy']}")

def load_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """
    Load saved embeddings for use in training/testing

    Args:
        embeddings_path (str): Path to saved embeddings file

    Returns:
        Dict: Dictionary containing embeddings and metadata
    """
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)

    print(f"Loaded embeddings: {embeddings_data['embeddings'].shape}")
    print(f"Model used: {embeddings_data['model_name']}")
    print(f"Pooling strategy: {embeddings_data.get('pooling_strategy', 'cls')}")

    return embeddings_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate BERT embeddings from text data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input_file', '-i', required=True, 
                       help='Input file path (supports .json, .pkl, .csv, .tsv)')
    parser.add_argument('--text_column', '-t', required=True, 
                       help='Name of the column containing text data')
    parser.add_argument('--output_file', '-o', required=True, 
                       help='Output file path for saved embeddings (.pkl)')
    
    # Model arguments
    parser.add_argument('--model_name', '-m', default='UBC-NLP/ARBERTv2',
                       help='BERT model name from HuggingFace')
    parser.add_argument('--pooling_strategy', '-p', choices=['cls', 'mean'], default='cls',
                       help='Pooling strategy for sentence embeddings')
    
    # Data arguments
    parser.add_argument('--id_column', help='Name of the column containing IDs (optional)')
    parser.add_argument('--label_column', '-l', help='Name of the column containing labels (optional)')
    
    # Processing arguments
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for tokenization')
    
    # Multiple models
    parser.add_argument('--models', nargs='+',
                       help='Process multiple models (space-separated list)')
    
    args = parser.parse_args()
    
    # Determine which models to process
    models_to_process = args.models if args.models else [args.model_name]
    
    # Load data once
    print("Loading dataset...")
    embedder_temp = BERTEmbedder(models_to_process[0])
    data = embedder_temp.load_data(args.input_file, args.text_column, 
                                  args.id_column, args.label_column)
    del embedder_temp
    
    # Process each model
    for model_name in models_to_process:
        try:
            print(f"\n{'='*60}")
            print(f"Processing with model: {model_name}")
            print(f"{'='*60}")
            
            # Initialize embedder
            embedder = BERTEmbedder(model_name, args.pooling_strategy)
            
            # Generate embeddings
            embeddings_data = embedder.generate_embeddings(
                data,
                text_column=args.text_column,
                id_column=args.id_column,
                label_column=args.label_column,
                batch_size=args.batch_size,
                max_length=args.max_length
            )
            
            # Determine output path
            if len(models_to_process) > 1:
                # Multiple models: modify filename to include model name
                base_name, ext = os.path.splitext(args.output_file)
                model_short_name = model_name.split('/')[-1]
                output_path = f"{base_name}_{model_short_name}{ext}"
            else:
                output_path = args.output_file
            
            # Save embeddings
            embedder.save_embeddings(embeddings_data, output_path)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete model to free memory
            del embedder
            
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            continue
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    main()