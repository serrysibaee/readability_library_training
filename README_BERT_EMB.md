# BERT Embedding Generator

A flexible command-line tool for generating BERT embeddings from text data. This tool supports multiple BERT models, different pooling strategies, and various input file formats.

## Features

- **Multiple File Formats**: Supports JSON, pickle (.pkl), CSV, and TSV files
- **Flexible Column Selection**: Choose which columns contain text, IDs, and labels
- **Multiple BERT Models**: Process data with different BERT models in one run
- **Pooling Strategies**: Choose between CLS token or mean pooling for sentence embeddings
- **Batch Processing**: Configurable batch sizes for efficient GPU utilization
- **GPU Support**: Automatically uses CUDA if available

## Installation

### Requirements

```bash
pip install torch transformers numpy pandas tqdm
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
torch>=1.9.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
```

## Usage

### Basic Usage

```bash
python bert_embedder.py -i input_file.csv -t "text_column" -o embeddings.pkl
```

### Full Command Line Options

```bash
python bert_embedder.py \
    --input_file data.csv \
    --text_column "sentence" \
    --output_file embeddings.pkl \
    --model_name "UBC-NLP/ARBERTv2" \
    --pooling_strategy cls \
    --id_column "id" \
    --label_column "label" \
    --batch_size 16 \
    --max_length 512
```

### Process Multiple Models

```bash
python bert_embedder.py \
    --input_file data.csv \
    --text_column "sentence" \
    --output_file embeddings.pkl \
    --models "UBC-NLP/ARBERTv2" "UBC-NLP/MARBERTv2" "aubmindlab/bert-base-arabertv02" \
    --batch_size 8
```

## Command Line Arguments

### Required Arguments
- `--input_file, -i`: Input file path (supports .json, .pkl, .csv, .tsv)
- `--text_column, -t`: Name of the column containing text data
- `--output_file, -o`: Output file path for saved embeddings (.pkl)

### Optional Arguments

#### Model Configuration
- `--model_name, -m`: BERT model name from HuggingFace (default: 'UBC-NLP/ARBERTv2')
- `--pooling_strategy, -p`: Pooling strategy - 'cls' or 'mean' (default: 'cls')
- `--models`: Process multiple models (space-separated list)

#### Data Configuration
- `--id_column`: Name of the column containing IDs (optional)
- `--label_column, -l`: Name of the column containing labels (optional)

#### Processing Configuration
- `--batch_size, -b`: Batch size for processing (default: 8)
- `--max_length`: Maximum sequence length for tokenization (default: 512)

## Input File Formats

### CSV/TSV Example
```csv
id,sentence,label
1,"This is a sample sentence",0
2,"Another example text",1
```

### JSON Example
```json
[
    {"ID": 1, "Sentence": "This is a sample sentence", "Label": 0},
    {"ID": 2, "Sentence": "Another example text", "Label": 1}
]
```

## Output Format

The tool saves embeddings as a pickle file containing a dictionary with:

```python
{
    'embeddings': numpy.ndarray,      # Shape: (n_samples, embedding_dim)
    'ids': List,                      # List of IDs
    'sentences': List[str],           # Original sentences
    'labels': List,                   # Labels (if provided)
    'model_name': str,                # Name of the BERT model used
    'pooling_strategy': str,          # Pooling strategy used
    'text_column': str,               # Name of text column
    'id_column': str,                 # Name of ID column
    'label_column': str               # Name of label column
}
```

## Loading Embeddings

```python
import pickle

# Load embeddings
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['embeddings']  # Feature matrix
y = data['labels']      # Labels (if available)
ids = data['ids']       # Sample IDs
sentences = data['sentences']  # Original text
```

Or use the built-in function:

```python
from bert_embedder import load_embeddings

data = load_embeddings('embeddings.pkl')
```

## Examples

### Example 1: Basic Text Classification Data
```bash
python bert_embedder.py \
    -i reviews.csv \
    -t "review_text" \
    -l "sentiment" \
    -o review_embeddings.pkl \
    -b 16
```

### Example 2: Multiple Arabic BERT Models
```bash
python bert_embedder.py \
    -i arabic_text.json \
    -t "text" \
    -o arabic_embeddings.pkl \
    --models "UBC-NLP/ARBERTv2" "UBC-NLP/MARBERTv2" "aubmindlab/bert-base-arabertv02" \
    -p mean \
    -b 32
```

### Example 3: Large Dataset with Custom Columns
```bash
python bert_embedder.py \
    -i large_dataset.tsv \
    -t "document_text" \
    --id_column "doc_id" \
    --label_column "category" \
    -o document_embeddings.pkl \
    -m "bert-base-uncased" \
    -b 64 \
    --max_length 256
```

## Pooling Strategies

- **CLS Token (`cls`)**: Uses the [CLS] token embedding as sentence representation
- **Mean Pooling (`mean`)**: Averages all token embeddings (excluding padding tokens)

## Supported BERT Models

The tool works with any BERT-compatible model from HuggingFace Hub. Popular choices include:

### Arabic Models
- `UBC-NLP/ARBERTv2`
- `UBC-NLP/MARBERTv2` 
- `aubmindlab/bert-base-arabertv02`

### English Models
- `bert-base-uncased`
- `bert-large-uncased`
- `distilbert-base-uncased`

### Multilingual Models
- `bert-base-multilingual-cased`
- `distilbert-base-multilingual-cased`

## Performance Tips

1. **Batch Size**: Increase batch size for faster processing on GPUs with more memory
2. **Max Length**: Reduce max_length for shorter texts to save memory and time
3. **Multiple Models**: When processing multiple models, they're processed sequentially to avoid memory issues
4. **GPU Memory**: The tool automatically clears GPU cache between models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size or max_length
2. **Column Not Found**: Check column names in your data file
3. **Model Loading Error**: Verify model name exists on HuggingFace Hub

### Memory Management

The tool automatically:
- Clears GPU cache between models
- Processes data in batches
- Removes models from memory after processing

## License

This tool is provided as-is for research and educational purposes.