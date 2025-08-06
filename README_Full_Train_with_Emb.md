# ML Pipeline for Text Classification

A comprehensive command-line tool for text classification using sentence transformers and neural networks. This pipeline handles everything from embedding generation to model training and evaluation.

## Features

- **End-to-End Pipeline**: Complete workflow from raw text to trained models
- **Multiple Models**: Test multiple sentence transformer models in one run
- **Flexible Data Formats**: Supports JSON, pickle, CSV, and TSV files
- **Configurable Architecture**: Customizable neural network architectures
- **Advanced Training**: Early stopping, learning rate scheduling, gradient clipping
- **Comprehensive Evaluation**: Multiple metrics including QWK, accuracy variants
- **GPU Support**: Automatic CUDA utilization with memory management
- **Configuration Management**: YAML config files with CLI override support
- **Detailed Reporting**: Comprehensive results and training history

## Installation

### Requirements

```bash
pip install torch transformers sentence-transformers scikit-learn matplotlib seaborn pandas numpy tqdm pyyaml
```

### Requirements.txt
```txt
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.62.0
pyyaml>=6.0
```

## Quick Start

### 1. Create Default Configuration
```bash
python ml_pipeline.py --create_config
```

This creates `config.yaml` with default settings.

### 2. Basic Usage
```bash
python ml_pipeline.py \
    --train_data train.csv \
    --test_data test.csv \
    --text_column "sentence" \
    --label_column "label" \
    --models "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 3. Using Configuration File
```bash
python ml_pipeline.py --config config.yaml
```

## Command Line Arguments

### Configuration
- `--config, -c`: Path to YAML configuration file
- `--create_config`: Create a default configuration file

### Data Arguments
- `--train_data`: Path to training data file
- `--test_data`: Path to test data file
- `--text_column`: Name of text column (default: 'Sentence')
- `--id_column`: Name of ID column (optional)
- `--label_column`: Name of label column (default: 'Readability_Level_19')

### Model Arguments
- `--models`: List of sentence transformer models to use
- `--embedding_batch_size`: Batch size for embedding generation (default: 1024)

### Training Arguments
- `--hidden_layers`: Hidden layer sizes (default: [512, 256, 128])
- `--dropout_rate`: Dropout rate (default: 0.3)
- `--learning_rate`: Learning rate (default: 0.001)
- `--batch_size`: Training batch size (default: 128)
- `--epochs`: Maximum epochs (default: 100)
- `--early_stopping_patience`: Early stopping patience (default: 10)

### Output Arguments
- `--embeddings_dir`: Directory to save embeddings (default: 'embeddings')
- `--models_dir`: Directory to save trained models (default: 'models')
- `--output_dir`: Directory to save results (default: 'results')
- `--no_plots`: Disable saving plots

## Configuration File Format

Create a YAML configuration file for complex setups:

```yaml
# Data configuration
train_data_path: "data/train.pkl"
test_data_path: "data/test.pkl"
text_column: "Sentence"
id_column: "ID"
label_column: "Readability_Level_19"

# Models to evaluate
models_to_use:
  - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  - "sentence-transformers/LaBSE"
  - "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"

# Directory settings
embeddings_dir: "embeddings"
models_dir: "models"
output_dir: "results"

# Embedding settings
embedding_batch_size: 2048

# Training configuration
training_config:
  # Neural network architecture
  hidden_layers: [1024, 512, 256]
  dropout_rate: 0.4
  
  # Training parameters
  learning_rate: 0.001
  batch_size: 256
  epochs: 200
  weight_decay: 0.00001
  
  # Training control
  early_stopping_patience: 15
  scheduler_patience: 5
  print_every: 20

# General settings
save_plots: true
```

## Pipeline Stages

### Stage 1: Embedding Generation
- Loads data from specified files
- Generates embeddings using sentence transformers
- Saves embeddings for reuse
- Supports batched processing for memory efficiency

### Stage 2: Neural Network Training
- Loads generated embeddings
- Preprocesses data (normalization, label encoding)
- Creates configurable neural network
- Trains with advanced features:
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
  - Batch normalization

### Stage 3: Model Evaluation
- Tests trained models on test data
- Computes comprehensive metrics
- Saves predictions and results
- Generates evaluation reports

## Output Structure

```
results/
├── final_results_20240101_120000.csv
├── embeddings/
│   ├── train_model1_embeddings.pkl
│   └── test_model1_embeddings.pkl
└── models/
    └── neural_net_20240101_120000_model1/
        ├── best_model.pth
        ├── best_training_artifacts.pkl
        ├── training_history.png
        ├── predictions_model1.csv
        └── evaluation_metrics.json
```

## Supported Models

The pipeline works with any sentence transformer model from Hugging Face Hub:

### Multilingual Models
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/LaBSE`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### Arabic Models
- `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2`
- `sentence-transformers/paraphrase-arabic-mt5-base`

### English Models  
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

### Advanced Models
- `jinaai/jina-embeddings-v3`
- `Snowflake/snowflake-arctic-embed-m-v2.0`

## Examples

### Example 1: Quick Classification Task
```bash
python ml_pipeline.py \
    --train_data reviews_train.csv \
    --test_data reviews_test.csv \
    --text_column "review_text" \
    --label_column "sentiment" \
    --models "sentence-transformers/all-MiniLM-L6-v2" \
    --epochs 50 \
    --batch_size 64
```

### Example 2: Arabic Text Classification
```bash
python ml_pipeline.py \
    --train_data arabic_train.pkl \
    --test_data arabic_test.pkl \
    --text_column "text" \
    --label_column "category" \
    --models \
        "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2" \
        "sentence-transformers/LaBSE" \
    --hidden_layers 1024 512 256 \
    --learning_rate 0.0005 \
    --embedding_batch_size 512
```

### Example 3: Large-Scale Experiment
```bash
python ml_pipeline.py --config large_experiment.yaml
```

With `large_experiment.yaml`:
```yaml
train_data_path: "large_dataset_train.pkl"
test_data_path: "large_dataset_test.pkl"
text_column: "document_text"
label_column: "difficulty_level"

models_to_use:
  - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  - "sentence-transformers/LaBSE"
  - "sentence-transformers/all-mpnet-base-v2"
  - "jinaai/jina-embeddings-v3"

embedding_batch_size: 4096

training_config:
  hidden_layers: [2048, 1024, 512, 256]
  dropout_rate: 0.5
  learning_rate: 0.0001
  batch_size: 512
  epochs: 300
  early_stopping_patience: 20
```

## Evaluation Metrics

The pipeline computes several metrics:

- **Accuracy**: Exact match accuracy
- **Accuracy ±1**: Predictions within 1 class of true label
- **Accuracy ±3**: Predictions within 3 classes of true label
- **Average Absolute Distance**: Mean absolute error between predictions and true labels
- **QWK (Quadratic Weighted Kappa)**: Primary metric for ranking models

## Advanced Features

### Memory Management
- Automatic GPU memory clearing between models
- Efficient batch processing
- Model cleanup after processing

### Training Optimization
- Xavier weight initialization
- Gradient clipping (max norm = 1.0)
- ReduceLROnPlateau scheduler
- Early stopping based on training loss

### Extensibility
- Easy to add new models
- Configurable neural network architectures
- Custom metrics can be added to ModelTester class

## Performance Tips

1. **Batch Sizes**: 
   - Increase embedding_batch_size for faster embedding generation
   - Increase training batch_size for faster training (if GPU memory allows)

2. **Model Selection**:
   - Smaller models (MiniLM) for speed
   - Larger models (mpnet, jina) for better quality

3. **Hardware Optimization**:
   - Use GPU for significant speedup
   - Consider model size vs. available GPU memory

4. **Data Preprocessing**:
   - Clean text data beforehand for better results
   - Ensure consistent label encoding

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch sizes
   --embedding_batch_size 512 --batch_size 64
   ```

2. **Model Loading Errors**:
   - Verify model name exists on Hugging Face Hub
   - Check internet connection for model download

3. **Data Loading Issues**:
   - Verify column names match your data
   - Check file format and encoding

4. **Poor Performance**:
   - Try different model architectures
   - Adjust learning rate and dropout
   - Increase training epochs

### Memory Management
The pipeline automatically:
- Clears GPU cache between models
- Removes models from memory after processing
- Uses efficient batch processing

## License

This tool is provided for research and educational purposes. Please check the licenses of individual models used.

## Contributing

To extend the pipeline:
1. Add new metrics in `ModelTester.compute_metrics()`
2. Customize neural network architectures in `MultiLabelNN`
3. Add new data loaders in `SentenceTransformerEmbedder.load_data()`
4. Extend configuration options in `create_default_config()`