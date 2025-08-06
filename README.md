# ArabicNLP2025 Library: Unified Overview

This repository provides a modular toolkit for Arabic and multilingual NLP, focusing on embedding generation, neural network training, and end-to-end text classification pipelines. Below is a summary of the main components and their usage.

---

## 1. BERT Embedding Generator (`bert_embedder.py`)
- **Purpose:** Generate BERT embeddings from text data using various models and pooling strategies.
- **Features:**
  - Supports JSON, pickle, CSV, TSV input
  - Flexible column selection (text, ID, label)
  - Multiple BERT models in one run
  - CLS/mean pooling, batch processing, GPU support
- **Usage Example:**
  ```bash
  python bert_embedder.py -i input.csv -t "text_column" -o embeddings.pkl
  # For multiple models:
  python bert_embedder.py --input_file data.csv --text_column "sentence" --output_file embeddings.pkl --models "UBC-NLP/ARBERTv2" "UBC-NLP/MARBERTv2"
  ```
- **Output:** Pickle file with embeddings, IDs, sentences, labels, and metadata.
- **Loading Embeddings:**
  ```python
  from bert_embedder import load_embeddings
  data = load_embeddings('embeddings.pkl')
  ```

---

## 2. Full ML Pipeline (`Full_train_with_emb.py`)
- **Purpose:** End-to-end pipeline for text classification using sentence transformers and neural networks.
- **Features:**
  - Embedding generation, model training, and evaluation
  - Supports multiple models, flexible data formats, YAML config
  - Advanced training (early stopping, LR scheduling, gradient clipping)
  - Detailed reporting and output structure
- **Quick Start:**
  ```bash
  python Full_train_with_emb.py --create_config
  python Full_train_with_emb.py --train_data train.csv --test_data test.csv --text_column "sentence" --label_column "label" --models "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  python Full_train_with_emb.py --config config.yaml
  ```
- **Output:**
  - Results, embeddings, trained models, plots, and evaluation reports in organized directories.
- **Supported Models:**
  - Any HuggingFace sentence transformer (Arabic, English, multilingual)
- **Metrics:**
  - Accuracy, ±1/±3 accuracy, QWK, average absolute distance, classification report

---

## 3. Neural Network Training & Testing (`trainer_tester.py`, `trainer_tester_combined.py`)
- **Purpose:** Modular PyTorch pipeline for training/testing on precomputed embeddings.
- **Features:**
  - Customizable neural network (GELU, dropout, batch norm)
  - Early stopping, LR scheduling, CLI hyperparameters
  - Training/testing visualizations, confusion matrix, artifact saving
- **Usage Example:**
  ```bash
  python trainer_tester.py --model_name sent --train_embeddings train_embeddings.pkl --test_embeddings test_embeddings.pkl --hidden_layers 1024 512 --dropout 0.3 --lr 0.001 --batch_size 2048 --epochs 50 --early_stopping 10 --scheduler_patience 5 --run_testing --save_plots
  ```
- **Output:**
  - Model checkpoints, training history, confusion matrix, predictions, evaluation metrics
- **Input Format:**
  - Pickle files with `embeddings`, `labels`, `ids`, `sentences` (optional), `model_name`

---

## Installation
Install requirements for all modules:
```bash
pip install torch transformers sentence-transformers scikit-learn matplotlib seaborn pandas numpy tqdm pyyaml
```

---

## Notes
- All tools are GPU-accelerated if CUDA is available.
- Input/output formats are consistent for easy integration between modules.
- See individual module READMEs for advanced configuration, troubleshooting, and extensibility tips.

---

## License
- Provided for research and educational purposes. Check individual model licenses as needed.
