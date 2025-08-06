# Neural Network Training & Testing Pipeline

This repository provides a reusable PyTorch-based training and evaluation pipeline for multi-class classification using precomputed embeddings. It is designed to be flexible and configurable through command-line arguments.

---

## 🧠 Features

- Modular design for training and testing phases
- Fully configurable via terminal
- Custom neural network architecture with:
  - GELU activation
  - Dropout
  - Batch normalization
- Early stopping and learning rate scheduling
- Training visualizations (accuracy/loss)
- Confusion matrix visualization
- Model and artifact saving
- CLI-based hyperparameter selection

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install torch scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Run the Pipeline

```bash
python train.py \
  --model_name sent \
  --train_embeddings clean_train_embeddings/bert_embeddings.pkl \
  --test_embeddings clean_test_embeddings/bert_embeddings.pkl \
  --hidden_layers 1024 512 \
  --dropout 0.3 \
  --lr 0.001 \
  --batch_size 2048 \
  --epochs 50 \
  --early_stopping 10 \
  --scheduler_patience 5 \
  --run_testing \
  --save_plots
```

---

## ⚙️ Arguments

| Argument              | Type      | Description                                    |
|-----------------------|-----------|------------------------------------------------|
| `--model_name`        | `str`     | Name used in the saved model directory         |
| `--train_embeddings`  | `str`     | Path to training embeddings pickle file        |
| `--test_embeddings`   | `str`     | Path to test embeddings pickle file            |
| `--hidden_layers`     | `int...`  | List of hidden layer sizes                     |
| `--dropout`           | `float`   | Dropout rate                                   |
| `--lr`                | `float`   | Learning rate                                  |
| `--batch_size`        | `int`     | Batch size                                     |
| `--epochs`            | `int`     | Maximum number of training epochs              |
| `--early_stopping`    | `int`     | Patience for early stopping                    |
| `--scheduler_patience`| `int`     | Patience for LR scheduler                      |
| `--run_testing`       | `flag`    | Include to run testing phase after training    |
| `--save_plots`        | `flag`    | Include to save loss and accuracy plots        |

---

## 📁 Output

When training finishes, you will find a folder like:

```
models/neural_net_20250806_153012_sent/
├── best_model.pth
├── best_training_artifacts.pkl
├── training_confusion_matrix.png
├── training_history.png
├── predictions_20250806_154501.csv
```

---

## 📊 Evaluation Metrics

- Accuracy
- Accuracy ±1
- Average absolute distance
- Quadratic weighted kappa (QWK)
- Classification report
- Confusion matrix

---

## 📌 Notes

- Input pickle files must include:
  - `embeddings`: `np.ndarray`
  - `labels`: list/array of class labels
  - `ids`: document/sample IDs
  - `sentences`: original sentences (optional)
  - `model_name`: name of embedding model used
- Make sure test labels are included in the test pickle file for metric calculation.

---

## 🛠️ License

Apache