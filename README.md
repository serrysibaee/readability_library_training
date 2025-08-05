### Readability library Training

This repository contains a collection of Python scripts that together form a complete machine learning pipeline for a text-based task, likely either classification or regression. The core idea is to first generate numerical representations (embeddings) of text data and then use these embeddings to train and evaluate neural network models.

#### Files

* **`Full_train_with_emb.py`**
    This script is the main entry point for the entire workflow. It is designed to be highly configurable, allowing a user to define the neural network's architecture, specify training parameters, and set paths for data and model saving. The script's key functions include:
    * Loading and preprocessing data.
    * Generating sentence embeddings using the `SentenceTransformer` library.
    * Initializing and training a neural network model.
    * Visualizing the training and validation history (e.g., loss curves).
    * Saving the trained model and associated artifacts for future use.
    * This script acts as a powerful orchestrator, bringing together the embedding and training components into a single, executable process.

* **`trainer_tester.py`**
    This file encapsulates the fundamental logic for training and testing a neural network for a multi-label classification task. It defines a `MultiLabelNN` class, which is a simple feed-forward neural network. The script utilizes the `CrossEntropyLoss` function, which is a standard choice for classification problems. The script is structured to be modular, making it easy to integrate the training and testing functionality into other parts of the project, such as the `Full_train_with_emb.py` script.

* **`trainer_tester_combined.py`**
    This script provides an alternative training approach, specifically tailored for regression tasks. Instead of `CrossEntropyLoss`, it uses `MSELoss` (Mean Squared Error Loss) as the optimization objective. To work with this loss function, it prepares the target data by one-hot encoding it, which is a common practice for this type of problem. This file demonstrates a different loss function and data preparation method compared to `trainer_tester.py`, offering flexibility in how the model is trained.

* **`bert_embedder.py`**
    This is a specialized utility script for creating sentence embeddings using a pre-trained BERT model. It uses the "UBC-NLP/ARBERTv2" model, which is an Arabic BERT model, suggesting the project is focused on Arabic text. The script processes the input text and uses a mean pooling strategy to aggregate the token embeddings into a single, fixed-size vector representation for each sentence. This file is crucial for the first step of the pipeline, transforming raw text into a format that the neural networks can process.

* **`FT_Arbert.py`**
    This script is dedicated to the process of fine-tuning the "UBC-NLP/ARBERTv2" model. It extends the pre-trained BERT model by adding a custom regression head, allowing it to be used for a specific regression task. It leverages the Hugging Face `Trainer` class, a high-level API that simplifies the fine-tuning process by handling the training loop, evaluation, and logging. This script is a key part of the project when the goal is to adapt a powerful pre-trained language model to a specific downstream task.
