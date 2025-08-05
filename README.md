# FT_Arbert.py — Fine-Tuning ARBERTv2 for Arabic Readability Regression

## Overview

`FT_Arbert.py` is a Python script for fine-tuning the ARBERTv2 transformer model on an Arabic readability regression task. It leverages HuggingFace Transformers and PyTorch to train a model that predicts readability scores for Arabic sentences.

---

## Features

- **Custom Regression Model:**  
  Wraps ARBERTv2 with a regression head for predicting continuous readability scores.

- **Custom Dataset Loader:**  
  Loads and tokenizes data from a pickle file containing Arabic sentences and their readability levels.

- **Train/Eval Split:**  
  Splits the dataset into training and evaluation sets using scikit-learn.

- **Trainer Integration:**  
  Uses HuggingFace's `Trainer` and `TrainingArguments` for streamlined training and evaluation.

---

## File Structure

- **BertRegressionModel:**  
  A PyTorch `nn.Module` that adds a regression head to ARBERTv2 and uses mean pooling over token embeddings.

- **ReadabilityDataset:**  
  A custom `Dataset` class for loading and tokenizing the data.

- **Data Loading:**  
  Loads a pickle file (`sent_full_train_clean.pkl`) containing a list of dictionaries with `"Sentence"` and `"Readability_Level_19"` fields.

- **Training Setup:**  
  Splits data, initializes datasets, sets up training arguments, and runs training with HuggingFace's `Trainer`.

---

## Usage

1. **Prepare Data:**  
   Ensure you have a pickle file named `sent_full_train_clean.pkl` in the same directory.  
   The file should contain a list of dictionaries, each with:
   - `"Sentence"`: The Arabic sentence (string)
   - `"Readability_Level_19"`: The readability score (float or int)

2. **Install Requirements:**
   ```bash
   pip install torch transformers scikit-learn pandas numpy
   ```

3. **Run the Script:**
   ```bash
   python FT_Arbert.py
   ```

   The script will:
   - Load and split the data
   - Tokenize and prepare datasets
   - Fine-tune ARBERTv2 for regression
   - Save model checkpoints and logs

---

## Key Code Sections

- **Model Definition:**
  ```python
  class BertRegressionModel(nn.Module):
      ...
  ```

- **Dataset Preparation:**
  ```python
  class ReadabilityDataset(Dataset):
      ...
  ```

- **Training:**
  ```python
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
  )
  trainer.train()
  ```

---

## Customization

- **Change Model:**  
  Modify `model_name` to use a different transformer model.

- **Adjust Training Parameters:**  
  Edit the `TrainingArguments` section to change batch size, epochs, learning rate, etc.

- **Data File:**  
  Change the filename in the `open()` call if your data is named differently.

---

## License

This script is provided for research and educational purposes.

---

## Contact

For questions or contributions, please open an issue or pull request on the repository.