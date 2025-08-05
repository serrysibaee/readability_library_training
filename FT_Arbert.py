# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import pandas as pd
import pickle
from typing import Dict, List
import numpy as np

# Load tokenizer and base BERT model (not AutoModelForMaskedLM!)
model_name = "UBC-NLP/ARBERTv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# === Custom Model ===
class BertRegressionModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # output scalar
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Use mean pooling over the sequence
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embeddings = summed / counts

        preds = self.regressor(mean_embeddings).squeeze(-1)

        if labels is not None:
            loss = self.loss_fn(preds, labels.float())
            return {"loss": loss, "logits": preds}
        return {"logits": preds}

# === Custom Dataset ===
class ReadabilityDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        inputs = self.tokenizer(
            row["Sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(row["Readability_Level_19"], dtype=torch.float)
        return item

# === Load Pickle Dataset ===
with open("sent_full_train_clean.pkl", "rb") as f:
    dataset = pickle.load(f)  # list of dicts

# Optional: split
from sklearn.model_selection import train_test_split
train_data, eval_data = train_test_split(dataset, test_size=0.1, random_state=42)

train_dataset = ReadabilityDataset(train_data, tokenizer)
eval_dataset = ReadabilityDataset(eval_data, tokenizer)

# === Training ===
model = BertRegressionModel(base_model)

training_args = TrainingArguments(
    output_dir="./arbert-regression",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    #evaluation_strategy="epoch",
    save_strategy="steps",
    logging_dir="./logs",
    logging_steps=500,
    #load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

