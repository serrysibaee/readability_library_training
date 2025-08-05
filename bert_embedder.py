import json
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import os

class BERTEmbedder:
    def __init__(self, model_name="UBC-NLP/ARBERTv2"):
        """
        Initialize the BERT embedder

        Args:
            model_name (str): Name of the BERT model to use
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")

    def load_data(self, json_file_path):
        """Load data from JSON file"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {json_file_path}")
        return data

    def get_sentence_embedding(self, sentence, max_length=512):
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
            outputs = self.model.bert(**inputs)  # Get BERT embeddings (not MLM head)

            # Use [CLS] token embedding as sentence representation
            #cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Alternative: Use mean pooling of all tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()

        return mean_embedding.squeeze()

    def generate_embeddings(self, data, batch_size=8, max_length=512):
        """
        Generate embeddings for all sentences in the dataset

        Args:
            data (list): List of dictionaries containing the data
            batch_size (int): Batch size for processing
            max_length (int): Maximum sequence length

        Returns:
            dict: Dictionary containing embeddings, IDs, sentences, and labels
        """
        sentences = [item['Sentence'] for item in data]
        ids = [item['ID'] for item in data]
        labels = [item['Readability_Level_19'] for item in data]

        embeddings = []

        print("Generating BERT embeddings...")

        # Process in batches
        for i in tqdm(range(0, len(sentences), batch_size)):
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

                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        return {
            'embeddings': embeddings,
            'ids': ids,
            'sentences': sentences,
            'labels': labels,
            'model_name': self.model_name
        }

    def save_embeddings(self, embeddings_data, output_path):
        """Save embeddings to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_data, f)

        print(f"Embeddings saved to: {output_path}")
        print(f"Embedding shape: {embeddings_data['embeddings'].shape}")
        print(f"Model used: {embeddings_data['model_name']}")

def main():
    # Configuration
    JSON_FILE_PATH = "sent_full_train_clean.pkl"  # Update this path
    OUTPUT_DIR = "blind_train_embeddings/"

    # List of BERT models to try
    bert_models = [
        "UBC-NLP/ARBERTv2",
        "UBC-NLP/MARBERTv2",
        "aubmindlab/bert-base-arabertv02",
        # Add more BERT models as needed
    ]

    # Load data once
    print("Loading dataset...")
    # with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    import pickle
    with open(JSON_FILE_PATH, 'rb') as f:
        data = pickle.load(f)

    # Generate embeddings for each model
    for model_name in bert_models:
        try:
            print(f"\n{'='*50}")
            print(f"Processing with model: {model_name}")
            print(f"{'='*50}")

            # Initialize embedder
            embedder = BERTEmbedder(model_name)

            # Generate embeddings
            embeddings_data = embedder.generate_embeddings(
                data,
                batch_size=2048 ,  # Smaller batch size for BERT
                max_length=512
            )

            # Create output filename
            model_short_name = model_name.split('/')[-1]
            output_path = f"{OUTPUT_DIR}bert_{model_short_name}_embeddings.pkl"

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

# Function to load saved embeddings for training/testing
def load_embeddings(embeddings_path):
    """
    Load saved embeddings for use in training/testing

    Args:
        embeddings_path (str): Path to saved embeddings file

    Returns:
        dict: Dictionary containing embeddings and metadata
    """
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)

    print(f"Loaded embeddings: {embeddings_data['embeddings'].shape}")
    print(f"Model used: {embeddings_data['model_name']}")

    return embeddings_data

# Alternative embedding extraction method using mean pooling
class BERTEmbedderMeanPooling(BERTEmbedder):
    """BERT Embedder using mean pooling instead of CLS token"""

    def get_sentence_embedding(self, sentence, max_length=512):
        """Get embedding using mean pooling of all tokens"""
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.bert(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()

        return mean_embedding.squeeze()

if __name__ == "__main__":
    main()

    # Example of how to load embeddings for training
    # embeddings_data = load_embeddings("embeddings/bert_ARBERTv2_embeddings.pkl")
    # X = embeddings_data['embeddings']
    # y = embeddings_data['labels']
    # ids = embeddings_data['ids']

    # Connect with drive
