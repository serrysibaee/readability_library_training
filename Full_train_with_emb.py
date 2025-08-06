#!/usr/bin/env python3
"""
ML Pipeline for Text Classification using Sentence Transformers and Neural Networks

A comprehensive command-line tool for generating embeddings, training neural networks,
and evaluating models for text classification tasks.
"""

import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import argparse
import yaml
import sys
from typing import List, Dict, Any, Union, Tuple

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        """
        Initialize the embedder with a specified model.

        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print(f"Loaded model: {model_name}")
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

    def generate_embeddings(self, data: List[Dict], text_column: str, 
                          id_column: str = None, label_column: str = None,
                          batch_size: int = 1024) -> Dict[str, Any]:
        """
        Generate embeddings for all sentences in the dataset

        Args:
            data (List[Dict]): List of dictionaries containing the data
            text_column (str): Name of the column containing text
            id_column (str): Name of the column containing IDs
            label_column (str): Name of the column containing labels
            batch_size (int): Batch size for processing

        Returns:
            Dict: Dictionary containing embeddings, IDs, sentences, and labels
        """
        sentences = [item[text_column] for item in data]
        ids = [item.get(id_column, i) for i, item in enumerate(data)] if id_column else list(range(len(data)))
        labels = [item.get(label_column) for item in data] if label_column else [None] * len(data)

        print("Generating embeddings...")
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return {
            'embeddings': embeddings,
            'ids': ids,
            'sentences': sentences,
            'labels': labels,
            'model_name': self.model_name,
            'text_column': text_column,
            'id_column': id_column,
            'label_column': label_column
        }

    def save_embeddings(self, embeddings_data: Dict[str, Any], output_path: str):
        """Save embeddings to file"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_data, f)

        print(f"Embeddings saved to: {output_path}")
        print(f"Embedding shape: {embeddings_data['embeddings'].shape}")

class MultiLabelNN(nn.Module):
    """
    Configurable Neural Network for Multi-label Classification
    """
    def __init__(self, input_size: int, hidden_layers: List[int], 
                 num_classes: int, dropout_rate: float = 0.3):
        super(MultiLabelNN, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

class NeuralNetworkTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration

        Args:
            config (Dict): Configuration dictionary containing training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.train_losses = []
        self.train_accuracies = []

    def load_embeddings(self, embeddings_path: str) -> Tuple[np.ndarray, List]:
        """Load embeddings from pickle file"""
        print(f"Loading embeddings from: {embeddings_path}")

        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)

        self.embeddings = embeddings_data['embeddings']
        self.labels = embeddings_data['labels']
        self.ids = embeddings_data['ids']
        self.sentences = embeddings_data['sentences']
        self.model_name = embeddings_data['model_name']

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Number of samples: {len(self.labels)}")
        print(f"Embedding model: {self.model_name}")

        return self.embeddings, self.labels

    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the data for training"""
        print("Preprocessing data...")

        # Normalize embeddings
        self.embeddings_scaled = self.scaler.fit_transform(self.embeddings)

        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

        # Get unique classes
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")

        # Convert to tensors
        self.X_train = torch.FloatTensor(self.embeddings_scaled)
        self.y_train = torch.LongTensor(self.labels_encoded)

        print(f"Training set size: {len(self.X_train)}")

        return self.X_train, self.y_train

    def create_model(self) -> nn.Module:
        """Create the neural network model"""
        input_size = self.embeddings.shape[1]

        self.model = MultiLabelNN(
            input_size=input_size,
            hidden_layers=self.config['hidden_layers'],
            num_classes=self.num_classes,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)

        print(f"Created model with architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden layers: {self.config['hidden_layers']}")
        print(f"Output classes: {self.num_classes}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")

        return self.model

    def train_epoch(self, train_loader, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self) -> nn.Module:
        """Main training loop"""
        print("Starting training...")

        # Create data loader
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        # Initialize criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config['scheduler_patience'],
            factor=0.5
        )

        best_train_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Update learning rate based on training loss
            scheduler.step(train_loss)

            # Print progress
            if (epoch + 1) % self.config['print_every'] == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 50)

            # Early stopping based on training loss improvement
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
                # Save best model
                self.save_model(is_best=True)
            else:
                patience_counter += 1

            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        print("Training completed!")
        return self.model

    def save_model(self, is_best: bool = False):
        """Save the model and training artifacts"""
        os.makedirs(self.config['save_dir'], exist_ok=True)

        # Model state
        model_path = f"{self.config['save_dir']}/{'best_' if is_best else ''}model.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save complete training artifacts
        artifacts = {
            'model_config': {
                'input_size': self.embeddings.shape[1],
                'hidden_layers': self.config['hidden_layers'],
                'num_classes': self.num_classes,
                'dropout_rate': self.config['dropout_rate']
            },
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'training_config': self.config,
            'training_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies
            },
            'embedding_model_name': self.model_name,
            'class_names': self.label_encoder.classes_
        }

        artifacts_path = f"{self.config['save_dir']}/{'best_' if is_best else ''}training_artifacts.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.config['save_dir']}/training_history.png")
        plt.close()

class ModelTester:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")

    def test_model(self, model_artifacts_path: str, test_embeddings_path: str, 
                   output_path: str) -> Tuple[List, List]:
        """Test the trained model on test data"""
        print("Starting model testing...")

        # Load training artifacts
        with open(model_artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        # Load test embeddings
        with open(test_embeddings_path, 'rb') as f:
            test_data = pickle.load(f)

        # Initialize model
        model_config = artifacts['model_config']
        model = MultiLabelNN(**model_config).to(self.device)

        # Load model weights
        model_path = model_artifacts_path.replace('training_artifacts.pkl', 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Preprocess test data
        scaler = artifacts['scaler']
        label_encoder = artifacts['label_encoder']
        test_embeddings_scaled = scaler.transform(test_data['embeddings'])
        test_tensor = torch.FloatTensor(test_embeddings_scaled).to(self.device)

        # Make predictions
        with torch.no_grad():
            outputs = model(test_tensor)
            _, predicted_indices = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted_indices.cpu().numpy())

        # Save predictions
        results = pd.DataFrame({
            'Document ID': test_data['ids'],
            'Prediction': predicted_labels
        })
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        print(f"Total predictions: {len(results)}")

        return predicted_labels, test_data['labels']

    def compute_metrics(self, y_true: List, y_pred: List) -> Dict[str, float]:
        """Compute evaluation metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred) * 100
        accuracy_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        accuracy_3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
        avg_abs_dist = np.mean(np.abs(y_true - y_pred))
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic') * 100

        return {
            "accuracy": accuracy,
            "accuracy+-1": accuracy_1,
            "accuracy+-3": accuracy_3,
            "avg_abs_dist": avg_abs_dist,
            "qwk": qwk
        }

class MLPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the complete ML pipeline

        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.results = {}

    def run_pipeline(self):
        """Run the complete pipeline for all models"""
        print("="*80)
        print("STARTING COMPLETE ML PIPELINE")
        print("="*80)

        # Process each model
        for model_name in self.config['models_to_use']:
            try:
                print(f"\n{'='*80}")
                print(f"PROCESSING MODEL: {model_name}")
                print(f"{'='*80}")

                # Step 1: Generate embeddings
                train_embeddings_path, test_embeddings_path = self.generate_embeddings(model_name)

                # Step 2: Train neural network
                model_artifacts_path = self.train_neural_network(train_embeddings_path, model_name)

                # Step 3: Test model
                metrics = self.test_model(model_artifacts_path, test_embeddings_path, model_name)

                # Store results
                self.results[model_name] = metrics

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing model {model_name}: {str(e)}")
                continue

        # Save and display final results
        self.save_final_results()

    def generate_embeddings(self, model_name: str) -> Tuple[str, str]:
        """Generate embeddings for train and test data"""
        print(f"\nSTEP 1: GENERATING EMBEDDINGS FOR {model_name}")
        print("-" * 60)

        # Initialize embedder
        embedder = SentenceTransformerEmbedder(model_name)

        # Create model short name for file naming
        model_short_name = model_name.split('/')[-1]

        # Generate train embeddings
        print("Loading training data...")
        train_data = embedder.load_data(
            self.config['train_data_path'], 
            self.config['text_column'],
            self.config.get('id_column'),
            self.config.get('label_column')
        )

        print("Generating training embeddings...")
        train_embeddings = embedder.generate_embeddings(
            train_data, 
            self.config['text_column'],
            self.config.get('id_column'),
            self.config.get('label_column'),
            batch_size=self.config['embedding_batch_size']
        )

        train_embeddings_path = f"{self.config['embeddings_dir']}/train_{model_short_name}_embeddings.pkl"
        embedder.save_embeddings(train_embeddings, train_embeddings_path)

        # Generate test embeddings
        print("Loading test data...")
        test_data = embedder.load_data(
            self.config['test_data_path'], 
            self.config['text_column'],
            self.config.get('id_column'),
            self.config.get('label_column')
        )

        print("Generating test embeddings...")
        test_embeddings = embedder.generate_embeddings(
            test_data, 
            self.config['text_column'],
            self.config.get('id_column'),
            self.config.get('label_column'),
            batch_size=self.config['embedding_batch_size']
        )

        test_embeddings_path = f"{self.config['embeddings_dir']}/test_{model_short_name}_embeddings.pkl"
        embedder.save_embeddings(test_embeddings, test_embeddings_path)

        return train_embeddings_path, test_embeddings_path

    def train_neural_network(self, train_embeddings_path: str, model_name: str) -> str:
        """Train neural network on embeddings"""
        print(f"\nSTEP 2: TRAINING NEURAL NETWORK FOR {model_name}")
        print("-" * 60)

        # Create model-specific save directory
        model_short_name = model_name.split('/')[-1]
        save_dir = f"{self.config['models_dir']}/neural_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_short_name}"

        # Update config with paths
        train_config = self.config['training_config'].copy()
        train_config['save_dir'] = save_dir

        # Initialize and run trainer
        trainer = NeuralNetworkTrainer(train_config)
        trainer.load_embeddings(train_embeddings_path)
        trainer.preprocess_data()
        trainer.create_model()
        trainer.train()

        # Plot training history
        if self.config.get('save_plots', True):
            trainer.plot_training_history()

        # Save final model
        trainer.save_model()

        return f"{save_dir}/best_training_artifacts.pkl"

    def test_model(self, model_artifacts_path: str, test_embeddings_path: str, 
                   model_name: str) -> Dict[str, float]:
        """Test the trained model"""
        print(f"\nSTEP 3: TESTING MODEL FOR {model_name}")
        print("-" * 60)

        # Initialize tester
        tester = ModelTester()

        # Prepare output path
        model_short_name = model_name.split('/')[-1]
        output_path = f"{os.path.dirname(model_artifacts_path)}/predictions_{model_short_name}.csv"

        # Test model
        predicted_labels, true_labels = tester.test_model(
            model_artifacts_path,
            test_embeddings_path,
            output_path
        )

        # Compute metrics
        metrics = tester.compute_metrics(true_labels, predicted_labels)

        # Display metrics
        print("\nEVALUATION METRICS:")
        print("-" * 30)
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        # Save metrics to JSON
        metrics_path = f"{os.path.dirname(model_artifacts_path)}/evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def save_final_results(self):
        """Save and display final results for all models"""
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)

        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(2)

        # Display results
        print(results_df)

        # Save results
        results_path = f"{self.config['output_dir']}/final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path)

        print(f"\nFinal results saved to: {results_path}")

        # Find best model
        if not results_df.empty:
            best_model = results_df['qwk'].idxmax()
            best_qwk = results_df.loc[best_model, 'qwk']

            print(f"\nBEST MODEL: {best_model}")
            print(f"BEST QWK SCORE: {best_qwk:.2f}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        sys.exit(1)

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        # Data paths and columns
        'train_data_path': 'train_data.pkl',
        'test_data_path': 'test_data.pkl',
        'text_column': 'Sentence',
        'id_column': 'ID',
        'label_column': 'Readability_Level_19',
        
        # Models to test
        'models_to_use': [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/LaBSE",
        ],
        
        # Output directories
        'embeddings_dir': 'embeddings',
        'models_dir': 'models',
        'output_dir': 'results',
        
        # Embedding parameters
        'embedding_batch_size': 1024,
        
        # Training configuration
        'training_config': {
            # Model architecture
            'hidden_layers': [512, 256, 128],
            'dropout_rate': 0.3,
            
            # Training parameters
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 100,
            'weight_decay': 1e-5,
            
            # Training control
            'early_stopping_patience': 10,
            'scheduler_patience': 5,
            'print_every': 10,
        },
        
        # General settings
        'save_plots': True,
    }

def main():
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Text Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument('--config', '-c', help='Path to YAML configuration file')
    parser.add_argument('--create_config', action='store_true', 
                       help='Create a default configuration file')
    
    # Data arguments
    parser.add_argument('--train_data', help='Path to training data file')
    parser.add_argument('--test_data', help='Path to test data file')
    parser.add_argument('--text_column', default='Sentence', 
                       help='Name of text column')
    parser.add_argument('--id_column', help='Name of ID column')
    parser.add_argument('--label_column', default='Readability_Level_19',
                       help='Name of label column')
    
    # Model arguments
    parser.add_argument('--models', nargs='+', 
                       help='List of sentence transformer models to use')
    parser.add_argument('--embedding_batch_size', type=int, default=1024,
                       help='Batch size for embedding generation')
    
    # Training arguments
    parser.add_argument('--hidden_layers', nargs='+', type=int, 
                       default=[512, 256, 128], help='Hidden layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--embeddings_dir', default='embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--models_dir', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--output_dir', default='results',
                       help='Directory to save results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable saving plots')
    
    args = parser.parse_args()
    
    # Create default config file if requested
    if args.create_config:
        config = create_default_config()
        config_path = 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Default configuration saved to: {config_path}")
        print("Edit the configuration file and run again with --config config.yaml")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.train_data:
        config['train_data_path'] = args.train_data
    if args.test_data:
        config['test_data_path'] = args.test_data
    if args.text_column:
        config['text_column'] = args.text_column
    if args.id_column:
        config['id_column'] = args.id_column
    if args.label_column:
        config['label_column'] = args.label_column
    if args.models:
        config['models_to_use'] = args.models
    if args.embedding_batch_size:
        config['embedding_batch_size'] = args.embedding_batch_size
    if args.embeddings_dir:
        config['embeddings_dir'] = args.embeddings_dir
    if args.models_dir:
        config['models_dir'] = args.models_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Override training config
    training_config = config['training_config']
    if args.hidden_layers:
        training_config['hidden_layers'] = args.hidden_layers
    if args.dropout_rate:
        training_config['dropout_rate'] = args.dropout_rate
    if args.learning_rate:
        training_config['learning_rate'] = args.learning_rate
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.epochs:
        training_config['epochs'] = args.epochs
    if args.early_stopping_patience:
        training_config['early_stopping_patience'] = args.early_stopping_patience
    
    # Other settings
    config['save_plots'] = not args.no_plots
    
    # Validate required arguments
    if not config.get('train_data_path') or not config.get('test_data_path'):
        print("Error: Training and test data paths must be specified")
        print("Use --train_data and --test_data or provide them in config file")
        sys.exit(1)
    
    if not config.get('models_to_use'):
        print("Error: At least one model must be specified")
        print("Use --models or provide them in config file")
        sys.exit(1)
    
    # Display configuration
    print("Configuration:")
    print("-" * 40)
    print(f"Train data: {config['train_data_path']}")
    print(f"Test data: {config['test_data_path']}")
    print(f"Text column: {config['text_column']}")
    print(f"Label column: {config['label_column']}")
    print(f"Models: {config['models_to_use']}")
    print(f"Hidden layers: {config['training_config']['hidden_layers']}")
    print(f"Learning rate: {config['training_config']['learning_rate']}")
    print(f"Batch size: {config['training_config']['batch_size']}")
    print(f"Max epochs: {config['training_config']['epochs']}")
    print("-" * 40)
    
    # Run pipeline
    pipeline = MLPipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()