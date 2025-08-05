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

class SentenceTransformerEmbedder:
    def __init__(self, model_name):
        """
        Initialize the embedder with a specified model.

        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print(f"Loaded model: {model_name}")

    def load_data(self, file_path):
        """Load data from pickle or JSON file"""
        try:
            # Try pickle first
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} samples from {file_path} (pickle)")
        except:
            # Try JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} samples from {file_path} (JSON)")
        return data

    def generate_embeddings(self, data, batch_size=1024):
        """
        Generate embeddings for all sentences in the dataset

        Args:
            data (list): List of dictionaries containing the data
            batch_size (int): Batch size for processing

        Returns:
            dict: Dictionary containing embeddings, IDs, sentences, and labels
        """
        sentences = [item['Sentence'] for item in data]
        ids = [item['ID'] for item in data]
        labels = [item['Readability_Level_19'] for item in data]

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

class MultiLabelNN(nn.Module):
    """
    Configurable Neural Network for Multi-label Classification
    """
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.3):
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
    def __init__(self, config):
        """
        Initialize the trainer with configuration

        Args:
            config (dict): Configuration dictionary containing training parameters
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

    def load_embeddings(self, embeddings_path):
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

    def preprocess_data(self):
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

    def create_model(self):
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

    def train_epoch(self, train_loader, criterion, optimizer):
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

    def train(self):
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
            factor=0.5,
            #verbose=True
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

    def save_model(self, is_best=False):
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
        #plt.show()

class ModelTester:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")

    def test_model(self, model_artifacts_path, test_embeddings_path, output_path):
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

    def compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred) * 100
        accuracy_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        accuracy_3 = np.mean(np.abs(y_true - y_pred) <= 3) * 100
        accuracy_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        accuracy_7 = np.mean(np.abs(y_true - y_pred) <= 7) * 100
        avg_abs_dist = np.mean(np.abs(y_true - y_pred))
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic') * 100

        return {
            "accuracy": accuracy,
            "accuracy+-1": accuracy_1,
            #"accuracy+-3": accuracy_3,
            #"accuracy+-5": accuracy_5,
            #"accuracy+-7": accuracy_7,
            "avg_abs_dist": avg_abs_dist,
            "qwk": qwk
        }

    def save_results_to_txt(self, model_name, config, metrics, output_path):
        """Save configuration, model name, and evaluation results to a text file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            # Timestamp
            f.write(f"Evaluation Date: {timestamp}\n\n")

            # Model Information
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Model Short Name: {model_name.split('/')[-1]}\n\n")

            # Configuration
            f.write("CONFIGURATION:\n")
            f.write("-" * 40 + "\n")

            # Data paths
            f.write("Data Paths:\n")
            f.write(f"  - Training Data: {config.get('train_data_path', 'N/A')}\n")
            f.write(f"  - Test Data: {config.get('test_data_path', 'N/A')}\n\n")

            # Model architecture
            if 'training_config' in config:
                training_config = config['training_config']
                f.write("Model Architecture:\n")
                f.write(f"  - Hidden Layers: {training_config.get('hidden_layers', 'N/A')}\n")
                f.write(f"  - Dropout Rate: {training_config.get('dropout_rate', 'N/A')}\n\n")

                # Training parameters
                f.write("Training Parameters:\n")
                f.write(f"  - Learning Rate: {training_config.get('learning_rate', 'N/A')}\n")
                f.write(f"  - Batch Size: {training_config.get('batch_size', 'N/A')}\n")
                f.write(f"  - Maximum Epochs: {training_config.get('epochs', 'N/A')}\n")
                f.write(f"  - Weight Decay: {training_config.get('weight_decay', 'N/A')}\n")
                f.write(f"  - Early Stopping Patience: {training_config.get('early_stopping_patience', 'N/A')}\n")
                f.write(f"  - Scheduler Patience: {training_config.get('scheduler_patience', 'N/A')}\n\n")

            # Embedding parameters
            f.write("Embedding Parameters:\n")
            f.write(f"  - Embedding Batch Size: {config.get('embedding_batch_size', 'N/A')}\n\n")

            # Evaluation Results
            f.write("EVALUATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {metric_value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")

class MLPipeline:
    def __init__(self, config):
        """
        Initialize the complete ML pipeline

        Args:
            config (dict): Configuration dictionary
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

    def generate_embeddings(self, model_name):
        """Generate embeddings for train and test data"""
        print(f"\nSTEP 1: GENERATING EMBEDDINGS FOR {model_name}")
        print("-" * 60)

        # Initialize embedder
        embedder = SentenceTransformerEmbedder(model_name,)

        # Create model short name for file naming
        model_short_name = model_name.split('/')[-1]

        # Generate train embeddings
        print("Loading training data...")
        train_data = embedder.load_data(self.config['train_data_path'])

        print("Generating training embeddings...")
        train_embeddings = embedder.generate_embeddings(train_data, batch_size=self.config['embedding_batch_size'])

        train_embeddings_path = f"{self.config['embeddings_dir']}/train_{model_short_name}_embeddings.pkl"
        embedder.save_embeddings(train_embeddings, train_embeddings_path)

        # Generate test embeddings
        print("Loading test data...")
        test_data = embedder.load_data(self.config['test_data_path'])

        print("Generating test embeddings...")
        test_embeddings = embedder.generate_embeddings(test_data, batch_size=self.config['embedding_batch_size'])

        test_embeddings_path = f"{self.config['embeddings_dir']}/test_{model_short_name}_embeddings.pkl"
        embedder.save_embeddings(test_embeddings, test_embeddings_path)

        return train_embeddings_path, test_embeddings_path

    def train_neural_network(self, train_embeddings_path, model_name):
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
        if self.config['save_plots']:
            trainer.plot_training_history()

        # Save final model
        trainer.save_model()

        return f"{save_dir}/best_training_artifacts.pkl"

    def test_model(self, model_artifacts_path, test_embeddings_path, model_name):
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

        # NEW: Save detailed results to TXT file
        txt_results_path = f"{os.path.dirname(model_artifacts_path)}/evaluation_results.txt"
        tester.save_results_to_txt(model_name, self.config, metrics, txt_results_path)
        print(f"Detailed results saved to: {txt_results_path}")

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

        # NEW: Save comprehensive summary to TXT file
        summary_txt_path = f"{self.config['output_dir']}/final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.save_comprehensive_summary(results_df, summary_txt_path)
        print(f"Comprehensive summary saved to: {summary_txt_path}")

        # Find best model
        best_model = results_df['qwk'].idxmax()
        best_qwk = results_df.loc[best_model, 'qwk']

        print(f"\nBEST MODEL: {best_model}")
        print(f"BEST QWK SCORE: {best_qwk:.2f}")

    def save_comprehensive_summary(self, results_df, output_path):
        """Save a comprehensive summary of all model results to a text file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE MODEL COMPARISON SUMMARY\n")
            f.write("=" * 100 + "\n\n")

            # Timestamp
            f.write(f"Evaluation Date: {timestamp}\n\n")

            # Configuration Summary
            f.write("PIPELINE CONFIGURATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Training Data: {self.config.get('train_data_path', 'N/A')}\n")
            f.write(f"Test Data: {self.config.get('test_data_path', 'N/A')}\n")
            f.write(f"Embedding Batch Size: {self.config.get('embedding_batch_size', 'N/A')}\n")

            if 'training_config' in self.config:
                tc = self.config['training_config']
                f.write(f"Neural Network Architecture: {tc.get('hidden_layers', 'N/A')}\n")
                f.write(f"Dropout Rate: {tc.get('dropout_rate', 'N/A')}\n")
                f.write(f"Learning Rate: {tc.get('learning_rate', 'N/A')}\n")
                f.write(f"Batch Size: {tc.get('batch_size', 'N/A')}\n")
                f.write(f"Max Epochs: {tc.get('epochs', 'N/A')}\n")

            f.write("\n")

            # Models tested
            f.write("MODELS EVALUATED:\n")
            f.write("-" * 50 + "\n")
            for i, model in enumerate(self.config['models_to_use'], 1):
                f.write(f"{i}. {model}\n")
            f.write("\n")

            # Results table
            f.write("RESULTS COMPARISON:\n")
            f.write("-" * 50 + "\n")

            # Table header
            f.write(f"{'Model':<50} {'Accuracy':<12} {'Acc±1':<12} {'Avg Dist':<12} {'QWK':<12}\n")
            f.write("-" * 98 + "\n")

            # Table rows
            for model_name, row in results_df.iterrows():
                model_short = model_name.split('/')[-1][:45]  # Truncate long model names
                f.write(f"{model_short:<50} {row['accuracy']:>8.2f}%   {row['accuracy+-1']:>8.2f}%   {row['avg_abs_dist']:>8.2f}     {row['qwk']:>8.2f}\n")

            f.write("\n")

            # Best model
            best_model = results_df['qwk'].idxmax()
            best_qwk = results_df.loc[best_model, 'qwk']
            f.write("BEST PERFORMING MODEL:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: {best_model}\n")
            f.write(f"QWK Score: {best_qwk:.2f}\n")
            f.write(f"Accuracy: {results_df.loc[best_model, 'accuracy']:.2f}%\n")
            f.write(f"Accuracy ±1: {results_df.loc[best_model, 'accuracy+-1']:.2f}%\n")
            f.write(f"Average Absolute Distance: {results_df.loc[best_model, 'avg_abs_dist']:.2f}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF COMPREHENSIVE SUMMARY\n")
            f.write("=" * 100 + "\n")

def main():
    """Main function to run the complete pipeline"""

    # CONFIGURATION
    config = {
        # Data paths
        'train_data_path': 'sen_train_data.pkl',  # Update this path
        'test_data_path': 'sent_test_data.pkl',    # Update this path

        # Models to test
        'models_to_use': [
            "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            #"Snowflake/snowflake-arctic-embed-m-v2.0",
            #"jinaai/jina-embeddings-v3",
            "sentence-transformers/LaBSE",
        ],

        # Output directories
        'embeddings_dir': 'embeddings',
        'models_dir': 'models',
        'output_dir': 'results',

        # Embedding parameters
        'embedding_batch_size': 4*2048,

        # Training configuration
        'training_config': {
            # Model architecture
            'hidden_layers': [8*512, 4*512, 1*512],
            'dropout_rate': 0.4,

            # Training parameters
            'learning_rate': 0.001,
            'batch_size': 1024*8,
            'epochs': 400,
            'weight_decay': 1e-5,

            # Training control
            'early_stopping_patience': 25,
            'scheduler_patience': 5,
            'print_every': 50,
        },

        # General settings
        'save_plots': True,
    }

    # Run pipeline
    pipeline = MLPipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
