import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
import pandas as pd
from datetime import datetime
import glob

class MultiLabelNN(nn.Module):
    """
    Configurable Neural Network for Multi-label Classification with MSE Loss
    """
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.3):
        """
        Initialize the neural network

        Args:
            input_size (int): Size of input embeddings
            hidden_layers (list): List of hidden layer sizes [512, 256, 128]
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(MultiLabelNN, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),  # GELU activation as requested
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size

        # Output layer - no activation for MSE loss
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
    def __init__(self, config, experiment_name):
        """
        Initialize the trainer with configuration

        Args:
            config (dict): Configuration dictionary containing training parameters
            experiment_name (str): Name of the current experiment
        """
        self.config = config
        self.experiment_name = experiment_name
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

        # For MSE loss, create one-hot encoded targets
        self.y_train_onehot = torch.zeros(len(self.labels_encoded), self.num_classes)
        for i, label in enumerate(self.labels_encoded):
            self.y_train_onehot[i, label] = 1.0

        # Keep original encoded labels for accuracy calculation
        self.y_train = torch.LongTensor(self.labels_encoded)

        print(f"Training set size: {len(self.X_train)}")
        print(f"One-hot targets shape: {self.y_train_onehot.shape}")

        return self.X_train, self.y_train_onehot

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

        for batch_X, batch_y_onehot, batch_y_original in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y_onehot = batch_y_onehot.to(self.device)
            batch_y_original = batch_y_original.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)

            # Use MSE loss with one-hot targets
            loss = criterion(outputs, batch_y_onehot)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # For accuracy calculation, use argmax on outputs
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y_original.size(0)
            correct += (predicted == batch_y_original).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self):
        """Main training loop"""
        print("Starting training...")

        # Create data loader with both one-hot and original labels
        train_dataset = TensorDataset(self.X_train, self.y_train_onehot, self.y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        # Initialize MSE criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler (based on training loss since no validation)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config['scheduler_patience'],
            factor=0.3,
            #verbose=True
        )

        best_train_loss = float('inf')
        patience_counter = 0

        print(f"Using MSE Loss for training on {self.num_classes} classes")

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
                print(f'Train Loss (MSE): {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
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

    def evaluate_on_training_data(self):
        """Evaluate the model on training data (for monitoring purposes)"""
        print("Evaluating model on training data...")

        self.model.eval()
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)

                # For classification prediction, use argmax
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)

        # Convert back to original labels for reporting
        original_labels = self.label_encoder.inverse_transform(all_labels)
        original_predictions = self.label_encoder.inverse_transform(all_predictions)

        print(f"Training Data Accuracy: {accuracy:.4f}")
        print("\nClassification Report (Training Data):")
        print(classification_report(original_labels, original_predictions))

        # Confusion Matrix
        cm = confusion_matrix(original_labels, original_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.experiment_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.config['save_dir']}/training_confusion_matrix.png")
        if self.config['save_plots']:
            plt.show()
        plt.close()

        return accuracy, all_predictions, all_labels

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss (MSE)', color='blue')
        ax1.set_title(f'Model Loss - {self.experiment_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.set_title(f'Model Accuracy - {self.experiment_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.config['save_dir']}/training_history.png")
        if self.config['save_plots']:
            plt.show()
        plt.close()

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
            'class_names': self.label_encoder.classes_,
            'experiment_name': self.experiment_name
        }

        artifacts_path = f"{self.config['save_dir']}/{'best_' if is_best else ''}training_artifacts.pkl"
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)

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

            # For classification from MSE-trained model, use argmax
            _, predicted_indices = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted_indices.cpu().numpy())

        # Save predictions
        results = pd.DataFrame({
            'Sentence ID': test_data['ids'],
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

class ExperimentManager:
    def __init__(self, base_config):
        """
        Initialize the experiment manager

        Args:
            base_config (dict): Base configuration that will be used for all experiments
        """
        self.base_config = base_config
        self.experiment_results = []

    def find_embedding_pairs(self, train_dir, test_dir):
        """
        Find matching train-test embedding pairs

        Args:
            train_dir (str): Directory containing training embeddings
            test_dir (str): Directory containing test embeddings

        Returns:
            list: List of tuples (train_path, test_path, experiment_name)
        """
        train_files = glob.glob(os.path.join(train_dir, "*.pkl"))
        test_files = glob.glob(os.path.join(test_dir, "*.pkl"))

        pairs = []

        for train_file in train_files:
            train_basename = os.path.basename(train_file)
            # Look for corresponding test file
            test_file = os.path.join(test_dir, train_basename)

            if os.path.exists(test_file):
                # Extract experiment name from filename
                experiment_name = os.path.splitext(train_basename)[0]
                pairs.append((train_file, test_file, experiment_name))
            else:
                print(f"Warning: No matching test file found for {train_file}")

        return pairs

    def run_experiment(self, train_path, test_path, experiment_name):
        """
        Run a single experiment with given train/test pair

        Args:
            train_path (str): Path to training embeddings
            test_path (str): Path to test embeddings
            experiment_name (str): Name of the experiment

        Returns:
            dict: Experiment results
        """
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT: {experiment_name}")
        print(f"{'='*80}")

        # Create experiment-specific config
        experiment_config = self.base_config.copy()
        experiment_config['train_embeddings_path'] = train_path
        experiment_config['test_embeddings_path'] = test_path
        experiment_config['save_dir'] = f"{self.base_config['save_dir']}/{experiment_name}"

        print(f"Train embeddings: {train_path}")
        print(f"Test embeddings: {test_path}")
        print(f"Save directory: {experiment_config['save_dir']}")

        try:
            # TRAINING PHASE
            print(f"\n{'-'*60}")
            print("TRAINING PHASE (Using MSE Loss)")
            print(f"{'-'*60}")

            trainer = NeuralNetworkTrainer(experiment_config, experiment_name)
            trainer.load_embeddings(train_path)
            trainer.preprocess_data()
            trainer.create_model()
            trainer.train()

            # Evaluate on training data
            train_accuracy, _, _ = trainer.evaluate_on_training_data()

            # Plot training history
            trainer.plot_training_history()

            # Save final model
            trainer.save_model()

            # TESTING PHASE
            if experiment_config['run_testing']:
                print(f"\n{'-'*60}")
                print("TESTING PHASE")
                print(f"{'-'*60}")

                tester = ModelTester(device=trainer.device)

                # Prepare paths
                model_artifacts_path = f"{experiment_config['save_dir']}/best_training_artifacts.pkl"
                output_path = f"{experiment_config['save_dir']}/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                # Test model
                predicted_labels, true_labels = tester.test_model(
                    model_artifacts_path,
                    test_path,
                    output_path
                )

                # Compute metrics
                metrics = tester.compute_metrics(true_labels, predicted_labels)

                # Save metrics
                metrics_path = f"{experiment_config['save_dir']}/evaluation_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Add experiment info to metrics
                metrics['experiment_name'] = experiment_name
                metrics['train_accuracy'] = train_accuracy
                metrics['embedding_model'] = trainer.model_name

                print(f"\nEVALUATION METRICS for {experiment_name}:")
                print(f"{'-'*60}")
                for key, value in metrics.items():
                    if key != 'experiment_name':
                        print(f"{key}: {value:.2f}")

                return metrics

            else:
                return {
                    'experiment_name': experiment_name,
                    'train_accuracy': train_accuracy,
                    'embedding_model': trainer.model_name,
                    'status': 'training_only'
                }

        except Exception as e:
            print(f"Error in experiment {experiment_name}: {str(e)}")
            return {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e)
            }

    def run_all_experiments(self, train_dir, test_dir):
        """
        Run experiments for all embedding pairs

        Args:
            train_dir (str): Directory containing training embeddings
            test_dir (str): Directory containing test embeddings
        """
        print("="*80)
        print("MULTI-EMBEDDING EXPERIMENT PIPELINE (MSE Loss)")
        print("="*80)

        # Find embedding pairs
        pairs = self.find_embedding_pairs(train_dir, test_dir)

        if not pairs:
            print("No matching embedding pairs found!")
            return

        print(f"Found {len(pairs)} embedding pairs:")
        for i, (train_path, test_path, exp_name) in enumerate(pairs, 1):
            print(f"{i}. {exp_name}")

        # Run experiments
        for i, (train_path, test_path, experiment_name) in enumerate(pairs, 1):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(pairs)}: {experiment_name}")
            print(f"{'='*80}")

            result = self.run_experiment(train_path, test_path, experiment_name)
            self.experiment_results.append(result)

        # Create summary report
        self.create_summary_report()

        print(f"\n{'='*80}")
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"{'='*80}")
        print(f"Results saved in: {self.base_config['save_dir']}")

    def create_summary_report(self):
        """Create a summary report of all experiments"""
        summary_path = f"{self.base_config['save_dir']}/experiment_summary.json"

        # Save detailed results
        with open(summary_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)

        # Create CSV summary for easy viewing
        csv_data = []
        for result in self.experiment_results:
            if result.get('status') != 'failed':
                csv_data.append({
                    'Experiment': result['experiment_name'],
                    'Embedding Model': result.get('embedding_model', 'N/A'),
                    'Train Accuracy': result.get('train_accuracy', 0),
                    'Test Accuracy': result.get('accuracy', 0),
                    'Accuracy ±1': result.get('accuracy+-1', 0),
                    'Avg Abs Distance': result.get('avg_abs_dist', 0),
                    'QWK': result.get('qwk', 0)
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = f"{self.base_config['save_dir']}/experiment_summary.csv"
            df.to_csv(csv_path, index=False)

            print(f"\nEXPERIMENT SUMMARY:")
            print(f"{'='*80}")
            print(df.to_string(index=False))
            print(f"\nSummary saved to: {csv_path}")

def main():
    """Main function for running multiple experiments"""
    print("="*80)
    print("MULTI-EMBEDDING NEURAL NETWORK TRAINING PIPELINE (MSE Loss)")
    print("="*80)

    # Base configuration (will be used for all experiments)
    base_config = {
        # Directory paths
        'train_embeddings_dir': 'blind_train_embeddings',  # Directory containing train embeddings
        'test_embeddings_dir':'blind_test_embeddings',    # Directory containing test embeddings
        'random_state': 42,

        # Model architecture
        'hidden_layers': [8*512, 4*512, 2*512, 2*512, 512],  # Customize layer sizes
        'dropout_rate': 0.0,

        # Training parameters
        'learning_rate': 0.1,
        'batch_size': 1024*16*4, # 65536
        'epochs': 2500,
        'weight_decay': 5e-5,

        # Training control
        'early_stopping_patience': 500,
        'scheduler_patience': 25,
        'print_every': 250,

        # Save parameters
        'save_dir': f'experiments/multi_embedding_mse_blind_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

        # Testing parameters
        'run_testing': True,
        'save_plots': False
    }

    print("Base Configuration:")
    for key, value in base_config.items():
        print(f"  {key}: {value}")
    print()

    # Create experiment manager
    experiment_manager = ExperimentManager(base_config)

    # Run all experiments
    experiment_manager.run_all_experiments(
        base_config['train_embeddings_dir'],
        base_config['test_embeddings_dir']
    )

if __name__ == "__main__":
    main()
