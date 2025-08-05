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

class MultiLabelNN(nn.Module):
    """
    Configurable Neural Network for Multi-label Classification
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

        # Convert to tensors (use all data for training)
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

        # Learning rate scheduler (based on training loss since no validation)
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
        plt.title('Confusion Matrix (Training Data)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.config['save_dir']}/training_confusion_matrix.png")
        plt.show()

        return accuracy, all_predictions, all_labels

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

def main():
    """Main function that handles both training and testing"""
    print("="*60)
    print("NEURAL NETWORK TRAINING AND TESTING PIPELINE")
    print("="*60)

    # Configuration
    name_of_model = "sent"
    config = {
        # Data parameters
        'train_embeddings_path': 'clean_train_embeddings/bert_bert-base-arabertv02_embeddings.pkl',
        'test_embeddings_path': 'clean_test_embeddings/bert_bert-base-arabertv02_embeddings.pkl',
        'random_state': 42,

        # Model architecture
        'hidden_layers': [8*512, 4*512, ],  # Customize layer sizes
        'dropout_rate': 0.3,

        # Training parameters
        'learning_rate': 0.001,  # Customize learning rate
        'batch_size': 1024*16*4, # 62155
        'epochs': 75,
        'weight_decay': 0,

        # Training control
        'early_stopping_patience': 25,  # Based on training loss plateau
        'scheduler_patience': 5,
        'print_every': 50,

        # Save parameters
        'save_dir': f'models/neural_net_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{name_of_model}',

        # Testing parameters
        'run_testing': True,  # Set to False if you only want to train
        'save_plots': True    # Set to False to skip saving plots
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # PHASE 1: TRAINING
    print("="*60)
    print("PHASE 1: TRAINING")
    print("="*60)

    # Initialize trainer
    trainer = NeuralNetworkTrainer(config)

    # Load and preprocess data
    trainer.load_embeddings(config['train_embeddings_path'])
    trainer.preprocess_data()

    # Create and train model
    trainer.create_model()
    trainer.train()

    # Evaluate model on training data (for monitoring)
    trainer.evaluate_on_training_data()

    # Plot training history
    if config['save_plots']:
        trainer.plot_training_history()

    # Save final model
    trainer.save_model()

    print("Training phase completed!")
    print(f"Model and artifacts saved in: {config['save_dir']}")

    # PHASE 2: TESTING
    if config['run_testing']:
        print("\n" + "="*60)
        print("PHASE 2: TESTING")
        print("="*60)

        # Initialize tester
        tester = ModelTester(device=trainer.device)

        # Prepare paths
        model_artifacts_path = f"{config['save_dir']}/best_training_artifacts.pkl"
        test_embeddings_path = config['test_embeddings_path']
        output_path = f"{config['save_dir']}/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Test model
        predicted_labels, true_labels = tester.test_model(
            model_artifacts_path,
            test_embeddings_path,
            output_path
        )

        # Compute and display metrics
        metrics = tester.compute_metrics(true_labels, predicted_labels)

        print("\n" + "="*60)
        print("EVALUATION METRICS:")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

        # Save metrics to file
        metrics_path = f"{config['save_dir']}/evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"All outputs saved in: {config['save_dir']}")

if __name__ == "__main__":
    main()
