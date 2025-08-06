import argparse
import json
from datetime import datetime
from trainer import NeuralNetworkTrainer
from tester import ModelTester


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Test a Multi-label Neural Network")

    # Required
    parser.add_argument("--train_embeddings", type=str, required=True, help="Path to training embeddings .pkl")
    parser.add_argument("--test_embeddings", type=str, required=True, help="Path to test embeddings .pkl")
    parser.add_argument("--model_name", type=str, default="sent", help="Name identifier for the model")

    # Optional Hyperparameters
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=[4096, 2048], help="List of hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=75, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--early_stop", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Scheduler patience")
    parser.add_argument("--print_every", type=int, default=10, help="Print every N epochs")

    parser.add_argument("--run_testing", action="store_true", help="Run testing phase")
    parser.add_argument("--save_plots", action="store_true", help="Save training plots")

    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/neural_net_{timestamp}_{args.model_name}"

    config = {
        'train_embeddings_path': args.train_embeddings,
        'test_embeddings_path': args.test_embeddings,
        'hidden_layers': args.hidden_layers,
        'dropout_rate': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stop,
        'scheduler_patience': args.scheduler_patience,
        'print_every': args.print_every,
        'save_dir': save_dir,
        'run_testing': args.run_testing,
        'save_plots': args.save_plots
    }

    print(json.dumps(config, indent=2))

    # Train
    trainer = NeuralNetworkTrainer(config)
    trainer.load_embeddings(config['train_embeddings_path'])
    trainer.preprocess_data()
    trainer.create_model()
    trainer.train()
    trainer.evaluate_on_training_data()

    if config['save_plots']:
        trainer.plot_training_history()

    trainer.save_model()

    # Test
    if config['run_testing']:
        tester = ModelTester(device=trainer.device)
        model_artifacts_path = f"{config['save_dir']}/best_training_artifacts.pkl"
        output_path = f"{config['save_dir']}/predictions_{timestamp}.csv"
        predicted_labels, true_labels = tester.test_model(model_artifacts_path, config['test_embeddings_path'], output_path)
        metrics = tester.compute_metrics(true_labels, predicted_labels)

        print("\nEVALUATION METRICS:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()
