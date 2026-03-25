import argparse

import torch
import yaml

from models import neural, tfidf
from models.load import loading

LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_config(config_path) -> dict:
    "loads configuration from a yaml file"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_parser(config) -> argparse.ArgumentParser:
    "builds the parser based on config"
    parser = argparse.ArgumentParser(
        description="Pipeline for multiple models"
    )

    subparsers = parser.add_subparsers(dest="model", required=True)

    for model_name, model_cfg in config["model"].items():
        sp = subparsers.add_parser(model_name, help=f"{model_name} model")

        # Architecture choices
        sp.add_argument(
            "architecture",
            choices=model_cfg["architectures"],
            help=f"Architecture for {model_name}",
        )

        # dataset only change for final
        sp.add_argument(
            "--path",
            default=config["dataset"]["train"]["path"],
            help="Dataset path (default: train)",
        )

        # training hyperparameters
        sp.add_argument(
            "--lr",
            type=float,
            default=config["training"]["lr"],
            help="Learning rate (default: 0.001)",
        )

        sp.add_argument(
            "--batch_size",
            type=int,
            default=config["training"]["batch_size"],
            help="Batch size (default: 4)",
        )

        sp.add_argument(
            "--epochs",
            type=int,
            default=config["training"]["epochs"],
            help="Number of epochs (default: 80)",
        )

        sp.add_argument(
            "--patience",
            type=int,
            default=config["training"]["patience"],
            help="Early stopping patience (default: 10)",
        )

        sp.add_argument(
            "--max-len",
            type=int,
            default=config["training"]["max_len"],
            help="Maximum token sequence length; longer sequences are truncated (default: 128)",
        )

        sp.add_argument(
            "--dropout",
            type=float,
            default=config["training"]["dropout"],
            help="Dropout probability (default: 0.3)",
        )

        # model details
        sp.add_argument(
            "--verbose",
            choices=config["verbosity"]["level"],
            type=str,
            default="low",
            help="prints all model details (default: low)",
        )

    return parser


def details_model(args) -> None:
    """
    Prints details about the models and their architectures.
    """
    print(f"Model: {args.model}")
    print(f"Architecture: {args.architecture}")

    if args.verbose == "high":
        print(f"Dataset path: {args.path}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Patience: {args.patience}")
    elif args.verbose == "medium":
        print(f"Learning rate: {args.lr}")
    else:
        pass


def load_data(config) -> tuple:
    """
    Load train, dev, and test datasets.
    """

    loader = loading(
        dataset=config["dataset"]["train"]["path"],
        seed=42,
        ratio=config["dataset"]["split"]["train_size"],
    )
    train_dataset, dev_dataset, test_dataset = loader.split()

    return train_dataset, dev_dataset, test_dataset


if __name__ == "__main__":
    config = load_config("config.yaml")
    parser = build_parser(config)
    args = parser.parse_args()
    details_model(args)

    print("Loading data...")
    train_dataset, dev_dataset, test_dataset = load_data(config)

    train_df = train_dataset.df
    dev_df = dev_dataset.df
    test_df = test_dataset.df

    if args.model == "tfidf":
        print("Training TF-IDF model...")
        X_train, X_dev, X_test = tfidf.vectorize_datasets(
            train_df, dev_df, test_df
        )
        y_train, y_dev, y_test = tfidf.extract_labels(
            train_df, dev_df, test_df
        )

        model = tfidf.train_model(args.architecture, X_train, y_train)

        print("Performing error analysis on Test set:")
        tfidf.error_analysis(test_df, y_test, X_test, model)

        print("Evaluating Dev:")
        tfidf.evaluate_model(model, X_dev, y_dev, "Dev")

        # print("Evaluating Test:")
        # evaluate_model(model, X_test, y_test, "Test")
    elif args.model == "neural":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        texts = train_df["description"]
        vocab_dict = neural.build_vocab(texts)
        vocab_size = len(vocab_dict)

        if args.architecture == "lstm":
            print("Training LSTM model...")
            model = neural.LSTMTextClassifier(vocab_size, dropout=args.dropout)
            model = neural.train_neural(
                model, train_dataset, dev_dataset, vocab_dict, args, device
            )
        elif args.architecture == "cnn":
            print("Training CNN model...")
            model = neural.CNNTextClassifier(vocab_size, dropout=args.dropout)
            model = neural.train_neural(
                model, train_dataset, dev_dataset, vocab_dict, args, device
            )
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        print("Evaluating Dev:")
        neural.evaluate_neural_model(
            model, dev_dataset, vocab_dict, device, args.max_len, "Dev"
        )

        print("Evaluating Test:")
        neural.evaluate_neural_model(
            model, test_dataset, vocab_dict, device, args.max_len, "Test"
        )

        print("Performing error analysis on Test set:")
        neural.error_analysis_neural(
            model, test_dataset, vocab_dict, args.max_len, device
        )
