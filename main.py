import argparse

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from models import tfidf, cnn
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
        ratio=config["model"]["tfidf"]["split"]["train_size"],
    )
    train_dataset, dev_dataset, test_dataset = loader.split()

    return train_dataset, dev_dataset, test_dataset


def evaluate_model(model, X_test, y_test, dataset_name: str = "Test"):
    """
    Evaluate model and return metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

    disp.plot(xticks_rotation="vertical")
    plt.title("Confusion Matrix: TF-IDF + Logistic Regression")
    plt.show()


def error_analysis(test_df, y_test, X_test, model):
    """Perform error analysis by displaying misclassified examples.

    Converts numeric labels back to readable text using `LABELS` and
    compares true vs predicted labels.
    """

    true_labels_text = [
        LABELS.get(int(i) + 1, str(int(i) + 1)) for i in y_test
    ]

    pred_numeric = model.predict(X_test)
    pred_labels_text = [
        LABELS.get(int(i) + 1, str(int(i) + 1)) for i in pred_numeric
    ]

    df_predictions = pd.DataFrame(
        {
            "text": test_df["description"].values,
            "true_label": true_labels_text,
            "pred_label": pred_labels_text,
        }
    )

    errors = df_predictions[
        df_predictions["true_label"] != df_predictions["pred_label"]
    ]

    for e in errors.head(20).itertuples():
        print(f"Text: {e.text}")
        print(f"True Label: {e.true_label}, Predicted Label: {e.pred_label}")
        print("\n" + "-" * 50 + "\n")


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
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

        X_train = vectorizer.fit_transform(train_df["description"])
        X_dev = vectorizer.transform(dev_df["description"])
        X_test = vectorizer.transform(test_df["description"])

        y_train = train_df["label"].values - 1
        y_dev = dev_df["label"].values - 1
        y_test = test_df["label"].values - 1

        model = tfidf.train_model(args.architecture, X_train, y_train)

        print("Performing error analysis on Test set:")
        error_analysis(test_df, y_test, X_test, model)

        print("Evaluating Dev:")
        dev_results = evaluate_model(model, X_dev, y_dev, "Dev")

        print("Evaluating Test:")
        test_results = evaluate_model(model, X_test, y_test, "Test")
    elif args.model == "neural":
        if args.architecture == "lstm":
            print("Training LSTM model...")
            model = cnn.LSTMTextClassifier(vocab_size)
        elif args.architecture == "cnn":
            print("Training CNN model...")
            model = cnn.CNNTextClassifier(vocab_size)