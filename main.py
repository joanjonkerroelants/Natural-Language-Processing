import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import cnn, tfidf
from models.load import Preprocessing, loading

LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

PAD = "<pad>"
UNK = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


@dataclass
class Batch:
    x: torch.Tensor  # (B, T) token ids
    lengths: torch.Tensor  # (B,) true lengths
    y: torch.Tensor  # (B,) labels


def tokenize(text: str) -> list[str]:
    return Preprocessing(text).tokenize()


class NeuralDataset(Dataset):
    """Wraps a DatasetNews split and converts token strings to vocab indices."""

    def __init__(self, dataset, vocab: dict[str, int]) -> None:
        self.dataset = dataset
        self.vocab = vocab
        self.unk_idx = vocab[UNK]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        ids = [self.vocab.get(t, self.unk_idx) for t in item["tokens"]]
        return ids, item["label"]


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


def build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict:
    """
    Build a vocabulary mapping from tokens to integer indices.
    The vocabulary will include only tokens that appear at least `min_freq` times,
    and will be limited to `max_size` tokens (including PAD and UNK).
    """
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    # Reserve 0 for PAD and 1 for UNK.
    vocab = {PAD: PAD_IDX, UNK: UNK_IDX}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def collate(batch: list) -> Batch:
    """Collate function to convert a list of samples into a batch."""
    # batch: list of (ids_list, label)
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = (
        min(int(lengths.max().item()), args.max_len)
        if args.max_len
        else int(lengths.max().item())
    )
    x = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    y = torch.tensor([y for _, y in batch], dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids[:max_len])] = torch.tensor(
            ids[:max_len], dtype=torch.long
        )
    return Batch(x=x, lengths=lengths.clamp(max=max_len), y=y)


def train_neural(
    model: torch.nn.Module,
    train_dataset,
    dev_dataset,
    vocab_dict: dict[str, int],
    args,
    device: torch.device,
) -> torch.nn.Module:
    """Training loop with early stopping for neural models."""
    train_loader = DataLoader(
        NeuralDataset(train_dataset, vocab_dict),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        NeuralDataset(dev_dataset, vocab_dict),
        batch_size=args.batch_size,
        collate_fn=collate,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())

    train_losses: list[float] = []
    val_losses: list[float] = []
    dev_accs: list[float] = []
    best_epoch = 0

    saved_models_dir = Path("model_states")
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            X_batch, y_batch = batch.x.to(device), batch.y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch} - Validation"):
                X_batch, y_batch = batch.x.to(device), batch.y.to(device)
                logits = model(X_batch)
                val_loss += loss_fn(logits, y_batch).item()
                correct += (logits.argmax(dim=1) == y_batch).sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(dev_loader)
        val_acc = correct / len(dev_dataset)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        dev_accs.append(val_acc)

        print(
            f"Epoch {epoch:>3}/{args.epochs} "
            f"| Train Loss: {avg_train:.4f} "
            f"| Val Loss: {avg_val:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                best_state, saved_models_dir / f"{args.architecture}_best.pt"
            )
            best_epoch = epoch - 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best accuracy: {dev_accs[best_epoch]:.4f}")
                break

    model.load_state_dict(best_state)
    _save_learning_curves(train_losses, dev_accs, args.architecture)
    return model


def _save_learning_curves(
    train_losses: list[float], dev_accs: list[float], architecture: str
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(epochs, dev_accs, label="Dev Accuracy", color="tab:orange")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle(f"{architecture.upper()} Learning Curves")
    fig.tight_layout()

    out_path = Path("output_images") / f"{architecture}_learning_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Learning curves saved to {out_path}")


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
    plt.title(
        f"Confusion Matrix: {model.__class__.__name__} on {dataset_name}"
    )
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


def numericalize(tokens: list, vocab: dict) -> list:
    """
    Convert a list of tokens into a list of integer indices using the provided vocabulary.
    Tokens not found in the vocabulary will be mapped to the index of UNK.
    """
    return [vocab.get(tok, vocab[UNK]) for tok in tokens]


def error_analysis_neural(
    model: torch.nn.Module,
    dataset,
    vocab: dict[str, int],
    max_len: int,
    device: torch.device,
    max_items: int = 20,
):
    """Perform error analysis for neural models."""
    model.eval()
    errs: list[tuple[int, int, str]] = []

    for _, row in dataset.df.iterrows():
        text = row["description"]
        tokens = tokenize(row["description"])
        ids = numericalize(tokens, vocab)[:max_len]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        y = int(row["label"])
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
        if pred != y:
            snippet = text.replace("\n", " ")
            snippet = snippet[:250] + ("..." if len(snippet) > 250 else "")
            errs.append((y, pred, snippet))
        if len(errs) >= max_items:
            break

    for y, pred, snippet in errs:
        print(f"Text: {snippet}")
        print(
            f"True Label: {LABELS.get(y)} , Predicted Label: {LABELS.get(pred)}"
        )
        print("\n" + "-" * 50 + "\n")

    return errs


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
        evaluate_model(model, X_dev, y_dev, "Dev")

        # print("Evaluating Test:")
        # evaluate_model(model, X_test, y_test, "Test")
    elif args.model == "neural":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        texts = train_df["description"]
        vocab_dict = build_vocab(texts)
        vocab_size = len(vocab_dict)

        if args.architecture == "lstm":
            print("Training LSTM model...")
            model = cnn.LSTMTextClassifier(vocab_size, dropout=args.dropout)
            model = train_neural(
                model, train_dataset, dev_dataset, vocab_dict, args, device
            )
        elif args.architecture == "cnn":
            print("Training CNN model...")
            model = cnn.CNNTextClassifier(vocab_size, dropout=args.dropout)
            model = train_neural(
                model, train_dataset, dev_dataset, vocab_dict, args, device
            )
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        print("Performing error analysis on Test set:")
        error_analysis_neural(
            model, test_dataset, vocab_dict, args.max_len, device
        )
