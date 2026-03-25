import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main import LABELS
from models.load import Preprocessing

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


class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.rep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))
        emb_t = emb.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(emb_t))
            p = torch.max(z, dim=2).values
            pooled.append(p)
        rep = torch.cat(pooled, dim=1)
        rep = self.rep_dropout(rep)
        return self.fc(rep)


class LSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.sequence = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(sentence)
        lstm_out, _ = self.lstm(emb)
        pool_max = torch.max(lstm_out, dim=1).values
        pool_mean = torch.mean(lstm_out, dim=1)
        concentrated = torch.cat((pool_max, pool_mean), dim=1)
        rep = self.sequence(concentrated)
        rep = self.dropout(rep)
        return self.fc(rep)


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
        y = int(row["label"]) - 1
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
            f"True Label: {LABELS.get(y + 1)} , Predicted Label: {LABELS.get(pred + 1)}"
        )
        print("\n" + "-" * 50 + "\n")

    return errs


def evaluate_neural_model(
    model: torch.nn.Module,
    dataset,
    vocab: dict[str, int],
    device: torch.device,
    max_len: int | None = None,
    dataset_name: str = "Test",
):
    """Evaluate a neural model on a labelled dataset split."""
    loader = DataLoader(
        NeuralDataset(dataset, vocab),
        batch_size=64,
        shuffle=False,
        collate_fn=make_collate(max_len),
    )

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch.x.to(device))
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = batch.y.cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"{dataset_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=[LABELS[i + 1] for i in range(len(LABELS))],
    )
    disp.plot(xticks_rotation="vertical")
    plt.title(
        f"Confusion Matrix: {model.__class__.__name__} on {dataset_name}"
    )
    plt.show()

    return {"accuracy": accuracy, "macro_f1": macro_f1}


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
        collate_fn=make_collate(args.max_len),
    )
    dev_loader = DataLoader(
        NeuralDataset(dev_dataset, vocab_dict),
        batch_size=args.batch_size,
        collate_fn=make_collate(args.max_len),
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


def make_collate(max_len: int | None = None):
    """Factory that returns a collate function with optional truncation."""

    def _collate(batch: list) -> Batch:
        # batch: list of (ids_list, label)
        lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
        effective_max_len = (
            min(int(lengths.max().item()), max_len)
            if max_len is not None
            else int(lengths.max().item())
        )
        x = torch.full(
            (len(batch), effective_max_len), PAD_IDX, dtype=torch.long
        )
        y = torch.tensor([int(y) for _, y in batch], dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, : len(ids[:effective_max_len])] = torch.tensor(
                ids[:effective_max_len], dtype=torch.long
            )
        return Batch(x=x, lengths=lengths.clamp(max=effective_max_len), y=y)

    return _collate
