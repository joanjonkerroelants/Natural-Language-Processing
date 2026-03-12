import torch
import torch.nn as nn


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
