import logging
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)


class LSTM_Fixed(nn.Module):
    """
    LSTM (Fixed Size) - An LSTM Model with a Linear Classifier.

    Summary:
    (nn.Embedding, nn.Dropout, nn.LSTM, nn.Linear)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, drop=0.2):
        super(LSTM_Fixed, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop = drop

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x, *args):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


class LSTM_Padded(nn.Module):
    """
    LSTM (Padded Size) - An LSTM Model with a Linear Classifier.

    Summary:
    (nn.Embedding, nn.Dropout, nn.utils.rnn.pack_padded_sequence, nn.LSTM, nn.Linear)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, drop=0.2):
        super(LSTM_Padded, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop = drop

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


def validate_model(model: nn.Module, val_loader: DataLoader):
    model.eval()
    sum_corr, total = 0, 0
    sum_loss = 0
    sum_rmse = 0

    for x, y, l in val_loader:
        x = x.long()
        y = y.long()

        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]

        total += y.shape[0]
        sum_corr += (pred == y).float().sum()
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1))) * y.shape[0]

    return (
        sum_loss / total,
        sum_corr / total,
        sum_rmse / total
    )


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 10,
                lr: float = 0.001,
                prompt_epoch: int = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        model.train()
        sum_loss = 0
        total = 0

        for x, y, l in train_loader:
            x = x.long()
            y = y.long()

            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

        val_loss, val_acc, val_rmse = validate_model(model, val_loader)
        if i % prompt_epoch == 0:
            curr_time = datetime.now()
            curr_loss = sum_loss / total

            logging.info(
                f'{curr_time.strftime("%H:%M:%S")}: Train Loss={curr_loss:.3f}, Validation Loss={val_loss:.3f}, Validation Acc={val_acc:.3f}, Val RMSE={val_rmse:.3f}, T:{total}'
            )


def predict_model(model: nn.Module, data_loader: DataLoader):
    model.eval()
    output = []

    for x, y, l in data_loader:
        y_hat = model(x, l)
        pred = torch.max(y_hat, 1)[1]
        output.append(pred.item())

    return output
