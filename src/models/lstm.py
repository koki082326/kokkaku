import torch
import torch.nn as nn


class LSTMModel(nn.Module):
def __init__(self, input_dim=34, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
super().__init__()
self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
self.fc = nn.Linear(hidden_dim, num_classes)


def forward(self, x):
# x: (batch, seq_len, input_dim)
out, _ = self.lstm(x)
last = out[:, -1, :]
return self.fc(last)
