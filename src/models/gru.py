import torch
import torch.nn as nn


class GRUModel(nn.Module):
def __init__(self, input_dim=34, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
super().__init__()
self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
self.fc = nn.Linear(hidden_dim, num_classes)


def forward(self, x):
out, _ = self.gru(x)
last = out[:, -1, :]
return self.fc(last)
