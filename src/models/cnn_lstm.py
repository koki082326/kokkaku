import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
def __init__(self, input_dim=34, conv_channels=64, kernel_size=3, hidden_dim=128, num_layers=1, num_classes=2):
super().__init__()
self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size//2)
self.relu = nn.ReLU()
self.pool = nn.MaxPool1d(2)
self.lstm = nn.LSTM(conv_channels, hidden_dim, num_layers=num_layers, batch_first=True)
self.fc = nn.Linear(hidden_dim, num_classes)


def forward(self, x):
# x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
x = x.permute(0, 2, 1)
x = self.conv1(x)
x = self.relu(x)
x = self.pool(x)
x = x.permute(0, 2, 1)
out, _ = self.lstm(x)
last = out[:, -1, :]
return self.fc(last)
