# src/utils/dataset.py
import os
import numpy as np
from torch.utils.data import Dataset


class PoseDataset(Dataset):
"""Dataset that reads numpy feature files or preprocesses AlphaPose json on the fly.
Expected: each line in list file is a path to a feature .npy or an AlphaPose json
and label (optional) separated by comma: path,label
"""
def __init__(self, list_file, seq_len=64, transform=None):
self.samples = []
with open(list_file, 'r') as f:
for l in f:
l = l.strip()
if not l:
continue
if ',' in l:
path, label = l.split(',')
label = int(label)
else:
path = l
label = None
self.samples.append((path, label))
self.seq_len = seq_len
self.transform = transform


def __len__(self):
return len(self.samples)


def __getitem__(self, idx):
path, label = self.samples[idx]
if path.endswith('.npy'):
feats = np.load(path)
else:
# assume AlphaPose json -> use preprocess to convert
from preprocess import process_alphapose_json_to_features
feats = process_alphapose_json_to_features(path)
# feats: (T, D) -> ensure seq_len by padding or cropping
T, D = feats.shape
if T >= self.seq_len:
start = 0 if T==self.seq_len else np.random.randint(0, T - self.seq_len + 1)
feats = feats[start:start+self.seq_len]
else:
pad = np.zeros((self.seq_len - T, D), dtype=feats.dtype)
feats = np.concatenate([feats, pad], axis=0)
# convert to float32
feats = feats.astype('float32')
import torch
feats = torch.from_numpy(feats)
if label is None:
label = 0
return feats, torch.tensor(label, dtype=torch.long)
