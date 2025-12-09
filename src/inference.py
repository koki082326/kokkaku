# src/inference.py
import argparse
import numpy as np
import torch
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.cnn_lstm import CNNLSTMModel




def load_model(model_name, model_path, device, cfg):
if model_name == 'LSTM':
model = LSTMModel(input_dim=cfg['model']['input_dim'], hidden_dim=cfg['model']['hidden_dim'], num_classes=cfg['model']['num_classes'])
elif model_name == 'GRU':
model = GRUModel(input_dim=cfg['model']['input_dim'], hidden_dim=cfg['model']['hidden_dim'], num_classes=cfg['model']['num_classes'])
else:
model = CNNLSTMModel(input_dim=cfg['model']['input_dim'], hidden_dim=cfg['model']['hidden_dim'], num_classes=cfg['model']['num_classes'])
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
return model




def infer(model, feats):
# feats: numpy (T, D) -> convert to tensor (1, seq_len, D)
import torch
x = torch.from_numpy(feats.astype('float32')).unsqueeze(0)
with torch.no_grad():
logits = model(x)
prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
return prob




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='LSTM')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--config', type=str, default='config/train_config.yaml')
args = parser.parse_args()


import yaml
with open(args.config) as f:
cfg = yaml.safe_load(f)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(args.model, args.model_path, device, cfg)


# load features
if args.input.endswith('.npy'):
feats = np.load(args.input)
else:
from preprocess import process_alphapose_json_to_features
feats = process_alphapose_json_to_features(args.input)


prob = infer(model, feats)
print('probabilities:', prob)


if __name__ == '__main__':
main()
