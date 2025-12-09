# src/train.py


def main():
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/train_config.yaml')
parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM','GRU','CNNLSTM'])
args = parser.parse_args()


cfg = None
import yaml
with open(args.config) as f:
cfg = yaml.safe_load(f)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset
train_ds = PoseDataset(cfg['data']['train_list'], seq_len=cfg['model']['seq_len'])
val_ds = PoseDataset(cfg['data']['val_list'], seq_len=cfg['model']['seq_len'])


train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2)


# model
if args.model == 'LSTM':
model = LSTMModel(input_dim=cfg['model']['input_dim'], hidden_dim=cfg['model']['hidden_dim'], num_classes=cfg['model']['num_classes'])
elif args.model == 'GRU':
model = GRUModel(input_dim=cfg['model']['input_dim'], hidden_dim=cfg['model']['hidden_dim'], num_classes=cfg['model']['num_classes'])
else:
model = CNNLSTMModel(input_dim=cfg['model']['input_dim'], hidden_dim=cfg['model']['hidden_dim'], num_classes=cfg['model']['num_classes'])


model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])


best_acc = 0.0
save_dir = Path(cfg['train']['save_dir'])
save_dir.mkdir(parents=True, exist_ok=True)


for epoch in range(cfg['train']['epochs']):
train_loss = train_one_epoch(model, train_loader, optimizer, device)
val_acc, preds, trues = eval_model(model, val_loader, device)
print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")


# save best
if val_acc > best_acc:
best_acc = val_acc
torch.save(model.state_dict(), save_dir / f"best_{args.model}.pth")


print('Training finished. Best val acc:', best_acc)




if __name__ == '__main__':
main()
