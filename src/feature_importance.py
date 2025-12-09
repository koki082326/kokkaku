import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from src.utils.dataset import PoseDataset

def compute_feature_importance(list_file):
    ds = PoseDataset(list_file)
    X = np.array([x.numpy() for x, y in ds])
    y = np.array([y.numpy() for x, y in ds])

    # RandomForest で簡易的に特徴量重要度を確認
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X.reshape(X.shape[0], -1), y)
    importances = clf.feature_importances_
    return importances

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, required=True)
    args = parser.parse_args()
    imp = compute_feature_importance(args.list)
    print('Feature importances:', imp)
