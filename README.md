
# kokkaku: Hybrid Fall Detection Pipeline using AlphaPose and Deep Learning

This repository contains a hybrid fall-detection pipeline using AlphaPose and deep learning.
The system extracts human skeletons from video, preprocesses pose sequences, 
and classifies falls vs non-falls using LSTM/GRU/CNN-LSTM models.

kokkaku は、**AlphaPose を用いた骨格推定**と  
**LSTM / GRU / CNN-LSTM を用いた時系列解析**により  
転倒・非転倒を高精度に分類するためのハイブリッド転倒検知パイプラインです。

本研究では、RGB動画から骨格情報を抽出し、  
姿勢角度・速度・加速度などの特徴量を用いて、  
転倒に伴う連続的な崩れをモデル化します。

---

## 🚀 Features

- **AlphaPose による高精度なキーポイント抽出**
- **Pose JSON → NumPy の標準化前処理パイプライン**
- **LSTM / GRU / CNN-LSTM による時系列分類**
- **角度・速度・加速度の特徴量抽出**
- **フレーム長による性能比較**
- **SHAP / Grad-CAM による特徴量寄与と重要関節の可視化**
- **軽量かつ拡張性のある構成（dataset, preprocess, model, inference）**

---

## 📁 Project Structure

# kokkaku: Skeleton-based Fall Detection Pipeline

## 📌 Overview

kokkaku は、**AlphaPose を用いた骨格推定**と、**LSTM / GRU / CNN-LSTM 時系列モデル**による転倒検知を行う研究目的のフルパイプライン実装です。

このリポジトリは、以下の処理を一貫して実行できます：

1. **Pose Extraction** ‑ AlphaPose を用いて動画から 17/133 keypoints を JSON として抽出
2. **Preprocessing** ‑ 正規化・補間・角度/速度特徴抽出など
3. **Dataset Loader** ‑ シーケンスごとにデータを PyTorch Dataset 化
4. **Modeling** ‑ LSTM / GRU / CNN-LSTM による分類
5. **Training** ‑ GPU での学習・ログ保存
6. **Inference** ‑ 1動画を与えて転倒/非転倒の出力
7. **Feature Importance Analysis**（RandomForest / SHAP）
8. **Visualization** ‑ Grad-CAM, 時間方向の寄与度分析

---

## 🏗 Architecture

```
kokkaku/
│
├─ src/
│  ├─ pose_extraction/   → AlphaPose による keypoint JSON 抽出
│  ├─ preprocess/        → 補間 / 正規化 / 関節角度・速度特徴生成
│  ├─ models/            → LSTM / GRU / CNN-LSTM
│  ├─ train/              → 学習コード(train.py)
│  ├─ inference/         → 推論コード(inference.py)
│  ├─ analysis/          → 特徴量重要度(RF/SHAP)
│  └─ utils/             → Dataset, 共通関数
│
├─ notebooks/            → 可視化 & 検証用 Jupyter
├─ samples/              → サンプル動画・JSON
└─ README.md
```

---

## ✨ Research Background

高齢者の転倒は重大な事故の原因であり、監視カメラ映像を用いた**非接触の転倒検知**が強く求められている。本研究では RGB 動画からの骨格情報に基づく転倒検知に着目し、既存研究では困難であった「**転倒直前の姿勢崩れ（姿勢角度 + 速度変化）**」を高精度に捉えることを目的とする。

本手法では AlphaPose を用いて keypoint を抽出し、neck（肩の中点）を基準座標とした正規化を行う。続いて、関節角度、頭部/腰部の速度、体幹傾斜などの特徴量を生成し、これらを **LSTM 系モデル**へ入力することで、転倒の時間的パターンを学習させる。

特に CNN-LSTM は、短期的な局所特徴（姿勢角度の急変）と長期時系列特徴（段階的な姿勢崩れ）を同時に学習でき、転倒直前の一連の動きをより高精度に捉える。

---

## 🚀 Quick Start

### 1. Clone

```
git clone https://github.com/koki082326/kokkaku.git
cd kokkaku
```

### 2. Install

```
pip install -r requirements.txt
```

### 3. Run Training

```
python src/train/train.py \
  --data_dir dataset/processed \
  --model lstm
```

### 4. Run Inference

```
python src/inference/inference.py --video samples/sample_video.mp4
```

---

## 🧠 Models

### LSTM

* 長期依存を捉える
* 姿勢崩れの連続性を学習

### GRU

* LSTM の軽量版
* 少データでも安定

### CNN-LSTM

* 空間特徴（フレーム内）と時系列特徴（フレーム間）を同時に学習
* 本研究で最も高精度

---

## 🧪 Feature Engineering

使用した特徴量：

* neck を基準とした 2D 座標（正規化）
* hip / head / shoulder などの速度
* 体幹角度（neck‑hip‑ankle）
* 関節角度（肘・膝）
* 姿勢崩壊度（Angle Velocity）

---

## 📊 Feature Importance

```
src/analysis/feature_importance.py
```

で以下が可能：

* RandomForest によるランキング
* SHAP による個別フレーム寄与度の可視化

---

## 📝 Citation

（論文化したときに追記）

---

## 📄 License

MIT
