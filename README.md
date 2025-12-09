# kokkaku
This repository contains a hybrid fall-detection pipeline using AlphaPose and deep learning.
The system extracts human skeletons from video, preprocesses pose sequences, 
and classifies falls vs non-falls using LSTM/GRU/CNN-LSTM models.
# kokkaku: Hybrid Fall Detection Pipeline using AlphaPose and Deep Learning

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

