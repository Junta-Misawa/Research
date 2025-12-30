# TAM-Enhanced Semantic Segmentation with DINOv3

このリポジトリは、大規模言語モデル（Multimodal LLM）から得られる **Token Activation Map (TAM)** を活用し、**DINOv3** バックボーンと組み合わせることで、セマンティックセグメンテーションの精度を向上させるためのフレームワークです。特に、悪天候（雨、雪、霧）や夜間などの困難な条件下でのロバスト性を高めることを目的としています。

## 特徴

*   **TAM (Token Activation Map) の統合**: Qwen2-VL などの Multimodal LLM を使用して、画像内の特定のクラス（車、道路、人など）に対応する活性化マップ（TAM）を生成し、これをセグメンテーションモデルの入力として利用します。
*   **Frozen DINOv3 Backbone**: 強力な視覚特徴抽出器である DINOv3 を凍結（Frozen）状態で使用し、計算コストを抑えつつ高い表現力を活用します。
*   **Gated Fusion Mechanism**: DINOv3 の特徴量と TAM の特徴量を効果的に統合するためのゲート付き融合モジュール (`TAMGatedFusion`) を採用しています。
*   **多様なデータセット対応**: Cityscapes データセットに加え、C-driving (Cloudy, Rainy, Snowy, Overcast), Foggy Driving (FD), Nighttime Driving などの悪条件データセットでの実験をサポートしています。

## 環境構築

### 必要要件

*   Python 3.8+
*   PyTorch
*   Transformers
*   OpenCV, NumPy, Pillow

### インストール

リポジトリをクローンし、必要なパッケージをインストールしてください。

```bash
pip install -r requirements_seg.txt
# TAM生成に関連する追加パッケージが必要な場合
pip install -r TAM/requirements.txt
```

## データセットの準備

データセットは以下のディレクトリ構造で配置することを想定しています（`workspace_info`に基づく）。

```
Dataset/
├── cityscapes/
│   ├── gtFine_trainvaltest/
│   └── leftImg8bit_trainvaltest/
├── C-driving-cloudy/
├── C-driving-rainy/
├── C-driving-snowy/
├── NighttimeDrivingTest/
└── ...
```

## 使用方法

### 1. TAM (Token Activation Map) の事前計算

学習を行う前に、データセット内の画像に対する TAM を生成する必要があります。これは `precompute_tam.py` スクリプトを使用して行います。

**例: NighttimeDrivingTest データセットの TAM 生成**

複数のGPUを使用して並列処理を行うことが可能です（`--num_shards` と `--shard_id` を使用）。

```bash
# GPU 0 での処理 (前半)
CUDA_VISIBLE_DEVICES=0 python precompute_tam.py \
  --root /home/suzukilab/Research/Dataset/NighttimeDrivingTest \
  --split test \
  --out_dir /home/suzukilab/Research/TAM_maps/tam_maps_nd \
  --num_shards 2 \
  --shard_id 0 \
  --overlay_dir /home/suzukilab/Research/TAM_visualizations/tam_maps_nd

# GPU 1 での処理 (後半)
CUDA_VISIBLE_DEVICES=1 python precompute_tam.py \
  --root /home/suzukilab/Research/Dataset/NighttimeDrivingTest \
  --split test \
  --out_dir /home/suzukilab/Research/TAM_maps/tam_maps_nd \
  --num_shards 2 \
  --shard_id 1 \
  --overlay_dir /home/suzukilab/Research/TAM_visualizations/tam_maps_nd
```

*   `--root`: データセットのルートディレクトリ
*   `--split`: データセットの分割（train, val, test）
*   `--out_dir`: 生成された TAM (.npy) の保存先
*   `--overlay_dir`: 可視化画像（オーバーレイ）の保存先（オプション）

### 2. モデルの学習

`train_seg.py` を使用してセグメンテーションモデルを学習します。

#### 提案手法 (TAMを使用)

TAM を使用して学習する場合、`--tam_dir` (学習用) と `--tam_dir_val` (検証用) を指定します。

```bash
python train_seg.py \
  --root /home/suzukilab/Research/Dataset/cityscapes \
  --val_root /home/suzukilab/Research/Dataset/C-driving-overcast \
  --val_split test \
  --tam_dir /home/suzukilab/Research/TAM_maps/tam_maps_train \
  --tam_dir_val /home/suzukilab/Research/TAM_maps/tam_maps_cdo \
  --amp \
  --batch_size 8 \
  --epochs 32
```

#### ベースライン (TAMなし)

TAM を使用せず、DINOv3 + Linear Head のみで学習する場合は `--baseline_no_tam` フラグを使用します。

```bash
python train_seg.py \
  --root /home/suzukilab/Research/Dataset/cityscapes \
  --val_root /home/suzukilab/Research/Dataset/C-driving-overcast \
  --val_split test \
  --amp \
  --baseline_no_tam
```

**主な引数:**
*   `--root`: 学習データセットのルート
*   `--val_root`: 検証データセットのルート
*   `--tam_dir`: 学習用 TAM ディレクトリ
*   `--tam_dir_val`: 検証用 TAM ディレクトリ
*   `--dino_model`: 使用する DINOv3 モデル名 (デフォルト: `facebook/dinov3-vith16plus-pretrain-lvd1689m`)
*   `--amp`: 自動混合精度学習 (Automatic Mixed Precision) を有効化
*   `--save_dir`: チェックポイントの保存先 (デフォルト: `runs_seg`)

### 3. TAM の可視化

生成された TAM の `.npy` ファイルを可視化するには `visualize_tam_npy.py` を使用します。

```bash
python visualize_tam_npy.py
```

## ファイル構成

*   `train_seg.py`: 学習用メインスクリプト
*   `seg_model.py`: モデル定義 (FrozenDINOv3Backbone, TAMGatedFusion)
*   `precompute_tam.py`: TAM 生成用スクリプト
*   `cityscapes_seg.py`: データセットローダー
*   `metrics.py`: 評価指標 (mIoUなど) の計算
*   `TAM/`: TAM 関連のサブモジュール (Qwen2-VL ユーティリティなど)

## 謝辞

このプロジェクトは [TAM (Token Activation Map)](https://arxiv.org/abs/2506.23270) の技術をベースにしています。
