import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def inspect_segmentation_label(image_path):
    """
    セグメンテーション画像のラベル情報を解析・可視化する関数
    """
    if not os.path.exists(image_path):
        print(f"エラー: ファイルが見つかりません -> {image_path}")
        return

    # 1. 画像の読み込み
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 2. 基本情報の表示
    print("-" * 30)
    print(f"【画像情報】")
    print(f"ファイル名: {os.path.basename(image_path)}")
    print(f"サイズ (H, W): {img_array.shape[:2]}")
    print(f"チャンネル数: {img_array.shape[2] if len(img_array.shape) > 2 else 1}")
    print(f"画像モード: {img.mode}")
    print("-" * 30)

    # 3. ユニークな値（クラスID）の解析
    # RGB画像ではなく、クラスIDが入ったマップ(2D配列)の場合のみ解析
    if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img.mode == 'P'):
        unique_values, counts = np.unique(img_array, return_counts=True)
        total_pixels = img_array.size
        
        print(f"【含まれているクラスID (Pixel Values)】")
        print(f"{'Class ID':<10} | {'Count':<10} | {'Ratio (%)':<10}")
        print("-" * 36)
        
        for val, count in zip(unique_values, counts):
            ratio = (count / total_pixels) * 100
            print(f"{val:<10} | {count:<10} | {ratio:.2f}%")
            
        # 4. 可視化 (Matplotlib)
        plt.figure(figsize=(12, 6))
        
        # 左側: 生の画像（値が小さいと真っ黒に見える）
        plt.subplot(1, 2, 1)
        plt.imshow(img_array, cmap='gray')
        plt.title(f"Raw Image (Gray)\nMode: {img.mode}")
        plt.axis('off')
        
        # 右側: クラスごとに色分け（見やすく強調）
        plt.subplot(1, 2, 2)
        # 'nipy_spectral' や 'tab20' は異なる数値を異なる色にするのに適しています
        plt.imshow(img_array, cmap='nipy_spectral', interpolation='nearest')
        plt.colorbar(label='Class ID')
        plt.title("Colorized Visualization\n(Mapped by Class ID)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    else:
        # RGB画像（すでに色付けされているラベルなど）の場合
        print("注意: この画像は3チャンネル(RGB)です。ピクセル値がクラスIDではなく色情報の可能性があります。")
        print(f"ユニークな色の数: {len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array)
        plt.title("RGB Label Image")
        plt.axis('off')
        plt.show()

# ==========================================
# 使い方: ここに画像のパスを指定してください
# ==========================================
target_path = "/home/suzukilab/Research/Dataset/C-driving-cloudy/gtFine/test/14020ffa-b52ddff6_train_id.png"  # ← ここを書き換える

inspect_segmentation_label(target_path)
