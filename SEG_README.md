## TAM 前計算
```
# プロセス1（GPU0）
CUDA_VISIBLE_DEVICES=0 python precompute_tam.py \
  --root /home/suzukilab/Research/Dataset/NighttimeDrivingTest \
  --split test \
  --out_dir /home/suzukilab/Research/TAM_maps/tam_maps_nd \
  --num_shards 2 \
  --shard_id 0 \
  --overlay_dir /home/suzukilab/Research/TAM_visualizations/tam_maps_nd

# プロセス2（GPU1）
CUDA_VISIBLE_DEVICES=1 python precompute_tam.py \
  --root /home/suzukilab/Research/Dataset/NighttimeDrivingTest \
  --split test \
  --out_dir /home/suzukilab/Research/TAM_maps/tam_maps_nd \
  --num_shards 2 \
  --shard_id 1 \
  --overlay_dir /home/suzukilab/Research/TAM_visualizations/tam_maps_nd
```

## TAM可視化
```
python visualize_tam_npy.py
```

## 学習
```
# 提案手法
python train_seg.py \
  --root /home/suzukilab/Research/Dataset/cityscapes \
  --val_root /home/suzukilab/Research/Dataset/C-driving-overcast \
  --val_split test \
  --tam_dir /home/suzukilab/Research/TAM_maps/tam_maps_train \
  --tam_dir_val /home/suzukilab/Research/TAM_maps/tam_maps_cdo \
  --amp

  

# ベースライン
python train_seg.py \
  --root /home/suzukilab/Research/Dataset/cityscapes \
  --val_root /home/suzukilab/Research/Dataset/C-driving-overcast \
  --val_split test \
  --amp \
  --baseline_no_tam
```

## テスト
```
# 提案手法
python test_seg.py \
  --benchmark_all \
  --checkpoint /home/suzukilab/Research/runs_seg/best.pt \
  --save_vis \
  --vis_dir /home/suzukilab/Research/Vis_Seg
  

# ベースライン
python test_seg.py \
  --benchmark_all \
  --checkpoint /home/suzukilab/Research/runs_seg/best.pt \
  --baseline_no_tam \
  --save_vis \
  --vis_dir /home/suzukilab/Research/Vis_Seg
```