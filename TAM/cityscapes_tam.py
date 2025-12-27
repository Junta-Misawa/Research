import os
import glob
import cv2
import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText

from .tam import TAM
from .eval import ids_to_word_groups  # 再利用（単語グループ化）


# Cityscapes 19クラス（trainId順）
CITYSCAPES_CATEGORY: Dict[str, int] = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'wall': 3,
    'fence': 4,
    'pole': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person': 11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorcycle': 17,
    'bicycle': 18,
}


def build_cityscapes_pairs(root: str, split: str = 'val') -> List[tuple]:
    """Cityscapes画像一覧を取得（マスクは任意）。

    優先パターン:
      - <root>/leftImg8bit/<split>/<city>/*_leftImg8bit.png
      - <root>/leftImg8bit_trainvaltest/leftImg8bit/<split>/<city>/*_leftImg8bit.png
      - フォールバック: <root>/leftImg8bit/<split>/<city>/*.png

    マスクは存在すれば推定パスを添えるが、未使用のため存在しなくても追加。
    Returns: list of (image_path, optional_label_path or None)
    """
    img_patterns = [
        # 標準: 2階層（city/filename）
        os.path.join(root, 'leftImg8bit', split, '*', '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit', split, '*', '*.png'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*.png'),
        # 1階層（FDやFDDなどcityサブフォルダ無し）
        os.path.join(root, 'leftImg8bit', split, '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit', split, '*.png'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*.png'),
        # JPG support
        os.path.join(root, 'leftImg8bit', split, '*', '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*', '*.jpg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*.jpg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*.jpg'),
    ]
    seen = set()
    img_files: List[str] = []
    for pat in img_patterns:
        for p in glob.glob(pat):
            if p not in seen:
                seen.add(p)
                img_files.append(p)

    pairs = []
    for img in img_files:
        # 推定される複数のmaskパス候補（存在しなくても可）
        ext = os.path.splitext(img)[1]
        suffix = '_leftImg8bit' + ext
        mask_candidates = [
            img.replace('leftImg8bit_trainvaltest/leftImg8bit', 'gtFine_trainvaltest/gtFine').replace(suffix, '_gtFine_labelIds.png'),
            img.replace('leftImg8bit', 'gtFine').replace(suffix, '_gtFine_labelIds.png'),
            img.replace('leftImg8bit_trainvaltest/leftImg8bit', 'gtFine_trainvaltest/gtFine').replace(suffix, '_gtFine_trainIds.png'),
            img.replace('leftImg8bit', 'gtFine').replace(suffix, '_gtFine_trainIds.png'),
        ]
        # JPGの場合、拡張子置換のフォールバックも追加（_leftImg8bitサフィックスが無い場合など）
        if ext.lower() == '.jpg':
            mask_candidates.extend([
                img.replace('leftImg8bit_trainvaltest/leftImg8bit', 'gtFine_trainvaltest/gtFine').replace(ext, '.png'),
                img.replace('leftImg8bit', 'gtFine').replace(ext, '.png')
            ])

        mask = None
        for m in mask_candidates:
            if os.path.exists(m):
                mask = m
                break
        pairs.append((img, mask))
    return pairs


def aggregate_token_maps(img_maps: List[np.ndarray], tokens: List[int], processor, target_classes: Dict[str, int]) -> Dict[str, np.ndarray]:
    """単語グループ化後、クラス名に一致する語のトークンマップを集約。

    集約方法: 同一語が複数トークンで構成 → そのトークンマップをリサイズ後 pixel-wise max で統合。
    返却: {class_name: 2D activation map (float32 0-1)} （存在しない場合は除外）
    """
    # 注意: ids_to_word_groups は batch_decode を前提としているため、[tokens] で渡す
    words, token_groups = ids_to_word_groups([tokens], processor)
    # 正規化用: 各 img_map を 0-1 にスケーリング
    norm_maps = []
    for m in img_maps:
        m = m.astype('float32')
        if m.max() > m.min():
            m = (m - m.min()) / (m.max() - m.min())
        else:
            m = np.zeros_like(m, dtype='float32')
        norm_maps.append(m)

    # vision token grid サイズ取得
    h_t, w_t = img_maps[0].shape

    # 正規化関数（簡易レmmatize: 語尾の複数形s/es除去、例外人称）
    def normalize_word(w: str) -> str:
        w = w.lower().strip().replace('-', '')
        # 特殊: people -> person
        if w == 'people':
            return 'person'
        # 語尾es
        if w.endswith('es') and len(w) > 3:
            return w[:-2]
        # 語尾s
        if w.endswith('s') and len(w) > 3:
            return w[:-1]
        return w

    # クラス正規化キー（連結文字列）と語数
    cls_keys = {}
    for cls_name in target_classes.keys():
        parts = [normalize_word(p) for p in cls_name.split()]
        cls_keys[''.join(parts)] = (cls_name, len(parts))

    out: Dict[str, np.ndarray] = {}
    n = len(words)
    # 1語/2語フレーズで走査
    for i in range(n):
        for L in (1, 2):
            if i + L > n:
                continue
            parts = [normalize_word(w) for w in words[i:i+L]]
            key = ''.join(parts)
            if key in cls_keys and cls_keys[key][1] == L:
                cls_name = cls_keys[key][0]
                # 既に割当済みならスキップ（最初の一致を採用）
                if cls_name in out:
                    continue
                # トークン群マップを max 集約（フレーズ内の全トークン）
                agg = None
                for j in range(i, i+L):
                    for idx in token_groups[j]:
                        m = norm_maps[idx]
                        agg = m.copy() if agg is None else np.maximum(agg, m)
                if agg is not None:
                    out[cls_name] = agg
    return out


def overlay_and_save(raw_img: Image.Image, act_map: np.ndarray, save_path: str, cmap=cv2.COLORMAP_JET):
    """Activationマップを元画像に重ね合わせて保存。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_rgb = np.array(raw_img.convert('RGB'))
    h, w, _ = img_rgb.shape
    # トークングリッドを画像サイズへリサイズ
    act = cv2.resize(act_map, (w, h), interpolation=cv2.INTER_LINEAR).astype('float32')
    act_u8 = (act * 255).clip(0, 255).astype('uint8')
    heat = cv2.applyColorMap(act_u8, cmap)
    blended = (0.5 * heat + 0.5 * img_rgb).astype('uint8')
    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def run_cityscapes_tam(root: str,
                       output_dir: str = 'cityscapes_vis',
                       model_name: str = 'Qwen/Qwen2-VL-2B-Instruct',
                       split: str = 'val',
                       max_samples: int = -1,
                       prompt_mode: str = 'list',
                       max_new_tokens: int = 64,
                       compute_miou: bool = False):
    """Cityscapes画像に対して各クラス語の Token Activation Map を出力。

    prompt_mode:
      - 'list': クラス列挙→存在する対象を記述する指示
      - 'caption': 通常のキャプション記述
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Qwen2.5-VL 系は Qwen2VLForConditionalGeneration ではなく AutoModelForImageTextToText を利用
    if ('Qwen2.5' in model_name) or ('Qwen2.5-VL' in model_name) or ('Qwen2.5_VL' in model_name) or ('Qwen2.5' in model_name.replace('-', '')) or ('Qwen3' in model_name):
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype="auto", device_map="cpu")
    else:
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        except Exception:
            # フォールバック
            model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    pairs = build_cityscapes_pairs(root, split=split)
    if max_samples > 0:
        pairs = pairs[:max_samples]

    # special_ids（Qwen2-VL既存デモと同一）
    special_ids = {'img_id': [151652, 151653],
                   'prompt_id': [151653, [151645, 198, 151644, 77091]],
                   'answer_id': [[198, 151644, 77091, 198], -1]}

    if len(pairs) == 0:
        print(f"[Cityscapes TAM] No samples found under: {root} (split={split}).")
        print("Tried patterns:")
        print(f" - {os.path.join(root, 'leftImg8bit', split, '<city>', '*_leftImg8bit.png')}")
        print(f" - {os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '<city>', '*_leftImg8bit.png')}")
        print(f" - {os.path.join(root, 'leftImg8bit', split, '<city>', '*.png')}")
        print("Please set --root to the Cityscapes root directory (containing leftImg8bit).")
        return

    # mIoU集計用
    intersections = {k: 0 for k in CITYSCAPES_CATEGORY.keys()}
    unions = {k: 0 for k in CITYSCAPES_CATEGORY.keys()}

    # labelId -> trainId マッピング（Cityscapes公式仕様）
    labelid_to_trainid = {0:255,1:255,2:255,3:255,4:255,5:255,6:255,7:0,8:1,9:255,10:255,11:2,12:3,13:4,14:255,15:255,16:255,17:5,18:255,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,29:255,30:255,31:16,32:17,33:18}

    for i, (img_path, mask_path) in enumerate(pairs):
        img = Image.open(img_path).convert('RGB')
        if prompt_mode == 'list':
            cls_list = ', '.join(CITYSCAPES_CATEGORY.keys())
            prompt = (
                'From the following Cityscapes classes, first output a line starting with "Present:" '
                'followed by only the class names that appear in the image, separated by commas. '
                'Then, in a second line, write one short sentence describing the scene. '
                'Use the class names exactly as written (no synonyms, keep singular form), and respond in English.\n'
                f'Classes: {cls_list}'
            )
        else:
            prompt = 'Describe this urban street scene.'

        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_utils import process_vision_info  # 遅延import
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        generated_ids = outputs.sequences
        logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

        # vision token shape算出
        vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
        vis_inputs = image_inputs

        # 各ラウンドの画像トークンマップ取得（保存なしで高速化）
        img_maps: List[np.ndarray] = []
        raw_maps_records: List[np.ndarray] = []
        for round_idx in range(len(logits)):
            img_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape,
                logits,
                special_ids,
                vis_inputs,
                processor,
                '',  # 保存しない→eval_onlyモード
                round_idx,
                raw_maps_records,
                False
            )
            # 単一画像想定: 2D map
            img_maps.append(img_map)

        # 生成部のみを対象（入力+生成全体シーケンスから入力長を除いた後ろを使う）
        input_len = inputs.input_ids.shape[1]
        trimmed_ids = generated_ids[0][input_len:].cpu().tolist()

        class_maps = aggregate_token_maps(img_maps, trimmed_ids, processor, CITYSCAPES_CATEGORY)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        sample_out_dir = os.path.join(output_dir, base_name)
        os.makedirs(sample_out_dir, exist_ok=True)

        # 生成テキスト・単語グループを保存（デバッグ用）
        try:
            gen_txt = processor.batch_decode([trimmed_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            with open(os.path.join(sample_out_dir, 'generated.txt'), 'w') as f:
                f.write(gen_txt)
            wds, tgs = ids_to_word_groups([trimmed_ids], processor)
            with open(os.path.join(sample_out_dir, 'tokens_debug.txt'), 'w') as f:
                for i, w in enumerate(wds):
                    f.write(f"{i}\t{w}\t{tgs[i]}\n")
        except Exception:
            pass

        for cls_name, act in class_maps.items():
            save_fn = os.path.join(sample_out_dir, f'{cls_name}.png')
            overlay_and_save(img, act, save_fn)

            # mIoU計算（マスクが存在し、compute_miou=True のとき）
            if compute_miou and mask_path is not None and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                # labelIdsの場合 -> trainIdsへ変換
                if 'labelIds' in mask_path:
                    # vectorized mapping
                    mapped = np.full_like(mask, 255)
                    for lid, tid in labelid_to_trainid.items():
                        mapped[mask == lid] = tid
                    mask = mapped
                # trainIdsはそのまま
                gt_train_ids = mask
                cls_id = CITYSCAPES_CATEGORY[cls_name]
                # クラスが存在しない場合 unionは gt + pred で測るので gt クラスピクセル有無を確認
                gt_binary = (gt_train_ids == cls_id).astype(np.uint8)
                if gt_binary.sum() == 0 and act.max() == 0:
                    # 何もない→スキップ（IoU未定義）
                    continue
                # 予測マップ閾値化: Otsu （mapは0-1正規化済み）
                act_resized = cv2.resize(act, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_LINEAR)
                act_u8 = (act_resized * 255).clip(0,255).astype('uint8')
                try:
                    thr, pred_bin = cv2.threshold(act_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except Exception:
                    pred_bin = (act_u8 > 127).astype('uint8') * 255
                pred_binary = (pred_bin > 0).astype(np.uint8)
                inter = int((pred_binary * gt_binary).sum())
                union = int(((pred_binary + gt_binary) > 0).sum())
                if union > 0:
                    intersections[cls_name] += inter
                    unions[cls_name] += union
        if len(class_maps) == 0:
            # 何も一致しなかった場合の手掛かりを出力
            print(f"  (no class words matched in generated text)")

        print(f'[Cityscapes TAM] {i+1}/{len(pairs)} processed: {base_name}')

    if compute_miou:
        per_class_iou = {}
        valid_ious = []
        for k in CITYSCAPES_CATEGORY.keys():
            if unions[k] > 0:
                iou = intersections[k] / unions[k]
                per_class_iou[k] = iou
                valid_ious.append(iou)
            else:
                per_class_iou[k] = None
        miou = sum(valid_ious) / len(valid_ious) if len(valid_ious) > 0 else 0.0
        print('[mIoU Summary] model=%s split=%s samples=%d mIoU=%.4f (valid classes=%d)' % (
            model_name, split, len(pairs) if max_samples<0 else min(max_samples,len(pairs)), miou, len(valid_ious)))
        # 簡易保存
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'miou_{os.path.basename(model_name.replace("/","_"))}.txt'), 'w') as f:
            f.write('model\t%s\n' % model_name)
            f.write('mIoU\t%.6f\n' % miou)
            for k, v in per_class_iou.items():
                f.write('%s\t%s\n' % (k, 'NA' if v is None else ('%.6f' % v)))
        return {'model': model_name, 'miou': miou, 'per_class': per_class_iou}
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cityscapes Token Activation Map (Qwen2-VL)')
    parser.add_argument('--root', type=str, required=True, help='Cityscapes root directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='cityscapes_vis')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--max_samples', type=int, default=-1, help='Limit number of samples (-1 for all)')
    parser.add_argument('--prompt_mode', type=str, default='list', choices=['list', 'caption'])
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--compute_miou', action='store_true')
    args = parser.parse_args()
    run_cityscapes_tam(args.root, args.output_dir, args.model, args.split, args.max_samples, args.prompt_mode, args.max_new_tokens, args.compute_miou)
