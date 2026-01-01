"""Precompute TAM activation maps for all Cityscapes images.

This script reuses logic from TAM/cityscapes_tam.py but focuses on producing
numeric activation maps suitable for training (npy arrays) instead of PNG overlays.

Output: <out_dir>/<image_basename>.npy  with shape (19, H_t, W_t) where H_t,W_t are vision token grid.
Missing classes are filled with zeros.
"""
import os
import json
import re
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForImageTextToText
from typing import Dict, List, Tuple
from TAM.cityscapes_tam import build_cityscapes_pairs, CITYSCAPES_CATEGORY, aggregate_token_maps
from TAM.qwen_utils import process_vision_info


def load_model_and_processor(model_name: str):
    if ('Qwen2.5' in model_name) or ('Qwen3' in model_name):
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype='auto', device_map='cpu')
    else:
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def aggregate_line_maps(img_maps: List[np.ndarray], tokens: List[int], processor, target_classes: Dict[str, int]) -> Dict[str, np.ndarray]:
    out = {}
    line_ranges = []
    start_idx = 0
    last_newline_count = 0
    
    # Identify line boundaries based on newline characters in decoded text
    for i in range(len(tokens)):
        text = processor.decode(tokens[:i+1], skip_special_tokens=True)
        newline_count = text.count('\n')
        if newline_count > last_newline_count:
            line_ranges.append((start_idx, i + 1))
            start_idx = i + 1
            last_newline_count = newline_count
    if start_idx < len(tokens):
        line_ranges.append((start_idx, len(tokens)))

    for start, end in line_ranges:
        if start >= end:
            continue
        line_tokens = tokens[start:end]
        full_line_text = processor.decode(line_tokens, skip_special_tokens=True)
        line_text_stripped = full_line_text.strip()
        
        # Parse format: "- class_name: reason"
        match = re.match(r'^[-*]\s*([^:]+):(.*)', line_text_stripped)
        if match:
            raw_class_part = match.group(1)
            extracted_class_str = raw_class_part.strip()
            extracted_class_key = extracted_class_str.lower()
            reason_text = match.group(2).strip().lower()

            # Skip negative statements if model hallucinates absent classes
            if reason_text.startswith('no ') or 'not visible' in reason_text or 'not present' in reason_text:
                continue

            matched_cls = None
            if extracted_class_key in target_classes:
                matched_cls = extracted_class_key
            
            if matched_cls:
                # Calculate character range of the class name in the full decoded text
                part_start, part_end = match.span(1)
                sub_start = raw_class_part.find(extracted_class_str)
                
                stripped_start_idx = full_line_text.find(line_text_stripped)
                if stripped_start_idx == -1: stripped_start_idx = 0
                
                start_char = stripped_start_idx + part_start + sub_start
                end_char = start_char + len(extracted_class_str)
                
                target_token_indices = []
                
                # Map tokens to character ranges
                for t_idx in range(len(line_tokens)):
                    prev_text = processor.decode(line_tokens[:t_idx], skip_special_tokens=True)
                    curr_text = processor.decode(line_tokens[:t_idx+1], skip_special_tokens=True)
                    
                    token_start = len(prev_text)
                    token_end = len(curr_text)
                    
                    # Check overlap
                    if max(start_char, token_start) < min(end_char, token_end):
                        target_token_indices.append(t_idx)
                
                if target_token_indices:
                    segment_maps = [img_maps[start + t_idx] for t_idx in target_token_indices]
                    if segment_maps:
                        # Average heatmap for tokens corresponding to the class name
                        avg_map = np.mean(np.stack(segment_maps), axis=0)
                        out[matched_cls] = avg_map
    return out


def compute_tam_for_image(img_path: str,
                          model,
                          processor,
                          prompt: str,
                          max_new_tokens: int = 64) -> Tuple[Dict[str, np.ndarray], Tuple[int, int], str]:
    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True, output_hidden_states=True, return_dict_in_generate=True
    )
    generated_ids = outputs.sequences
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
    # vision token grid (following original code heuristics)
    vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
    # Build activation maps per generation round using TAM function (import locally)
    from TAM.tam import TAM as TAM_FN
    special_ids = {'img_id': [151652, 151653], 'prompt_id': [151653, [151645, 198, 151644, 77091]], 'answer_id': [[198, 151644, 77091, 198], -1]}
    vis_inputs = image_inputs
    img_maps: List[np.ndarray] = []
    raw_records: List[np.ndarray] = []
    for round_idx in range(len(logits)):
        img_map = TAM_FN(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            '',
            round_idx,
            raw_records,
            False
        )
        img_maps.append(img_map)
    input_len = inputs.input_ids.shape[1]
    trimmed_ids = generated_ids[0][input_len:].cpu().tolist()
    generated_text = processor.decode(trimmed_ids, skip_special_tokens=True)
    class_maps = aggregate_line_maps(img_maps, trimmed_ids, processor, CITYSCAPES_CATEGORY)
    return class_maps, vision_shape, generated_text


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Precompute TAM maps for Cityscapes')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'test_extra'])
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--overlay_dir', type=str, default='', help='If set, save class heatmap overlays as PNG under this directory')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--prompt_mode', type=str, default='list', choices=['list','caption'])
    parser.add_argument('--max_samples', type=int, default=-1)
    # Sharding options to enable multi-process parallelism across GPUs
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards (parallel workers)')
    parser.add_argument('--shard_id', type=int, default=0, help='Zero-based shard id for this process')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = build_cityscapes_pairs(args.root, split=args.split)
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    if len(pairs) == 0:
        print('No images found.')
        return
    # Validate sharding parameters
    if args.num_shards < 1:
        raise ValueError('--num_shards must be >= 1')
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError('--shard_id must be in [0, num_shards)')

    # Deterministic sharding: assign items by index modulo
    all_count = len(pairs)
    pairs = [item for idx, item in enumerate(pairs) if (idx % args.num_shards) == args.shard_id]
    print(f'Sharding: shard {args.shard_id}/{args.num_shards} -> {len(pairs)} items (total {all_count})')

    model, processor = load_model_and_processor(args.model)

    if args.prompt_mode == 'list':
        prompt = (
            "Identify which of the following classes are visually present in the image. For each present class, provide a brief reason based on visual evidence. Use only the exact singular class names from the list. Do not invent classes.\n"
            "STRICTLY output ONLY the classes that are visible. Do NOT list classes that are absent.\n"
            "Classes\n"
            "## road\n"
            "## sidewalk\n"
            "## building\n"
            "## wall\n"
            "## fence\n"
            "## pole\n"
            "## traffic light\n"
            "## traffic sign\n"
            "## vegetation\n"
            "## terrain\n"
            "## sky\n"
            "## person\n"
            "## rider\n"
            "## car\n"
            "## truck\n"
            "## bus\n"
            "## train\n"
            "## motorcycle\n"
            "## bicycle\n"
            "Output format:\n"
            "- present_class_name: reason\n"
            "- present_class_name: reason\n"
            "..."
        )
    else:
        prompt = 'Describe this urban street scene.'

    # Helper: overlay and save activation on original image
    def overlay_and_save(raw_img: Image.Image, act_map: np.ndarray, save_path: str, cmap=cv2.COLORMAP_JET):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_rgb = np.array(raw_img.convert('RGB'))
        h, w, _ = img_rgb.shape
        act = cv2.resize(act_map.astype('float32'), (w, h), interpolation=cv2.INTER_LINEAR)
        act_u8 = (act * 255).clip(0, 255).astype('uint8')
        heat = cv2.applyColorMap(act_u8, cmap)
        blended = (0.5 * heat + 0.5 * img_rgb).astype('uint8')
        cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # Local index for progress within the shard
    for i, (img_path, _) in enumerate(pairs):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out_dir, base + '.npy')
        json_path = os.path.join(args.out_dir, base + '.json')
        if os.path.exists(out_path) and os.path.exists(json_path):
            continue
        class_maps, vision_shape, generated_text = compute_tam_for_image(img_path, model, processor, prompt, args.max_new_tokens)
        
        # Save text output
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'image_path': img_path, 'generated_text': generated_text}, f, indent=2, ensure_ascii=False)

        # assemble array
        h_t, w_t = vision_shape
        arr = np.zeros((len(CITYSCAPES_CATEGORY), h_t, w_t), dtype='float32')
        def _robust_normalize(act_map: np.ndarray) -> np.ndarray:
            # Min-max normalize
            m = act_map.astype('float32')
            m = m - m.min()
            maxv = m.max()
            if maxv > 0:
                m = m / maxv
            # Suppress near-uniform low noise by percentile thresholding
            # This mitigates cases like 'road' showing weak global activation.
            thr = np.percentile(m, 85.0)
            m = m - thr
            m[m < 0] = 0
            # Re-normalize to [0,1]
            maxv2 = m.max()
            if maxv2 > 0:
                m = m / maxv2
            return m

        for cls_name, act in class_maps.items():
            cid = CITYSCAPES_CATEGORY[cls_name]
            arr[cid] = _robust_normalize(act)
        np.save(out_path, arr)
        # Optional overlays
        if args.overlay_dir:
            img = Image.open(img_path).convert('RGB')
            sample_dir = os.path.join(args.overlay_dir, base)
            for cls_name, act in class_maps.items():
                vis_path = os.path.join(sample_dir, f'{cls_name}.png')
                overlay_and_save(img, _robust_normalize(act), vis_path)
        print(f'[shard {args.shard_id}] [{i+1}/{len(pairs)}] saved {out_path}' + ('' if not args.overlay_dir else f' and overlays to {os.path.join(args.overlay_dir, base)}'))


if __name__ == '__main__':
    main()
