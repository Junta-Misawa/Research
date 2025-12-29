import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from cityscapes_seg import CityscapesSegmentation, collate_fn, CITYSCAPES_CLASSES, infer_label_path
from seg_model import SegmentationModel
from train_seg import evaluate


CITYSCAPES_PALETTE = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    70, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]
CITYSCAPES_PALETTE += [0] * (256 * 3 - len(CITYSCAPES_PALETTE))


# Pre-defined dataset configurations for benchmarking
# Adjust paths as necessary for your environment
DATASET_CONFIGS = {
    'NighttimeDrivingTest': {
        'root': 'Dataset/NighttimeDrivingTest',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_nd'
    },
    'FoggyDriving-Dense': {
        'root': 'Dataset/FDD',
        'split': 'test_extra',
        'tam_dir': 'TAM_maps/tam_maps_fdd'
    },
    'FoggyDriving': {
        'root': 'Dataset/FD',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_fd'
    },
    'FoggyZurich': {
        'root': 'Dataset/FZ',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_fz'
    },
    'C-driving-cloudy': {
        'root': 'Dataset/C-driving-cloudy',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_cdc'
    },
    'C-driving-rainy': {
        'root': 'Dataset/C-driving-rainy',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_cdr'
    },
    'C-driving-snowy': {
        'root': 'Dataset/C-driving-snowy',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_cds'
    },
    'C-driving-overcast': {
        'root': 'Dataset/C-driving-overcast',
        'split': 'test',
        'tam_dir': 'TAM_maps/tam_maps_cdo'
    },
}


def save_visualizations(model, loader, device, vis_dir):
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)
    print(f'Saving visualizations to {vis_dir}...')
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            tams = batch['tam']
            paths = batch['path']
            
            if tams is not None:
                tams = tams.to(device)
            
            logits = model.predict(imgs, tams, out_size=imgs.shape[2:])
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(preds)):
                pred = preds[i].astype(np.uint8)
                img_name = os.path.basename(paths[i])
                # Force PNG extension for visualization
                img_name = os.path.splitext(img_name)[0] + '.png'
                save_path = os.path.join(vis_dir, img_name)
                
                # Colorize
                color_img = Image.fromarray(pred).convert('P')
                color_img.putpalette(CITYSCAPES_PALETTE)
                color_img.save(save_path)


def run_evaluation(args, model, device, num_classes, dataset_name, root, split, tam_dir):
    print(f'\n[Info] Starting evaluation on {dataset_name}...')
    print(f'       Root: {root}')
    print(f'       Split: {split}')
    print(f'       TAM Dir: {tam_dir}')

    # Dataset & loader
    use_tam = not args.baseline_no_tam
    # If tam_dir is not provided for this dataset, disable TAM for this run or warn?
    # Here we assume if use_tam is True, tam_dir must be valid or we might fail/skip.
    current_tam_dir = tam_dir if (use_tam and tam_dir) else None
    
    if use_tam and not current_tam_dir:
        print(f'[Warning] TAM is enabled but no TAM dir provided for {dataset_name}. Running without TAM maps.')
        use_tam_for_ds = False
    else:
        use_tam_for_ds = use_tam

    try:
        test_ds = CityscapesSegmentation(
            root=root,
            split=split,
            tam_dir=current_tam_dir,
            image_size=tuple(args.image_size),
            use_tam=use_tam_for_ds,
            augment=False,
            tam_exclude=[s.strip() for s in args.tam_exclude.split(',') if s.strip()]
        )
    except FileNotFoundError as e:
        print(f'[Error] Dataset not found or invalid path: {e}')
        return

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    if args.debug:
        # Inspect a few samples
        print(f'[Debug] {dataset_name}: {len(test_ds)} samples.')

    # Evaluate
    miou, inters, unis = evaluate(model, test_loader, device, num_classes)
    print(f'[{dataset_name}] mIoU={miou:.4f}')
    
    # Per-class IoU
    ious = inters / (unis + 1e-6)
    print(f'[{dataset_name}] Per-class IoU:')
    class_names = list(CITYSCAPES_CLASSES.keys())
    for i, iou in enumerate(ious):
        print(f'  {class_names[i]:<15}: {iou.item():.4f}')

    if args.save_vis:
        # Create a subdirectory for this dataset
        ds_vis_dir = os.path.join(args.vis_dir, dataset_name)
        save_visualizations(model, test_loader, device, ds_vis_dir)


def main():
    parser = argparse.ArgumentParser(description='Evaluate on Cityscapes-like test split (e.g., NighttimeDrivingTest) using best checkpoint with or without TAM maps.')
    parser.add_argument('--root', type=str, default='', help='Dataset root (required unless --benchmark_all is set)')
    parser.add_argument('--tam_dir', type=str, default='', help='Directory of precomputed TAM maps (.npy) for test split')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('runs_seg', 'best.pt'))
    parser.add_argument('--split', type=str, default='test', help='Dataset split name (e.g., test, test_extra)')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--image_size', type=int, nargs=2, default=[1024, 2048])
    parser.add_argument('--tam_exclude', type=str, default='road,sky', help='Comma-separated class names to exclude from TAM features')
    parser.add_argument('--baseline_no_tam', action='store_true', help='Evaluate baseline without TAM features')
    parser.add_argument('--debug', action='store_true', help='Print label statistics and sample checks')
    parser.add_argument('--save_vis', action='store_true', help='Save segmentation visualizations')
    parser.add_argument('--vis_dir', type=str, default='vis_results', help='Directory to save visualizations')
    parser.add_argument('--benchmark_all', action='store_true', help='Run evaluation on all pre-defined datasets (ignores --root, --split, --tam_dir)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(CITYSCAPES_CLASSES)

    # Model
    tam_exclude_names = [s.strip() for s in args.tam_exclude.split(',') if s.strip()]
    use_tam = not args.baseline_no_tam
    tam_channels = 0 if not use_tam else (19 - len([n for n in tam_exclude_names if n in CITYSCAPES_CLASSES]))
    model = SegmentationModel(dino_model_name=args.dino_model, num_classes=num_classes, use_tam=use_tam, tam_channels=tam_channels)
    model = model.to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print(f'Loaded weights from {args.checkpoint}')

    if args.benchmark_all:
        print("=== Running Benchmark on All Datasets ===")
        for ds_name, config in DATASET_CONFIGS.items():
            run_evaluation(
                args, 
                model, 
                device, 
                num_classes, 
                dataset_name=ds_name, 
                root=config['root'], 
                split=config['split'], 
                tam_dir=config['tam_dir']
            )
    else:
        if not args.root:
            parser.error("--root is required unless --benchmark_all is set.")
        
        run_evaluation(
            args, 
            model, 
            device, 
            num_classes, 
            dataset_name='CustomDataset', 
            root=args.root, 
            split=args.split, 
            tam_dir=args.tam_dir
        )


if __name__ == '__main__':
    main()
