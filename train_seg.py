import os
import math
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cityscapes_seg import CityscapesSegmentation, collate_fn, CITYSCAPES_CLASSES
from seg_model import SegmentationModel
from metrics import compute_confusion, compute_miou


def create_dataloaders(root, val_root, val_split, tam_dir_train, tam_dir_val, image_size, batch_size, num_workers, tam_exclude_names, use_tam: bool):
    train_ds = CityscapesSegmentation(root=root, split='train', tam_dir=tam_dir_train, image_size=image_size, use_tam=use_tam, augment=True, tam_exclude=tam_exclude_names)
    val_ds = CityscapesSegmentation(root=val_root, split=val_split, tam_dir=tam_dir_val, image_size=image_size, use_tam=use_tam, augment=False, tam_exclude=tam_exclude_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader, train_ds


def evaluate(model, loader, device, num_classes):
    model.eval()
    intersections = torch.zeros(num_classes, dtype=torch.float64, device=device)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            tams = batch['tam']
            if tams is not None:
                tams = tams.to(device)
            logits = model.predict(imgs, tams, out_size=imgs.shape[2:])  # (B,C,H,W)
            preds = torch.argmax(logits, dim=1)
            inter, uni = compute_confusion(preds, labels, num_classes)
            intersections += inter
            unions += uni
    miou, per_cls = compute_miou(intersections, unions)
    return miou, intersections.cpu(), unions.cpu()


def main():
    parser = argparse.ArgumentParser(description='Train segmentation with frozen DINOv3 + LinearHead (optional TAM)')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--val_root', type=str, default='', help='Root directory for validation dataset (defaults to --root)')
    parser.add_argument('--val_split', type=str, default='val', help='Split name for validation (default: val)')
    parser.add_argument('--tam_dir', type=str, default='', help='Directory of TAM maps for training split')
    parser.add_argument('--tam_dir_val', type=str, default='', help='Directory of TAM maps for validation split (defaults to tam_dir if not set)')
    parser.add_argument('--baseline_no_tam', action='store_true', help='Train baseline without TAM maps (DINOv3 + LinearHead only)')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, nargs=2, default=[1024,2048])
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='runs_seg')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--tam_exclude', type=str, default='road,sky', help='Comma-separated class names to exclude from TAM features (e.g., "road,wall")')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(CITYSCAPES_CLASSES)

    tam_exclude_names = [s.strip() for s in args.tam_exclude.split(',') if s.strip()]
    use_tam = not args.baseline_no_tam
    # Validate TAM directories only if TAM is enabled
    if use_tam:
        if not args.tam_dir:
            raise ValueError('TAM is enabled but --tam_dir is not provided')
        tam_dir_val = args.tam_dir_val if args.tam_dir_val else args.tam_dir.replace('train', 'val')
    else:
        tam_dir_val = ''
    val_root = args.val_root if args.val_root else args.root
    train_loader, val_loader, train_ds = create_dataloaders(args.root, val_root, args.val_split, args.tam_dir if use_tam else None, tam_dir_val if use_tam else None, tuple(args.image_size), args.batch_size, args.num_workers, tam_exclude_names, use_tam)
    if use_tam:
        tam_channels = 19 - len([n for n in tam_exclude_names if n in CITYSCAPES_CLASSES])
    else:
        tam_channels = 0
    model = SegmentationModel(dino_model_name=args.dino_model, num_classes=num_classes, use_tam=use_tam, tam_channels=tam_channels)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * math.ceil(len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['sched'])
        start_epoch = ckpt.get('epoch', 0)
        best_miou = ckpt.get('best_miou', 0.0)
        print(f'Resumed from {args.resume} epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            tams = batch['tam']
            if tams is not None:
                tams = tams.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs, tams)
                # upsample to image size
                logits_up = torch.nn.functional.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                loss = F.cross_entropy(logits_up, labels, ignore_index=255)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        dt = time.time() - t0
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f'Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} time={dt:.1f}s lr={scheduler.get_last_lr()[0]:.6f}')

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'sched': scheduler.state_dict(), 'epoch': epoch+1, 'best_miou': best_miou}, ckpt_path)
        print(f'  Saved checkpoint to {ckpt_path}')

        if (epoch + 1) % args.val_interval == 0:
            miou, inters, unis = evaluate(model, val_loader, device, num_classes)
            print(f'  Validation mIoU={miou:.4f}')
            if miou > best_miou:
                best_miou = miou
                ckpt_path = os.path.join(args.save_dir, 'best.pt')
                torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'sched': scheduler.state_dict(), 'epoch': epoch+1, 'best_miou': best_miou}, ckpt_path)
                print(f'  Saved best checkpoint to {ckpt_path}')

    print(f'Training finished. Best mIoU={best_miou:.4f}')


if __name__ == '__main__':
    main()