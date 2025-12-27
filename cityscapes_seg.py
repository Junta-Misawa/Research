import os
import glob
from typing import List, Tuple, Optional, Dict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Cityscapes 19 classes (trainId order) used for segmentation
CITYSCAPES_CLASSES: Dict[str, int] = {
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

# labelId -> trainId mapping (255 means ignored)
LABELID_TO_TRAINID = {
    0:255,1:255,2:255,3:255,4:255,5:255,6:255,7:0,8:1,9:255,10:255,11:2,12:3,13:4,14:255,15:255,16:255,17:5,18:255,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,29:255,30:255,31:16,32:17,33:18
}


def _build_image_list(root: str, split: str) -> List[str]:
    patterns = [
        # 標準の2階層（city/filename）
        os.path.join(root, 'leftImg8bit', split, '*', '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit', split, '*', '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*', '*_leftImg8bit.jpeg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*', '*_leftImg8bit.jpeg'),
        # 1階層のみ（FDなどでcityディレクトリが無いケース）
        os.path.join(root, 'leftImg8bit', split, '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit', split, '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*_leftImg8bit.jpeg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*_leftImg8bit.png'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*_leftImg8bit.jpg'),
        os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split, '*_leftImg8bit.jpeg'),
        # フォールバック: 2階層/1階層で任意のPNG/JPG
        os.path.join(root, 'leftImg8bit', split, '*', '*.png'),
        os.path.join(root, 'leftImg8bit', split, '*', '*.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*', '*.jpeg'),
        os.path.join(root, 'leftImg8bit', split, '*.png'),
        os.path.join(root, 'leftImg8bit', split, '*.jpg'),
        os.path.join(root, 'leftImg8bit', split, '*.jpeg'),
    ]
    files = []
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            if p not in seen:
                seen.add(p)
                files.append(p)
    return sorted(files)


def infer_label_path(img_path: str) -> Optional[str]:
    # Determine the suffix to remove to get the stem
    suffixes = ['_leftImg8bit.png', '_leftImg8bit.jpg', '_leftImg8bit.jpeg', '.png', '.jpg', '.jpeg']
    suffix = ''
    for s in suffixes:
        if img_path.endswith(s):
            suffix = s
            break
            
    fname = os.path.basename(img_path)
    stem = fname[:-len(suffix)] if suffix else fname

    # 1) 構造的マッピング: <root>/leftImg8bit/<split>[/<city>]/<stem>_leftImg8bit.png
    #                       -> <root>/gtFine/<split>[/<city>]/<stem>_gtFine_labelTrainIds.png (優先)
    #                       -> <root>/gtCoarse/<split>[/<city>]/<stem>_gtCoarse_labelTrainIds.png
    try:
        parts = img_path.split('/')
        if 'leftImg8bit' in parts:
            li = parts.index('leftImg8bit')
            base_dir = '/'.join(parts[:li])
            # split は直後、city はその次（存在すれば）
            split_name = parts[li+1] if len(parts) > li+1 else ''
            
            # Check if the next part is the filename or a city directory
            if len(parts) > li+2:
                if parts[li+2] == fname:
                    maybe_city = ''
                else:
                    maybe_city = parts[li+2]
            else:
                maybe_city = ''

            subpath = [split_name] if maybe_city == '' else [split_name, maybe_city]
            
            # gtFine candidates
            gf_dir = '/'.join([p for p in [base_dir, 'gtFine'] + subpath if p])
            fine_candidates = [
                os.path.join(gf_dir, stem + '_gtFine_labelTrainIds.png'),
                os.path.join(gf_dir, stem + '_gtFine_trainIds.png'),
                os.path.join(gf_dir, stem + '_gtFine_labelIds.png'),
                # FoggyZurich 等: サフィックス無しのラベルファイル
                os.path.join(gf_dir, stem + '.png'),
                os.path.join(gf_dir, stem + '.jpg'),
                os.path.join(gf_dir, stem + '.jpeg'),
            ]
            for c in fine_candidates:
                if os.path.exists(c):
                    return c
            # gtCoarse candidates
            gc_dir = '/'.join([p for p in [base_dir, 'gtCoarse'] + subpath if p])
            coarse_candidates = [
                os.path.join(gc_dir, stem + '_gtCoarse_labelTrainIds.png'),
                os.path.join(gc_dir, stem + '_gtCoarse_trainIds.png'),
                os.path.join(gc_dir, stem + '_gtCoarse_labelIds.png'),
                os.path.join(gc_dir, stem + '.png'),
                os.path.join(gc_dir, stem + '.jpg'),
                os.path.join(gc_dir, stem + '.jpeg'),
            ]
            for c in coarse_candidates:
                if os.path.exists(c):
                    return c
    except Exception:
        pass

    # NighttimeDrivingTest explicit mapping: leftImg8bit/test/night -> gtCoarse_daytime_trainvaltest/test/night
    if ('/leftImg8bit/test/night/' in img_path) and ('NighttimeDrivingTest' in img_path):
        base_dir = img_path.split('/leftImg8bit/test/night/')[0]
        ndt_dir = os.path.join(base_dir, 'gtCoarse_daytime_trainvaltest', 'test', 'night')
        ndt_candidates = [
            os.path.join(ndt_dir, stem + '_gtCoarse_labelTrainIds.png'),
            os.path.join(ndt_dir, stem + '_gtCoarse_labelIds.png'),
            os.path.join(ndt_dir, stem + '_gtCoarse_trainIds.png'),
            os.path.join(ndt_dir, stem + '.png'),
            os.path.join(ndt_dir, stem + '.jpg'),
            os.path.join(ndt_dir, stem + '.jpeg'),
        ]
        for c in ndt_candidates:
            if os.path.exists(c):
                return c

    # Fallback replacements using stem
    candidates = []
    
    # Define directory replacements
    dir_replacements = [
        ('leftImg8bit', 'gtFine'),
        ('leftImg8bit_trainvaltest/leftImg8bit', 'gtFine_trainvaltest/gtFine'),
        ('leftImg8bit', 'gtCoarse_daytime_trainvaltest/gtCoarse'),
        ('leftImg8bit', 'gtCoarse_daytime_trainvaltest'),
        ('leftImg8bit', 'gtCoarse'),
    ]
    
    # Define label suffixes
    label_suffixes = [
        '_gtFine_labelTrainIds.png', '_gtFine_trainIds.png', '_gtFine_labelIds.png',
        '_gtCoarse_labelTrainIds.png', '_gtCoarse_trainIds.png', '_gtCoarse_labelIds.png',
        '_train_id.png'
    ]

    for dir_from, dir_to in dir_replacements:
        if dir_from in img_path:
            new_path_base = img_path.replace(dir_from, dir_to)
            new_dir = os.path.dirname(new_path_base)
            
            for ls in label_suffixes:
                candidates.append(os.path.join(new_dir, stem + ls))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class CityscapesSegmentation(Dataset):
    """Cityscapes segmentation dataset that optionally loads precomputed TAM maps.

    TAM マップは precompute_tam.py により .npy 形式 (19, H_t, W_t) で保存されている想定。
    ここでは DINOv3 のパッチ解像度 (H/16, W/16) に整形し、特徴チャネルとして利用できるようにする。
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        tam_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 1024),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_tam: bool = True,
        tam_exclude: Optional[List[str]] = None,
        augment: bool = True,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.use_tam = use_tam and tam_dir is not None
        self.tam_dir = tam_dir
        self.files = _build_image_list(root, split)
        if len(self.files) == 0:
            raise RuntimeError(f"No Cityscapes images found under {root} split={split}")

        # transforms
        resize_ops = [T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)]
        self.to_tensor = T.Compose(resize_ops + [T.ToTensor(), T.Normalize(mean, std)])
        self.augment = augment and split == 'train'

        if self.augment:
            self.aug_color = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
            self.hflip = T.RandomHorizontalFlip(p=0.5)
        else:
            self.aug_color = None
            self.hflip = None

        # TAM exclusion settings (names -> indices)
        self.tam_exclude_names = tam_exclude or []
        name_to_id = CITYSCAPES_CLASSES
        self.tam_exclude_ids = sorted([name_to_id[n] for n in self.tam_exclude_names if n in name_to_id])

    def __len__(self):
        return len(self.files)

    def _load_label(self, label_path: Optional[str]) -> torch.Tensor:
        if label_path is None:
            # return ignore map
            h, w = self.image_size
            return torch.full((h, w), 255, dtype=torch.int64)
        arr = np.array(Image.open(label_path))
        # convert labelIds -> trainIds if needed
        if 'labelIds' in label_path:
            mapped = np.full_like(arr, 255)
            for lid, tid in LABELID_TO_TRAINID.items():
                mapped[arr == lid] = tid
            arr = mapped
        # resize with nearest (label)
        lab_img = Image.fromarray(arr)
        lab_img = lab_img.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        return torch.from_numpy(np.array(lab_img, dtype=np.int64))

    def _maybe_hflip(self, img_t: torch.Tensor, label_t: torch.Tensor, tam_t: Optional[torch.Tensor]):
        if self.hflip is None:
            return img_t, label_t, tam_t
        # RandomHorizontalFlip returns PIL/tensor only for PIL Images; emulate manually for tensors.
        if torch.rand(1).item() < 0.5:
            img_t = torch.flip(img_t, dims=[2])
            label_t = torch.flip(label_t, dims=[1])
            if tam_t is not None:
                tam_t = torch.flip(tam_t, dims=[2])
        return img_t, label_t, tam_t

    def _load_tam(self, img_path: str) -> Optional[torch.Tensor]:
        if not self.use_tam:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        npy_path = os.path.join(self.tam_dir, base + '.npy')
        if not os.path.exists(npy_path):
            return None
        arr = np.load(npy_path)  # (19, h_t, w_t)
        # Exclude specified class channels (e.g., road=0)
        if self.tam_exclude_ids:
            keep = [i for i in range(arr.shape[0]) if i not in self.tam_exclude_ids]
            arr = arr[keep]
        # interpolate to patch resolution of current image size
        H, W = self.image_size
        H_patch, W_patch = H // 16, W // 16
        tam = torch.from_numpy(arr).float().unsqueeze(0)  # (1,19,h_t,w_t)
        tam = torch.nn.functional.interpolate(tam, size=(H_patch, W_patch), mode='bilinear', align_corners=False)
        return tam.squeeze(0)  # (19, H_patch, W_patch)

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        label_path = infer_label_path(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.aug_color is not None:
            img = self.aug_color(img)
        img_t = self.to_tensor(img)  # (3,H,W)
        label_t = self._load_label(label_path)  # (H,W)
        tam_t = self._load_tam(img_path)  # (19,H/16,W/16) or None
        img_t, label_t, tam_t = self._maybe_hflip(img_t, label_t, tam_t)
        sample = {
            'image': img_t,
            'label': label_t,
            'tam': tam_t,
            'path': img_path,
        }
        return sample


def collate_fn(batch: List[Dict]):
    imgs = torch.stack([b['image'] for b in batch], dim=0)
    labels = torch.stack([b['label'] for b in batch], dim=0)
    # TAM may be missing → fill zeros
    has_tam = batch[0]['tam'] is not None
    if has_tam:
        C = batch[0]['tam'].shape[0]
        tam_list = [b['tam'] if b['tam'] is not None else torch.zeros(C, imgs.shape[2]//16, imgs.shape[3]//16) for b in batch]
        tams = torch.stack(tam_list, dim=0)
    else:
        tams = None
    paths = [b['path'] for b in batch]
    return {'image': imgs, 'label': labels, 'tam': tams, 'path': paths}
