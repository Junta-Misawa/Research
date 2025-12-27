import os
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Cityscapes 19 classes (trainId order)
CITYSCAPES_CATEGORY = {
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

# Reverse category mapping to get name from index
ID_TO_CATEGORY = {v: k for k, v in CITYSCAPES_CATEGORY.items()}

def find_image_file(basename, dataset_root):
    """Find the original image file for a given basename in the dataset root."""
    # Common patterns for Cityscapes and related datasets
    patterns = [
        f"**/{basename}.png",
        f"**/{basename}.jpg",
        f"**/{basename}.jpeg"
    ]
    
    candidates = []
    for pattern in patterns:
        # Use recursive glob
        matches = glob.glob(os.path.join(dataset_root, pattern), recursive=True)
        candidates.extend(matches)
    
    if not candidates:
        return None

    # Prioritize leftImg8bit
    for c in candidates:
        if 'leftImg8bit' in c:
            return c
            
    return candidates[0]

def overlay_and_save(raw_img_path, act_map, save_path, class_name, cmap=cv2.COLORMAP_JET):
    """Overlay activation map on image and save."""
    try:
        img = Image.open(raw_img_path).convert('RGB')
        img_rgb = np.array(img)
        h, w, _ = img_rgb.shape
        
        # Resize activation map to image size
        act = cv2.resize(act_map.astype('float32'), (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize for visualization if not already 0-1 (npy might be 0-1 or raw)
        # precompute_tam.py saves normalized [0,1] but let's be safe
        if act.max() > 1.0:
             act = act / act.max()
        
        act_u8 = (act * 255).clip(0, 255).astype('uint8')
        heat = cv2.applyColorMap(act_u8, cmap)
        
        # Blend
        blended = (0.5 * heat + 0.5 * img_rgb).astype('uint8')
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        print(f"Error processing {raw_img_path}: {e}")

def main():
    tam_maps_root = "TAM_maps"
    dataset_root = "Dataset"
    output_root = "TAM_visualizations"
    
    if not os.path.exists(tam_maps_root):
        print(f"TAM_maps directory not found at {tam_maps_root}")
        return

    # Walk through TAM_maps directory
    npy_files = []
    for root, dirs, files in os.walk(tam_maps_root):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    
    print(f"Found {len(npy_files)} npy files.")
    
    for npy_path in tqdm(npy_files):
        # Load npy
        try:
            tam_data = np.load(npy_path)
        except Exception as e:
            print(f"Failed to load {npy_path}: {e}")
            continue
            
        # Check shape
        if tam_data.ndim != 3 or tam_data.shape[0] != 19:
            print(f"Unexpected shape {tam_data.shape} for {npy_path}, skipping.")
            continue
            
        # Find original image
        basename = os.path.splitext(os.path.basename(npy_path))[0]
        image_path = find_image_file(basename, dataset_root)
        
        if not image_path:
            # Try removing _leftImg8bit suffix if present, sometimes npy might be named differently?
            # But precompute_tam.py uses basename of image, so it should match.
            # Let's try searching just by unique ID if possible, but basename is safest.
            print(f"Original image not found for {basename}")
            continue
            
        # Determine output directory structure
        # Mirror the structure of TAM_maps in output_root
        rel_path = os.path.relpath(os.path.dirname(npy_path), tam_maps_root)
        save_dir_base = os.path.join(output_root, rel_path, basename)
        
        # Iterate over classes
        for class_idx in range(19):
            act_map = tam_data[class_idx]
            
            # Skip if map is all zeros or very low activation
            if act_map.max() < 0.01:
                continue
                
            class_name = ID_TO_CATEGORY.get(class_idx, f"class_{class_idx}")
            save_path = os.path.join(save_dir_base, f"{class_name}.png")
            
            overlay_and_save(image_path, act_map, save_path, class_name)

if __name__ == "__main__":
    main()
