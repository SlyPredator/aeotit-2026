import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def create_masks_from_yolo(dataset_path):
    """
    We need masks for UNet training so
    converts YOLO segmentation .txt files to binary .png masks.
    This script expects structure: dataset/train/images and dataset/train/labels
    """
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        img_dir = Path(dataset_path) / split / 'images'
        lbl_dir = Path(dataset_path) / split / 'labels'
        mask_out_dir = Path(dataset_path) / split / 'masks'
        
        if not lbl_dir.exists():
            continue
            
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {split} split...")
        for img_file in tqdm(list(img_dir.glob('*'))):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            img = cv2.imread(str(img_file))
            h, w, _ = img.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            
            label_file = lbl_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) < 4: continue
                        class_id = int(parts[0])
                        coords = np.array(parts[1:]).reshape(-1, 2)
                        
                        coords[:, 0] *= w
                        coords[:, 1] *= h

                        cv2.fillPoly(mask, [coords.astype(np.int32)], 255)

            cv2.imwrite(str(mask_out_dir / f"{img_file.stem}.png"), mask)

create_masks_from_yolo(r"path/to/dataset")