import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import csv

# config variables
DATA_PATH = r"E:\Navneeth\Crack-Segmentation-yolo" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640  
BATCH_SIZE = 8
EPOCHS = 50
LOG_FILE = "unet_training_metrics.csv"
MODEL_SAVE_PATH = "best_unet.pth"

# load and transform dataset
class CrackDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.img_dir = Path(root_dir) / split / 'images'
        self.mask_dir = Path(root_dir) / split / 'masks'
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(str(self.img_dir / img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.mask_dir / f"{Path(img_name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32) 

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.05, scale=0.05, rotate=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# load model
model = smp.Unet(
    encoder_name="resnet50",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=1                      
).to(DEVICE)

# BCE + Dice loss
dice_loss_fn = smp.losses.DiceLoss(mode='binary')
bce_loss_fn = nn.BCEWithLogitsLoss()

def criterion(preds, targets):
    return bce_loss_fn(preds, targets) + dice_loss_fn(preds, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Scheduler to reduce LR in case it stagnates for `patience` number of epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

train_ds = CrackDataset(DATA_PATH, split='train', transform=train_transform)
val_ds = CrackDataset(DATA_PATH, split='valid', transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# logging metrics
headers = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "lr"]
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# train model
print(f"Starting Training on {DEVICE}...")
best_dice = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    model.eval()
    val_loss, val_dice, val_iou, val_prec, val_rec = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).unsqueeze(1)
            logits = model(images)
            
            v_loss = criterion(logits, masks)
            val_loss += v_loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            tp = (preds * masks).sum()
            fp = (preds * (1 - masks)).sum()
            fn = ((1 - preds) * masks).sum()
            
            val_dice += (2 * tp) / (2 * tp + fp + fn + 1e-8)
            val_iou += tp / (tp + fp + fn + 1e-8)
            val_prec += tp / (tp + fp + 1e-8)
            val_rec += tp / (tp + fn + 1e-8)

    num_val_batches = len(val_loader)
    avg_val_dice = (val_dice / num_val_batches).item()
    current_lr = optimizer.param_groups[0]['lr']

    epoch_results = [
        epoch + 1,
        train_loss / len(train_loader),
        val_loss / num_val_batches,
        avg_val_dice,
        (val_iou / num_val_batches).item(),
        (val_prec / num_val_batches).item(),
        (val_rec / num_val_batches).item(),
        current_lr
    ]

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(epoch_results)

    scheduler.step(avg_val_dice)

    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"--> Best Model Saved (Dice: {best_dice:.4f})")

    print(f"Summary: Loss {epoch_results[2]:.4f} | Dice {avg_val_dice:.4f} | LR {current_lr}")

print(f"Done! All metrics logged to {LOG_FILE}")