import os
import time
import torch
import torchvision
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# config variables
DATASET_COCO = r'E:\Navneeth\Crack-Segmentation-1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2  # Background + Crack
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_PATIENCE = 5  # Number of epochs to wait before dropping LR
SCHEDULER_FACTOR = 0.1   # Reduce LR by 10x (0.005 -> 0.0005)
LOG_FILE = "maskrcnn_training_metrics.csv"
MODEL_SAVE_PATH = "best_maskrcnn.pth"

# load dataset
class CocoSegmentationDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        
        all_ids = list(sorted(self.coco.imgs.keys()))
        self.ids = []
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.ids.append(img_id)
        
        print(f"Loaded {len(self.ids)} images with valid annotations from {ann_file}")
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes, masks, labels = [], [], []
        for ann in coco_annotation:
            xmin, ymin, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(1)
                mask = self.coco.annToMask(ann)
                masks.append(mask)

        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self.ids))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)

# load model
def get_model_instance_segmentation(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

train_ds = CocoSegmentationDataset(
    root=os.path.join(DATASET_COCO, "train"),
    ann_file=os.path.join(DATASET_COCO, "train/_annotations.coco.json"),
    transforms=torchvision.transforms.ToTensor()
)

loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=lambda x: tuple(zip(*x))
)

# load model
model = get_model_instance_segmentation(NUM_CLASSES).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Scheduler to reduce LR in case it stagnates for `patience` number of epochs
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=SCHEDULER_FACTOR, 
    patience=SCHEDULER_PATIENCE
)

# logging metrics
headers = ["epoch", "total_loss", "mask_loss", "box_loss", "classifier_loss", "lr"]
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# train model
print(f"Training started on {DEVICE}...")
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_total_loss = 0
    epoch_mask_loss = 0
    epoch_box_loss = 0
    epoch_classifier_loss = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, targets in loop:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_total_loss += losses.item()
        epoch_mask_loss += loss_dict['loss_mask'].item()
        epoch_box_loss += loss_dict['loss_box_reg'].item()
        epoch_classifier_loss += loss_dict['loss_classifier'].item()
        
        loop.set_postfix(total_loss=losses.item(), mask_loss=loss_dict['loss_mask'].item())

    avg_total = epoch_total_loss / len(loader)
    avg_mask = epoch_mask_loss / len(loader)
    avg_box = epoch_box_loss / len(loader)
    avg_classifier = epoch_classifier_loss / len(loader)
    lr_scheduler.step(avg_total)
    
    current_lr = optimizer.param_groups[0]['lr']

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_total, avg_mask, avg_box, avg_classifier, current_lr])

    print(f"\n--- Epoch {epoch+1} Results ---")
    print(f"Avg Loss: {avg_total:.4f} | Mask Loss: {avg_mask:.4f} | LR: {current_lr}")
    
    if avg_total < best_loss:
        best_loss = avg_total
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Status: New Best Model Saved (Loss: {best_loss:.4f})")
    print("-" * 30 + "\n")

print(f"Training Complete! Metrics logged to {LOG_FILE}")