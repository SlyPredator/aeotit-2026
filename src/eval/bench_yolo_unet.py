import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A

# This script is to compare YOLO-seg vs UNet models for segmentation.
# Only for 1 pair of image + ground truth mask combination.

# config variables
CONFIDENCE = 0.5 # better to ensure similar confidence level across models for explainability
IMAGE_PATH = r'E:\Navneeth\benchmark\Images\Ceramic_018.png'
MASK_PATH = r'E:\Navneeth\benchmark\Images\Ceramic_018.png'


# load models
    # YOLO-seg
yolo_model = YOLO(r"models\yolo_50.pt")

    # UNet
unet_model = smp.Unet(encoder_name="resnet34", classes=1)
unet_model.load_state_dict(torch.load(r"models\unet_50.pth"))
unet_model.eval().to("cuda")

# functions for metrics and main comparison
def get_metrics(mask1, mask2):
    """Calculates Dice and IoU for binary masks."""
    m1 = (mask1 > 0.5).astype(np.float32)
    m2 = (mask2 > 0.5).astype(np.float32)
    
    intersection = (m1 * m2).sum()
    total = m1.sum() + m2.sum()
    union = total - intersection
    
    dice = (2. * intersection) / (total + 1e-8)
    iou = intersection / (union + 1e-8)
    
    return dice, iou

def compare_models(img_path, ground_truth_mask_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(ground_truth_mask_path, 0)
    _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    gt_mask = gt_mask / 255.0

    # YOLO prediction
    yolo_results = yolo_model(img, conf=CONFIDENCE)[0] 
    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    
    yolo_conf_list = []
    if yolo_results.masks is not None:
        yolo_conf_list = yolo_results.boxes.conf.cpu().numpy() # Get conf per box
        for seg in yolo_results.masks.xy:
            poly = np.array(seg, dtype=np.int32)
            cv2.fillPoly(yolo_mask, [poly], 1)

    avg_yolo_conf = np.mean(yolo_conf_list) if len(yolo_conf_list) > 0 else 0

    # U-Net prediction
    transform = A.Compose([A.Resize(640, 640), A.Normalize(), ToTensorV2()])
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to("cuda")

    with torch.no_grad():
        unet_output = torch.sigmoid(unet_model(input_tensor))
        unet_probs_small = unet_output.cpu().numpy().squeeze()
        unet_probs = cv2.resize(unet_probs_small, (w, h))
        
        unet_mask = (unet_probs > CONFIDENCE).astype(np.float32)
        
        if unet_mask.sum() > 0:
            avg_unet_conf = unet_probs[unet_mask > 0].mean()
        else:
            avg_unet_conf = 0

    # calculate scores
    y_dice, y_iou = get_metrics(yolo_mask, gt_mask)
    u_dice, u_iou = get_metrics(unet_mask, gt_mask)
    
    print("-" * 30)
    print(f"YOLO  | Conf: {avg_yolo_conf:.2f} | Dice: {y_dice:.4f}")
    print(f"U-Net | Conf: {avg_unet_conf:.2f} | Dice: {u_dice:.4f}")
    print("-" * 30)

    # Visualization
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1); plt.imshow(img_rgb); plt.title("Original Image")
    plt.subplot(1, 4, 2); plt.imshow(gt_mask, cmap='gray'); plt.title("Ground Truth")
    
    plt.subplot(1, 4, 3)
    plt.imshow(yolo_mask, cmap='gray')
    plt.title(f"YOLO\nDice: {y_dice:.2f} | IoU: {y_iou:.2f}\nAvg Conf: {avg_yolo_conf:.2%}")
    
    plt.subplot(1, 4, 4)
    plt.imshow(unet_mask, cmap='gray')
    plt.title(f"U-Net\nDice: {u_dice:.2f} | IoU: {u_iou:.2f}\nAvg Conf: {avg_unet_conf:.2%}")
    
    plt.tight_layout()
    plt.show()

compare_models(IMAGE_PATH, MASK_PATH)