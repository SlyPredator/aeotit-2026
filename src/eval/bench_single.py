import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# This script is to compare YOLO-seg vs UNet models vs MaskRCNN for segmentation.
# Only for 1 pair of image + ground truth mask combination.

# config variables
CONFIDENCE = 0.5 # better to ensure similar confidence level across models for explainability
IMAGE_PATH = r'E:\Navneeth\benchmark\Images\Ceramic_018.png'
MASK_PATH = r'E:\Navneeth\benchmark\Images\Ceramic_018.png'

# load models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # YOLO-seg
yolo_model = YOLO(r"models\yolo_50.pt")

    # U-Net
unet_model = smp.Unet(encoder_name="resnet34", classes=1)
unet_model.load_state_dict(torch.load(r"models\unet_50.pth"))
unet_model.eval().to(device)

    # Mask R-CNN
def load_maskrcnn(checkpoint_path, num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval().to(device)
    return model

mrcnn_model = load_maskrcnn("best_maskrcnn_crack.pth")

# functions for metrics and main comparison
def get_metrics(mask1, mask2):
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
    yolo_results = yolo_model(img, conf=CONFIDENCE, verbose=False)[0] 
    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    yolo_conf = 0
    if yolo_results.masks is not None:
        yolo_conf = yolo_results.boxes.conf.cpu().numpy().mean()
        for seg in yolo_results.masks.xy:
            poly = np.array(seg, dtype=np.int32)
            cv2.fillPoly(yolo_mask, [poly], 1)

    # UNet prediction
    u_transform = A.Compose([A.Resize(640, 640), A.Normalize(), ToTensorV2()])
    u_input = u_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        u_out = torch.sigmoid(unet_model(u_input))
        u_probs = cv2.resize(u_out.cpu().numpy().squeeze(), (w, h))
        unet_mask = (u_probs > CONFIDENCE).astype(np.float32)
        unet_conf = u_probs[unet_mask > 0].mean() if unet_mask.sum() > 0 else 0

    # MaskRCNN prediction
    m_transform = torchvision.transforms.ToTensor()
    m_input = m_transform(img_rgb).to(device)
    with torch.no_grad():
        m_out = mrcnn_model([m_input])[0]
        mrcnn_mask = np.zeros((h, w), dtype=np.float32)
        mrcnn_confs = []

        for i, score in enumerate(m_out['scores']):
            if score > CONFIDENCE:
                mask = m_out['masks'][i, 0].cpu().numpy()
                mrcnn_mask = np.maximum(mrcnn_mask, mask)
                mrcnn_confs.append(score.item())
        
        mrcnn_binary = (mrcnn_mask > 0.5).astype(np.float32)
        mrcnn_avg_conf = np.mean(mrcnn_confs) if len(mrcnn_confs) > 0 else 0

    # calculate scores
    y_dice, y_iou = get_metrics(yolo_mask, gt_mask)
    u_dice, u_iou = get_metrics(unet_mask, gt_mask)
    m_dice, m_iou = get_metrics(mrcnn_binary, gt_mask)
    
    # Visualization
    results = [
        (img_rgb, "Original"),
        (gt_mask, "Ground Truth"),
        (yolo_mask, f"YOLO\nDice: {y_dice:.2f} | Conf: {yolo_conf:.2%}"),
        (unet_mask, f"U-Net\nDice: {u_dice:.2f} | Conf: {unet_conf:.2%}"),
        (mrcnn_binary, f"Mask R-CNN\nDice: {m_dice:.2f} | Conf: {mrcnn_avg_conf:.2%}")
    ]

    plt.figure(figsize=(20, 5))
    for i, (m, title) in enumerate(results):
        plt.subplot(1, 5, i+1)
        plt.imshow(m, cmap='gray' if i > 0 else None)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

compare_models(IMAGE_PATH, MASK_PATH)