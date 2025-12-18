import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import csv

# This script is to compare YOLO-seg vs UNet models vs MaskRCNN for segmentation.
# For separate folders of images and masks
# Logs all the relevant metrics to a csv file and saves comparison images across models

# config variables
IMAGE_DIR = r'E:\Navneeth\benchmark\testi'
MASK_DIR = r'E:\Navneeth\benchmark\testm'
BASE_OUTPUT = r'E:\Navneeth\Comparison_Results'
OUTPUT_VIS_DIR = os.path.join(BASE_OUTPUT, "Individual_Results")
SUMMARY_DIR = os.path.join(BASE_OUTPUT, "Summary")
CSV_OUTPUT_PATH = os.path.join(BASE_OUTPUT, 'model_comparison_metrics.csv')
LIMIT = 20
CONFIDENCE = 0.25 

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load models
    # YOLO-seg
yolo_model = YOLO(r"models\yolo_50.pt")

    # UNet
unet_model = smp.Unet(encoder_name="resnet50", classes=1)
unet_model.load_state_dict(torch.load(r"E:\Navneeth\best_unet_crack.pth", map_location=device))
unet_model.eval().to(device)

    # Mask RCNN
def load_maskrcnn(checkpoint_path, num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, box_score_thresh=CONFIDENCE)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)
    return model

mrcnn_model = load_maskrcnn(r"models\mrcnn_50.pth")

# functions for metrics and main comparison
def get_pixel_counts(pred, gt):
    p = (pred > 0.5).astype(np.uint8).flatten()
    g = (gt > 0.5).astype(np.uint8).flatten()
    
    # image length checks
    if p.shape != g.shape:
        from scipy.ndimage import zoom
        pass 

    # For confusion matrix
    tp = np.sum((p == 1) & (g == 1))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    
    # Dice and IoU
    dice = (2. * tp) / (2. * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    
    return dice, iou, [tp, fp, fn, tn]

def plot_confusion_matrix(counts, title, ax):
    """
    Plots a normalized confusion matrix.
    """
    # Layout: [[TN, FP], [FN, TP]]
    cm = np.array([[counts[3], counts[1]], [counts[2], counts[0]]]) 
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Background', 'Crack'], yticklabels=['Background', 'Crack'])
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

def process_and_compare():
    headers = ["image_number", "image_name", "dice_yolo", "dice_unet", "dice_mrcnn",
               "iou_yolo", "iou_unet", "iou_mrcnn", "conf_yolo", "conf_unet", "conf_mrcnn"]
    all_scores = []
    total_counts = {"YOLO": [0,0,0,0], "U-Net": [0,0,0,0], "M-RCNN": [0,0,0,0]}

    with open(CSV_OUTPUT_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_files = all_files[:LIMIT] if LIMIT is not None else all_files
        
        for idx, img_name in enumerate(image_files, 1):
            img_path = os.path.join(IMAGE_DIR, img_name)
            gt_path = os.path.join(MASK_DIR, img_name)
            if not os.path.exists(gt_path): continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gt_mask = cv2.imread(gt_path, 0) / 255.0

            # YOLO prediction
            y_res = yolo_model(img, conf=CONFIDENCE, verbose=False)[0]
            y_mask = np.zeros((h, w), dtype=np.uint8)
            y_conf = 0
            if y_res.masks is not None:
                y_conf = y_res.boxes.conf.cpu().numpy().mean()
                for seg in y_res.masks.xy:
                    cv2.fillPoly(y_mask, [np.array(seg, dtype=np.int32)], 1)

            # U-Net prediction
            u_tf = A.Compose([A.Resize(640, 640), A.Normalize(), ToTensorV2()])
            u_in = u_tf(image=img_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                u_out = torch.sigmoid(unet_model(u_in))
                u_p = cv2.resize(u_out.cpu().numpy().squeeze(), (w, h), interpolation=cv2.INTER_LINEAR)
                u_mask = (u_p > 0.5).astype(np.float32)
                u_conf = u_p[u_mask > 0].mean() if u_mask.sum() > 0 else 0

            # MaskRCNN prediction
            m_in = torchvision.transforms.ToTensor()(img_rgb).to(device)
            with torch.no_grad():
                m_out = mrcnn_model([m_in])[0]
                m_acc = np.zeros((h, w), dtype=np.float32)
                m_scs = m_out['scores'].cpu().numpy()
                m_masks = m_out['masks'].cpu().numpy()
                
                valid_indices = np.where(m_scs > CONFIDENCE)[0]
                if len(valid_indices) > 0:
                    for i in valid_indices:
                        m_acc = np.maximum(m_acc, m_masks[i, 0])
                    m_mask = (m_acc > 0.5).astype(np.float32)
                    m_conf = m_scs[valid_indices].mean()
                else:
                    m_mask = np.zeros((h, w), dtype=np.float32)
                    m_conf = 0

            # calculate scores
            y_dice, y_iou, y_c = get_pixel_counts(y_mask, gt_mask)
            u_dice, u_iou, u_c = get_pixel_counts(u_mask, gt_mask)
            m_dice, m_iou, m_c = get_pixel_counts(m_mask, gt_mask)

            for i in range(4):
                total_counts["YOLO"][i] += y_c[i]
                total_counts["U-Net"][i] += u_c[i]
                total_counts["M-RCNN"][i] += m_c[i]

            # log metrics
            row = [idx, img_name, y_dice, u_dice, m_dice, y_iou, u_iou, m_iou, y_conf, u_conf, m_conf]
            writer.writerow(row)
            all_scores.append(row[2:])

            # Visualizations
            viz_data = [
                (img_rgb, "Original"),
                (gt_mask, "Ground Truth"),
                (y_mask, f"YOLO\nDice: {y_dice:.2f} | IoU: {y_iou:.2f}\nConf: {y_conf:.2%}"),
                (u_mask, f"U-Net\nDice: {u_dice:.2f} | IoU: {u_iou:.2f}\nConf: {u_conf:.2%}"),
                (m_mask, f"Mask R-CNN\nDice: {m_dice:.2f} | IoU: {m_iou:.2f}\nConf: {m_conf:.2%}")
            ]
            fig, axes = plt.subplots(1, 5, figsize=(22, 7))
            for i, (m, title) in enumerate(viz_data):
                axes[i].imshow(m, cmap='gray' if i > 0 else None)
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
            plt.savefig(os.path.join(OUTPUT_VIS_DIR, f"res_{img_name}"), bbox_inches='tight', dpi=150)
            plt.close(fig)

            if idx % 5 == 0: print(f"Processed {idx}/{len(image_files)}...")

        # Charts
        if all_scores:
            averages = np.mean(all_scores, axis=0)
            writer.writerow(["-", "AVERAGE"] + averages.tolist())
            
            fig_sum, ax1 = plt.subplots(figsize=(10, 6))
            models = ['YOLO', 'U-Net', 'M-RCNN']
            x = np.arange(len(models))
            ax1.bar(x - 0.2, averages[0:3], 0.4, label='Dice', color='skyblue')
            ax1.bar(x + 0.2, averages[3:6], 0.4, label='IoU', color='salmon')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models)
            ax1.set_ylim(0, 1.1)
            ax1.legend()
            ax1.set_title("Average Performance across Test Set")
            fig_sum.savefig(os.path.join(SUMMARY_DIR, "performance_bars.png"))

            # confusion matrices
            fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
            for i, model_name in enumerate(["YOLO", "U-Net", "M-RCNN"]):
                plot_confusion_matrix(total_counts[model_name], f"{model_name} Confusion Matrix", axes_cm[i])
            
            plt.tight_layout()
            plt.savefig(os.path.join(SUMMARY_DIR, "confusion_matrices.png"))
            print(f"\nProcessing Complete. Metrics saved to {CSV_OUTPUT_PATH}")
            plt.show()

if __name__ == "__main__":
    process_and_compare()