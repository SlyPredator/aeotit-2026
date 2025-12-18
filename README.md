# Crack Segmentation Benchmark Repository

This repository provides a professional framework for training and comparing three distinct deep learning architectures—**YOLOv11-seg**, **U-Net**, and **Mask R-CNN**—specifically optimized for detecting and segmenting cracks in structural surfaces.

## Quick links to datasets (zip) and models used in this project

* [YOLO + UNet data](https://nitcacin-my.sharepoint.com/:u:/g/personal/navneeth_b230450me_nitc_ac_in/IQCoIrx9qbmdR4ZA24YFVa5XAeOUeJToJ4QlnNpUgQlBiVo?e=4ygABu)
* [MaskRCNN data](https://nitcacin-my.sharepoint.com/:u:/g/personal/navneeth_b230450me_nitc_ac_in/IQA1t7bUDUJcTLlmlKhNmC-1AW9uGRfsrG3BcuLGrNeB3M0?e=T1pIR8)
* [Trained models for evaluation](https://nitcacin-my.sharepoint.com/:f:/g/personal/navneeth_b230450me_nitc_ac_in/IgC0P7zbFywfQbXmiVF_jZy7AawEGASvldoKMQAtQ5txq4o?e=iLCVPE)
* [Benchmark data](https://nitcacin-my.sharepoint.com/:u:/g/personal/navneeth_b230450me_nitc_ac_in/IQDMfAvCT7iQSbZSVHtv-pKjAdUhlaaRNHLaCosY86RzpHY?e=8QvAo4)

## Repository Structure

### 1. Training Suite (`src/train/`)
These scripts handle the model-specific training logic and weight preservation.
* **`train_yolo.py`**: A script utilizing the Ultralytics framework to train a YOLOv11s-segmentation model. It is optimized for real-time performance and includes automatic hyperparameter tuning.
* **`train_unet.py`**: A custom PyTorch implementation of a U-Net with a ResNet50 encoder. It employs a hybrid **BCE + Dice Loss** to ensure both pixel-level accuracy and structural overlap, while a **ReduceLROnPlateau** scheduler manages the learning rate to capture fine crack details.
* **`train_maskrcnn.py`**: A script for Mask R-CNN that utilizes a ResNet50-FPN backbone. It features custom data handling for COCO-formatted datasets and includes safety checks to skip images without valid annotations during training.

### 2. Benchmarking & Evaluation (`src/eval/`)
Tools to generate quantitative metrics (Dice, IoU, Precision, Recall) and visual side-by-side comparisons.
* **`benchmark_single.py`**: Designed for rapid testing, this script runs a single image through all three models at a shared `CONFIDENCE` threshold and displays a 5-panel visualization (Original, GT, and 3 Predictions).
* **`benchmark_dataset.py`**: A comprehensive evaluation engine that iterates through a test dataset, logs 11 different metrics per image to a CSV file, and generates global summary charts, including average performance bars and normalized confusion matrices.
* **`legacy_bench_yolo_unet.py`**: A specialized comparison script for evaluating the direct performance of YOLO against U-Net, useful for comparing one-stage vs. two-stage segmentation logic.

### 3. Data Utilities (`src/data/`)
Essential tools for dataset management and format conversion.
* **`dataset_download_yolo.py`**: Connects to the Roboflow API to fetch the latest dataset version formatted for YOLO segmentation.
* **`dataset_download_coco.py`**: Fetches the same dataset version but formatted as a COCO JSON, required for the Mask R-CNN training pipeline.
* **`yolo_to_mask.py`**: A critical conversion utility that reads YOLO normalized polygon coordinates and renders them into high-resolution binary `.png` masks. This creates the ground-truth data necessary for the U-Net training process.

## Getting Started

1.  **Data Acquisition**: Execute `dataset_download_yolo.py` and `dataset_download_coco.py` to populate your local directories.
2.  **Preprocessing**: Run `yolo_to_mask.py` to generate the binary masks required for semantic segmentation training.
3.  **Model Training**: Use the scripts in `src/train/` to generate model weights (`.pt` for YOLO, `.pth` for U-Net and Mask R-CNN).
4.  **Final Comparison**: Execute `benchmark_dataset.py` to generate the `model_comparison_metrics.csv` file and visual summaries for your research or report.