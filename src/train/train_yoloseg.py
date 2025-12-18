from ultralytics import YOLO
import torch

# Training YOLOv11s-seg

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model = YOLO('yolo11s-seg.pt')
    model.train(
        data=r"E:\Navneeth\Crack-Segmentation-1\data.yaml",
        epochs=50,
        optimizer='auto',
        cache='disk',
        project='seg',
        name='run',
        workers=8,
        device=device
    )

if __name__ == "__main__":
    main()