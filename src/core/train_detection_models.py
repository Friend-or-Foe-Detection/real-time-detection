#!/usr/bin/env python3
"""
train_detection_models.py

Train and evaluate two lightweight detection models (SSDLite-MobileNetV3 and
Faster R-CNN MobileNetV3) on your combined fyp-1 and fyp-2 datasets for
friend/foe patch detection.

Usage:
  1. Install dependencies:
       pip install torch torchvision scikit-learn matplotlib pillow
  2. Place this script in your project root, alongside folders:
       fyp-1/images/, fyp-1/labels/, fyp-2/images/, fyp-2/labels/
  3. Run one of:
     # SSDLite-MobileNetV3
     python train_detection_models.py \
       --image-dirs fyp-1/images fyp-2/images \
       --label-dirs fyp-1/labels fyp-2/labels \
       --model ssdlite \
       --epochs 20 \
       --batch-size 4 \
       --imgsz 320

     # Faster R-CNN MobileNetV3
     python train_detection_models.py \
       --image-dirs fyp-1/images fyp-2/images \
       --label-dirs fyp-1/labels fyp-2/labels \
       --model fasterrcnn \
       --epochs 20 \
       --batch-size 4 \
       --imgsz 320
"""

import os
import argparse
import time
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    fasterrcnn_mobilenet_v3_large_fpn
)
import matplotlib.pyplot as plt


class DetectionDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, imgsz):
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.imgsz = imgsz
        self.transform = T.Compose([
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and apply transform
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)

        # Read YOLO-format labels and convert to pixel boxes
        W = H = self.imgsz
        boxes = []
        labels = []
        with open(self.lbl_paths[idx], 'r') as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.split())
                cx, cy = xc * W, yc * H
                bw, bh = w * W, h * H
                xmin = max(cx - bw/2, 0)
                ymin = max(cy - bh/2, 0)
                xmax = min(cx + bw/2, W)
                ymax = min(cy + bh/2, H)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(loader)


def evaluate_fps(model, loader, device):
    model.eval()
    start = time.time()
    count = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = list(img.to(device) for img in imgs)
            _ = model(imgs)
            count += len(imgs)
    elapsed = time.time() - start
    return count / elapsed


def main(args):
    # 1) Gather all image/label pairs
    img_paths, lbl_paths = [], []
    for imd, lbd in zip(args.image_dirs, args.label_dirs):
        for fn in os.listdir(imd):
            if fn.lower().endswith(('.jpg', '.png')):
                img = os.path.join(imd, fn)
                lbl = os.path.join(lbd, fn.rsplit('.',1)[0] + '.txt')
                if os.path.exists(lbl):
                    img_paths.append(img)
                    lbl_paths.append(lbl)
                else:
                    print(f"Warning: no label for {img}")

    # 2) Split into train/val
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_paths, lbl_paths,
        test_size=0.2,
        random_state=42
    )
    print(f"Dataset split: {len(train_imgs)} train, {len(val_imgs)} val samples")

    # 3) Create DataLoaders (drop_last on train)
    train_ds = DetectionDataset(train_imgs, train_lbls, args.imgsz)
    val_ds   = DetectionDataset(val_imgs,   val_lbls,   args.imgsz)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4) Device and model init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'ssdlite':
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=2).to(device)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 5) Training loop
    lrs = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {loss:.4f} — LR: {lr:.6f}")

    # 6) Plot LR schedule
    plt.figure(figsize=(6,4))
    plt.plot(range(1, args.epochs+1), lrs, marker='o')
    plt.title(f"{args.model} Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

    # 7) Evaluate inference speed
    fps = evaluate_fps(model, val_loader, device)
    print(f"\nInference speed on validation set: {fps:.2f} FPS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dirs', nargs='+', required=True,
                        help="Folders with images (space-separated)")
    parser.add_argument('--label-dirs', nargs='+', required=True,
                        help="Folders with labels (space-separated)")
    parser.add_argument('--model', choices=['ssdlite','fasterrcnn'], required=True,
                        help="Model to train: 'ssdlite' or 'fasterrcnn'")
    parser.add_argument('--epochs',    type=int,   default=20, help="Number of epochs")
    parser.add_argument('--batch-size',type=int,   default=4,  help="Batch size")
    parser.add_argument('--imgsz',     type=int,   default=320,help="Resize images to imgsz x imgsz")
    args = parser.parse_args()
    main(args)
