#!/usr/bin/env python3
"""
train_ssdlite.py

Train SS DLite-MobileNetV3 for friend/foe patch detection using:
  - pretrained ImageNet backbone (MobileNetV3-Large)
  - detection head initialized from scratch (2 classes)
  - automatic CUDA usage

Usage:
  1. Install:
       pip install torch torchvision scikit-learn matplotlib pillow
  2. Ensure:
       fyp-1/images/, fyp-1/labels/, fyp-2/images/, fyp-2/labels/
  3. Run:
       python train_ssdlite.py --epochs 20 --batch-size 4 --imgsz 320
"""

import os, time, argparse, shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import matplotlib.pyplot as plt

class DetectionDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, imgsz):
        self.img_paths, self.lbl_paths, self.imgsz = img_paths, lbl_paths, imgsz
        self.transform = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        # image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        # boxes & labels
        W = H = self.imgsz
        boxes, labels = [], []
        with open(self.lbl_paths[idx]) as f:
            for line in f:
                cls, xc, yc, w, h = map(float, line.split())
                cx, cy = xc*W, yc*H; bw, bh = w*W, h*H
                xmin, ymin = max(cx-bw/2,0), max(cy-bh/2,0)
                xmax, ymax = min(cx+bw/2,W), min(cy+bh/2,H)
                boxes.append([xmin,ymin,xmax,ymax]); labels.append(int(cls))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

def collate_fn(batch): return tuple(zip(*batch))

def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0
    for imgs, tgts in loader:
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(loss for loss in loss_dict.values())
        optim.zero_grad(); loss.backward(); optim.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def main(args):
    # gather pairs
    img_paths, lbl_paths = [], []
    for imd, lbd in zip(args.image_dirs, args.label_dirs):
        for fn in os.listdir(imd):
            if fn.lower().endswith(('.jpg','.png')):
                lp = os.path.join(lbd, fn.rsplit('.',1)[0]+'.txt')
                if os.path.exists(lp):
                    img_paths.append(os.path.join(imd, fn))
                    lbl_paths.append(lp)
    # split
    tr_i, val_i, tr_l, val_l = train_test_split(
        img_paths, lbl_paths, test_size=0.2, random_state=42)
    print(f"Split: {len(tr_i)} train, {len(val_i)} val")

    # loaders
    train_ds = DetectionDataset(tr_i, tr_l, args.imgsz)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # model: pretrained backbone, fresh head for 2 classes
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        num_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # train
    lrs, losses = [], []
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_ld, optimizer, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr); losses.append(loss)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {loss:.4f} — LR: {lr:.6f}")

    # plot
    epochs = list(range(1, args.epochs+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, losses, marker='o'); plt.title("Train Loss"); plt.xlabel("Epoch"); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, lrs,    marker='o'); plt.title("LR Schedule"); plt.xlabel("Epoch"); plt.grid(True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--image-dirs', nargs='+', default=['fyp-1/images','fyp-2/images'])
    p.add_argument('--label-dirs', nargs='+', default=['fyp-1/labels','fyp-2/labels'])
    p.add_argument('--epochs',     type=int,   default=5)
    p.add_argument('--batch-size', type=int,   default=)
    p.add_argument('--imgsz',      type=int,   default=320)
    p.add_argument('--lr',         type=float, default=0.005)
    p.add_argument('--step-size',  type=int,   default=5)
    p.add_argument('--gamma',      type=float, default=0.5)
    args = p.parse_args()
    main(args)
