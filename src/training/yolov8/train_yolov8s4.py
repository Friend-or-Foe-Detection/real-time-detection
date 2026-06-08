#!/usr/bin/env python3
"""
train_yolov8s4.py

Reproduces the historical training run for yolov8s.pt with the exact parameters used.

Usage:
  1. Install dependencies:
       pip install ultralytics torch
  2. Run the script:
       python train_yolov8s4.py \
         --epochs 5 \
         --batch-size 64 \
         --imgsz 640
"""

import argparse
import time
import torch
from ultralytics import YOLO

def main(args):
    print(f"--- Starting Reproduction of yolov8s4 ---")
    
    # 1) Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    device_arg = 0 if device.type == 'cuda' else 'cpu'
    
    # 2) Model initialization
    print(f"Initializing YOLO model: yolov8s.pt")
    model = YOLO("yolov8s.pt")
    
    # 3) Training
    print("Starting training with historical parameters...")
    start_time = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        optimizer="auto",
        device=device_arg,
        project="runs/detect",
        name="yolov8s4_reproduced"
    )
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes.")
    
    # 4) Validation / Evaluation
    print("Evaluating model...")
    val_results = model.val(device=device_arg)
    print("Evaluation complete. Results saved to runs/detect.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproduce YOLO training run yolov8s4.")
    parser.add_argument('--data', type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size")
    args = parser.parse_args()
    
    main(args)
