#!/usr/bin/env python3
"""
train_compare_yolo_fixed.py

Train YOLOv8s & YOLO11s for 5 epochs, then overlay their
training-loss and learning-rate curves in the same plot.

Usage:
  pip install ultralytics scikit-learn matplotlib pyyaml pillow
  python train_compare_yolo_fixed.py \
    --image-dirs fyp-1/images fyp-2/images \
    --label-dirs fyp-1/labels fyp-2/labels \
    --epochs 5 \
    --batch-size 4 \
    --imgsz 640
"""
import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- Data Preparation ---
def prepare_dataset(img_dirs, lbl_dirs, out_dir, test_size=0.2, seed=42):
    # Create train/val subfolders
    for sub in ('images/train','images/val','labels/train','labels/val'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    imgs, lbls = [], []
    for imd, lbd in zip(img_dirs, lbl_dirs):
        for fn in os.listdir(imd):
            if fn.lower().endswith(('.jpg','.png')):
                lp = os.path.join(lbd, fn.rsplit('.',1)[0] + '.txt')
                if os.path.exists(lp):
                    imgs.append(os.path.join(imd, fn))
                    lbls.append(lp)

    # Split 80/20
    tr_i, val_i, tr_l, val_l = train_test_split(imgs, lbls, test_size=test_size, random_state=seed)
    # Copy files
    for img_list, lbl_list, subset in ((tr_i,tr_l,'train'), (val_i,val_l,'val')):
        for img, lbl in zip(img_list, lbl_list):
            shutil.copy(img, os.path.join(out_dir, 'images', subset, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(out_dir, 'labels', subset, os.path.basename(lbl)))

# --- YAML Writing ---
def write_yaml(out_dir, classes):
    cfg = {
        'train': f'{out_dir}/images/train',
        'val':   f'{out_dir}/images/val',
        'nc':    len(classes),
        'names': classes
    }
    with open('data.yaml','w') as f:
        yaml.dump(cfg, f)
    print('✅ data.yaml written')

# --- Model Training ---
def train_model(weights, args):
    # Returns the run directory
    run_name = os.path.splitext(weights)[0]
    YOLO(weights).train(
        data='data.yaml',
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        name=run_name,
        plots=False  # disable built-in plots
    )
    return f'runs/detect/{run_name}/results.csv'

# --- Plotting Overlays ---
def overlay_plots(runs, args):
    import pandas as pd
    # Define possible column names for losses and LR
    loss_column_sets = [
        ('train/box_loss','train/cls_loss','train/dfl_loss'),
        ('box_loss','cls_loss','dfl_loss'),
        ('metrics/box','metrics/cls','metrics/dfl')
    ]
    lr_columns = ['lr','lr0','lr/pg0']

    # Plot Loss Overlay
    plt.figure(figsize=(8,5))
    for run_name, csv_path in runs.items():
        df = pd.read_csv(csv_path)
        # Sum appropriate loss columns
        for b_col, c_col, d_col in loss_column_sets:
            if all(col in df.columns for col in (b_col, c_col, d_col)):
                df['total_loss'] = df[b_col] + df[c_col] + df[d_col]
                break
        else:
            raise KeyError(f"Loss columns not found in {csv_path}: {list(df.columns)}")
        df = df[df['epoch'] <= args.epochs]
        plt.plot(df['epoch'], df['total_loss'], '-o', label=f'{run_name} Loss')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Learning Rate Overlay
    plt.figure(figsize=(8,5))
    for run_name, csv_path in runs.items():
        df = pd.read_csv(csv_path)
        for lr_col in lr_columns:
            if lr_col in df.columns:
                series = df[lr_col]
                break
        else:
            raise KeyError(f"LR column not found in {csv_path}: {list(df.columns)}")
        df = df[df['epoch'] <= args.epochs]
        plt.plot(df['epoch'], series, '-o', label=f'{run_name} LR')
    plt.title('Learning Rate vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dirs', nargs='+', required=True)
    parser.add_argument('--label-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', default='yolo_dataset')
    parser.add_argument('--classes', nargs='+', default=['friend','foe'])
    parser.add_argument('--epochs',    type=int, default=5)
    parser.add_argument('--batch-size',type=int, default=4)
    parser.add_argument('--imgsz',     type=int, default=640)
    args = parser.parse_args()

    # Prepare data and config
    prepare_dataset(args.image_dirs, args.label_dirs, args.output_dir)
    write_yaml(args.output_dir, args.classes)

    # Specify models
    models = [
        ('YOLOv8s',  'yolov8s.pt'),
        ('YOLO11s', 'yolo11s.pt')
    ]
    runs = {}
    for model_name, weights in models:
        if os.path.isfile(weights):
            print(f"📦 Found weights for {model_name}: {weights}")
            runs[model_name] = train_model(weights, args)
        else:
            print(f"⚠️ {weights} not found – skipping {model_name}")

    # Overlay plots for all trained models
    overlay_plots(runs, args)
