#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_and_process(csv_path, max_epoch=None):
    df = pd.read_csv(csv_path)
    # find all columns containing "loss" (case-insensitive)
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    if not loss_cols:
        print(f"❌ No loss columns found in {csv_path} (found: {df.columns.tolist()})", file=sys.stderr)
        sys.exit(1)
    df['total_loss'] = df[loss_cols].sum(axis=1)

    # find any column that starts with "lr"
    lr_cols = [c for c in df.columns if c.lower().startswith('lr')]
    if not lr_cols:
        print(f"❌ No LR columns found in {csv_path} (found: {df.columns.tolist()})", file=sys.stderr)
        sys.exit(1)
    df['lr_used'] = df[lr_cols[0]]

    # optionally truncate to max_epoch
    if max_epoch is not None and 'epoch' in df.columns:
        df = df[df['epoch'] <= max_epoch]
    return df

def main():
    # adjust these paths if your CSVs live elsewhere
    csv_paths = {
        'YOLOv8s':  'runs/detect/yolov8s3/results.csv',
        'YOLOv11s': 'runs/detect/yolo11s2/results.csv'
    }

    # load + process each
    dfs = {}
    for name, path in csv_paths.items():
        try:
            dfs[name] = load_and_process(path, max_epoch=5)
            print(f"✅ Loaded {name} from {path}")
        except FileNotFoundError:
            print(f"⚠️  File not found: {path}, skipping {name}", file=sys.stderr)

    # Plot both on one page
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # Loss overlay
    for name, df in dfs.items():
        ax1.plot(df['epoch'], df['total_loss'], '-o', label=name)
    ax1.set_title('Training Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.grid(True)
    ax1.legend()

    # LR overlay
    for name, df in dfs.items():
        ax2.plot(df['epoch'], df['lr_used'], '-o', label=name)
    ax2.set_title('Learning Rate vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)
    plt.show()
    print("🖼️  Saved figure as comparison_plot.png")

if __name__=='__main__':
    main()
