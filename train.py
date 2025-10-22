"""
Training script (CPU-optimized) for the EfficientNet-B0 age regression model.
Saves best model to models/age_predictor.pth
"""
import os
import argparse
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from efficientnet_model import get_efficientnet_b0_regression
from utils import UTKFaceDataset, default_transforms

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  # shape (B,)
        loss = criterion(outputs, targets.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Val  ", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets.squeeze(1))
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def main(args):
    device = torch.device("cpu")  # force CPU for MacBook i5
    os.makedirs("models", exist_ok=True)

    # Transforms
    train_tf = default_transforms(train=True)
    val_tf = default_transforms(train=False)

    # Datasets
    full_ds = UTKFaceDataset(args.data_dir, transform=train_tf)
    n = len(full_ds)
    if n == 0:
        raise RuntimeError(f"No images found in {args.data_dir}")

    # Simple split: 90% train, 10% val
    split = int(0.9 * n)
    train_files = full_ds.files[:split]
    val_files = full_ds.files[split:]

    train_ds = UTKFaceDataset(args.data_dir, transform=train_tf, limit=None)
    # hack: replace file list to split properly
    train_ds.files = train_files
    val_ds = UTKFaceDataset(args.data_dir, transform=val_tf, limit=None)
    val_ds.files = val_files

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_efficientnet_b0_regression(pretrained=True, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join("models", "age_predictor.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print("Saved best model to", save_path)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.2f} minutes. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./UTKFace",
                   help="Path to UTKFace image folder (images named age_gender_race_date.jpg)")
    p.add_argument("--epochs", type=int, default=6, help="Number of epochs (keep small for CPU)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size (16 recommended for CPU)")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = p.parse_args()
    main(args)
