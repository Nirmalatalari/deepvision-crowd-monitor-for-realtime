# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NumpyDataset
from model import build_model
import numpy as np
from tqdm import tqdm

# Config
BATCH_SIZE = 16
EPOCHS = 25
IMAGE_SIZE = (224, 224)
LR = 1e-4
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets))

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2))

def main():
    train_ds = NumpyDataset("data/train_images.npy", "data/train_labels.npy", image_size=IMAGE_SIZE)
    val_ds   = NumpyDataset("data/val_images.npy", "data/val_labels.npy", image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = build_model(pretrained=True, freeze_backbone=True).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_mae = 1e9
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)  # (B,1)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_mae, val_rmse = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                val_mae += torch.sum(torch.abs(preds - labels)).item()
                val_rmse += torch.sum((preds - labels).cpu().numpy() ** 2)

        val_mae = val_mae / len(val_loader.dataset)
        val_rmse = (val_rmse / len(val_loader.dataset)) ** 0.5

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_MAE={val_mae:.3f}, val_RMSE={val_rmse:.3f}")

        # save best by MAE
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    main()
