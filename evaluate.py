# evaluate.py
import torch
from dataset import NumpyDataset
from model import build_model
from torch.utils.data import DataLoader
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    test_ds = NumpyDataset("data/test_images.npy", "data/test_labels.npy", image_size=(224,224))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    model = build_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()

    total_mae = 0.0
    total_mse = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            preds = model(imgs)
            diff = (preds - labels).cpu().numpy().reshape(-1)
            total_mae += np.sum(np.abs(diff))
            total_mse += np.sum(diff**2)

    n = len(test_loader.dataset)
    mae = total_mae / n
    rmse = (total_mse / n) ** 0.5
    print(f"Test MAE: {mae:.3f}, Test RMSE: {rmse:.3f}")

if __name__ == "__main__":
    main()
