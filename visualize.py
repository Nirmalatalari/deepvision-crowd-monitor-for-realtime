# visualize.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import NumpyDataset
from model import build_model
from torch.utils.data import DataLoader
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (224,224)

# ---- helper to find last Conv2d layer ----
def find_last_conv_module(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv

def make_gradcam_heatmap(model, input_tensor, class_idx=None):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # (1,C,H,W)
    # find last conv
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
            last_conv_name = name
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model")

    features = None
    grads = None

    def forward_hook(module, inp, outp):
        nonlocal features
        features = outp.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()

    handle_f = last_conv.register_forward_hook(forward_hook)
    handle_b = last_conv.register_backward_hook(backward_hook)

    output = model(input_tensor)  # shape (1,1)
    if class_idx is None:
        score = output[0,0]
    else:
        score = output[0, class_idx]

    model.zero_grad()
    score.backward(retain_graph=True)

    handle_f.remove()
    handle_b.remove()

    # features: (1, C, Hf, Wf), grads: (1, C, Hf, Wf)
    weights = torch.mean(grads, dim=(2,3), keepdim=True)  # (1,C,1,1)
    cam = torch.sum(weights * features, dim=1).squeeze(0)  # (Hf, Wf)
    cam = torch.relu(cam).cpu().numpy()
    # normalize
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))  # resize to model input size
    return cam

def denormalize(tensor):
    # tensor: CxHxW normalized by Imagenet mean/std
    mean = np.array([0.485,0.456,0.406])[:,None,None]
    std  = np.array([0.229,0.224,0.225])[:,None,None]
    arr = tensor.cpu().numpy()
    arr = (arr * std) + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1,2,0))  # H,W,C
    return arr

def visualize_random(n=4):
    ds = NumpyDataset("data/test_images.npy", "data/test_labels.npy", image_size=IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    model = build_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()

    import random
    for i in range(n):
        idx = random.randint(0, len(ds)-1)
        img_tensor, label = ds[idx]
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(DEVICE)).cpu().item()

        heatmap = make_gradcam_heatmap(model, img_tensor)
        orig = denormalize(img_tensor)

        # heatmap color
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig[..., ::-1], 0.6, heatmap_color, 0.4, 0)  # orig is RGB -> convert to BGR

        # convert back to RGB for matplotlib
        overlay_rgb = overlay[..., ::-1]

        fig, ax = plt.subplots(1,3,figsize=(15,5))
        ax[0].imshow(orig); ax[0].set_title(f"Original\nTrue: {label:.0f}"); ax[0].axis("off")
        ax[1].imshow(heatmap, cmap="jet"); ax[1].set_title(f"Grad-CAM Heatmap\nPred: {pred:.1f}"); ax[1].axis("off")
        ax[2].imshow(overlay_rgb); ax[2].set_title("Overlay"); ax[2].axis("off")
        plt.tight_layout()
        fname = f"outputs/vis_{i}.png"
        plt.savefig(fname, dpi=150)
        print("Saved", fname)
        plt.show()

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    visualize_random(n=3)
