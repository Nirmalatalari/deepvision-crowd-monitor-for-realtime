# model.py
import torch
import torch.nn as nn
import torchvision

def build_model(pretrained=True, freeze_backbone=True):
    # MobileNetV2 backbone
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
    in_features = backbone.classifier[1].in_features  # last linear input size

    # replace classifier with regression head
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 1)   # single output (count)
    )

    if freeze_backbone:
        # freeze feature extractor (optional)
        for name, param in backbone.features.named_parameters():
            param.requires_grad = False

    return backbone
