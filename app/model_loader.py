# app/model_loader.py
import torch, torch.nn as nn
from torchvision import models

def load_resnet_auto(ckpt_path: str, device: str = "cpu"):
    """ckpt의 fc.out_features(=num_classes)를 자동 감지해서 모델 구성."""
    blob = torch.load(ckpt_path, map_location=device)
    sd = blob["model_state"]

    # fc.weight shape: [num_classes, 512]
    num_classes = sd["fc.weight"].shape[0]

    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()

    return m, num_classes
