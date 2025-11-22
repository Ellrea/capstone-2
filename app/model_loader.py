import torch
import torch.nn as nn
from torchvision import models

def load_resnet_single(ckpt_path: str, num_classes: int, device: str = "cpu"):

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = blob.get("model_state", blob)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model