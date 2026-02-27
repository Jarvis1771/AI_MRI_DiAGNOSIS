import torch
import torch.nn as nn
from torchvision import models


def load_model():
    # Load EfficientNet WITHOUT pretrained weights (no internet needed)
    model = models.efficientnet_b0(weights=None)

    # Modify classifier for medical classes
    num_features = model.classifier[1].in_features

    # Adjust number of classes as per LABELS
    model.classifier[1] = nn.Linear(num_features, len([0, 1, 2]))

    model.eval()
    return model
