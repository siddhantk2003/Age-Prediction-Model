# efficientnet_model.py
"""
EfficientNet-B0 backbone modified for age regression (single-output).
Works with multiple torchvision versions (fallback for different APIs).
"""
import torch
import torch.nn as nn
import torchvision

def get_efficientnet_b0_regression(pretrained=True, device='cpu'):
    """
    Returns EfficientNet-B0 with its classifier changed to output 1 value for age regression.
    """
    # Support older/newer torchvision versions
    try:
        # Newer torchvision: weights param
        # e.g. torchvision 0.13+ may support weights
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
    except Exception:
        # Older torchvision: pretrained boolean
        model = torchvision.models.efficientnet_b0(pretrained=pretrained)

    # Replace classifier head with single-output linear layer
    if hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
        # replace classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1)
        )
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise RuntimeError("Unexpected EfficientNet model structure. Please inspect model.")

    # Ensure model outputs a float scalar (no sigmoid)
    model.to(device)
    return model
