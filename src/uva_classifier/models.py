import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Keep the classifier shape aligned with the original 64x64 CNN script.
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def default_image_size_for_model(model_name):
    return 224 if model_name == "resnet18" else 64


def _load_resnet18(pretrained):
    weights_enum = getattr(models, "ResNet18_Weights", None)
    if weights_enum is not None:
        weights = weights_enum.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    return models.resnet18(pretrained=pretrained)


def build_model(model_name, num_classes=2, dropout=0.3, pretrained=True, freeze_backbone=False):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    if model_name == "resnet18":
        model = _load_resnet18(pretrained=pretrained)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        return model

    raise ValueError(f"Unsupported model name: {model_name}")
