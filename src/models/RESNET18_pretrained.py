# src/models/resnet18_pretrained.py

import torch.nn as nn
import torchvision.models as models


class ResNet18_Lettuce(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, head_only=False):
        super().__init__()

        # pretrained backbone 로드
        if pretrained:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.model = models.resnet18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # head-only 학습 옵션
        if head_only:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    @property
    def layer4(self):
        return self.model.layer4