import torch
import torch.nn as nn

from .modules import SEBlock, SpatialAttention, ResidualSEBlock


class LettuceResSEAttnCNN(nn.Module):
    """
    224x224 RGB 입력용:
    - 초반 stem conv
    - 4개의 stage: Residual + SE block
    - stage 사이에는 MaxPool로 downsampling
    - 마지막 stage 출력에 Spatial Attention 적용
    - AdaptiveAvgPool → FC 분류기
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # ▣ Stem: 224x224x3 → 224x224x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ▣ Stage 1: 224x224x32 → 112x112x32
        self.stage1_block = ResidualSEBlock(32, 32, stride=1, use_se=True)
        self.stage1_pool = nn.MaxPool2d(kernel_size=2)  # H,W / 2

        # ▣ Stage 2: 112x112x32 → 56x56x64
        self.stage2_block = ResidualSEBlock(32, 64, stride=1, use_se=True)
        self.stage2_pool = nn.MaxPool2d(kernel_size=2)

        # ▣ Stage 3: 56x56x64 → 28x28x128
        self.stage3_block = ResidualSEBlock(64, 128, stride=1, use_se=True)
        self.stage3_pool = nn.MaxPool2d(kernel_size=2)

        # ▣ Stage 4: 28x28x128 → 14x14x256
        # 여기서 한 번 더 깊게: block을 2개 쌓아서 살짝 깊이 늘림
        self.stage4_block1 = ResidualSEBlock(128, 256, stride=1, use_se=True)
        self.stage4_block2 = ResidualSEBlock(256, 256, stride=1, use_se=True)
        self.stage4_pool = nn.MaxPool2d(kernel_size=2)

        # ▣ 마지막 Spatial Attention (CBAM의 공간 부분)
        # [ABLATION] 실험1. Spatial Attention 제거
        self.spatial_att = SpatialAttention(kernel_size=7)

        # [ABLATION] 실험5. 뎁스 줄이기, Stage 4제거로 인한, 계수 변환
        # (128, 64) ->(64,3)
        # ▣ 분류기: AdaptiveAvgPool → FC
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),                 # (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)                 # (B, 32, 224, 224)

        # Stage 1
        x = self.stage1_block(x)         # (B, 32, 224, 224)
        x = self.stage1_pool(x)          # (B, 32, 112, 112)

        # Stage 2
        x = self.stage2_block(x)         # (B, 64, 112, 112)
        x = self.stage2_pool(x)          # (B, 64, 56, 56)

        # Stage 3
        x = self.stage3_block(x)         # (B, 128, 56, 56)
        x = self.stage3_pool(x)          # (B, 128, 28, 28)

        # [ABLATION] 실험5. 뎁스 줄이기, Stage 4 제거
        # Stage 4 (두 개 block으로 깊이 조금 더 줌)
        x = self.stage4_block1(x)        # (B, 256, 28, 28)
        x = self.stage4_block2(x)        # (B, 256, 28, 28)
        x = self.stage4_pool(x)          # (B, 256, 14, 14)

        # Spatial Attention
        # [ABLATION] 실험1. Spatial Attention 제거
        x = self.spatial_att(x)          # (B, 256, 14, 14)

        # 분류기
        x = self.classifier(x)           # (B, num_classes)
        return x
