# src/models/modules.py
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 블록.
    (B, C, H, W) → 채널별 중요도를 학습해 가중치를 곱해줌.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)
        hidden = max(channels // reduction, 4)   # 너무 작아지는 것 방지용

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)    # (B, C)
        y = self.fc(y).view(b, c, 1, 1)    # (B, C, 1, 1)
        return x * y                       # 채널별 가중치 곱
        

class SpatialAttention(nn.Module):
    """
    간단한 Spatial Attention.
    채널 방향으로 평균/최댓값을 뽑아서 (B, 2, H, W)로 만들고,
    7x7 conv로 어느 위치가 중요한지 학습.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 채널 방향 평균/최댓값: (B, 1, H, W) 두 개
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = self.conv(attn)                       # (B, 1, H, W)
        attn = self.sigmoid(attn)
        return x * attn                              # 위치별 가중치 곱
        

class ResidualSEBlock(nn.Module):
    """
    ResNet BasicBlock + SE.
    in_channels -> out_channels로 가면서,
    Conv-BN-ReLU-Conv-BN-SE + skip connection 구조.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # [ABLATION] 실험 2. SE 제거
        # [ABLATION] 실험 4. SE, Spatial attention 제거
        self.use_se = use_se
        self.se = SEBlock(out_channels) if use_se else None
        # self.use_se = False # 실험 2,4
        # self.se = None # 실험 2,4

        # in/out이 다르거나 stride가 1이 아니면 shortcut에서 차원 맞춰줌
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # [ABLATION] 실험2. SE 제거
        if self.use_se and self.se is not None:
            out = self.se(out)  # 채널 어텐션

        # [ABLATION] 실험3. Skip_connection 제거 
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out # self.relu(out)
