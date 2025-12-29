# src/models/CNN.py

import torch
import torch.nn as nn

from src.datasets.dataLoader import get_dataloaders


class LettuceCNN(nn.Module):
    """
    224x224 RGB 이미지를 입력으로 받는 간단한 CNN 분류기.
    conv → bn → relu → pool 반복 + AdaptiveAvgPool → FC
    """

    def __init__(self, num_classes: int=3):
        super().__init__()

        self.features = nn.Sequential(
            # 224x224x3 → 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 112x112x32 → 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 56x56x64 → 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 28x28x128 → 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # AdaptiveAvgPool2d(1x1) → 항상 (B, 256, 1, 1) → Linear(256 → num_classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),                 # (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),  # 최종 클래스 수
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model_and_dataloaders(
    processed_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str | None = None,
):
    """
    dataloader + 모델을 한 번에 만들어주는 편의 함수.

    - processed_root: crop_processed_data/lettuce_v1 같은 루트 경로
    - return: model, train_loader, test_loader, meta, device
    """

    # 1) DataLoader 생성 (네가 만든 함수 재사용)
    train_loader, test_loader, meta = get_dataloaders(
        processed_root=processed_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 2) 모델 생성 (클래스 개수는 meta에서 자동으로 가져옴)
    num_classes = meta["num_classes"]
    model = LettuceCNN(num_classes=num_classes)

    # 3) device 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    return model, train_loader, test_loader, meta, device


if __name__ == "__main__":
    """
    이 파일을 단독 실행했을 때, 데이터/모델이 잘 연결되는지 테스트용 코드.
    python -m src.models.CNN 이런 식으로 돌려보면 됨.
    """
    processed_root = "crop_processed_data/lettuce_v1"

    model, train_loader, test_loader, meta, device = create_model_and_dataloaders(
        processed_root=processed_root,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )

    print("[meta]", meta)

    # 배치 하나 뽑아서 모델에 통과시켜보기
    images, labels = next(iter(train_loader))
    print("이미지 배치 shape:", images.shape)   # (B, 3, 224, 224) 예상
    print("라벨 배치 shape:", labels.shape)    # (B,)

    images = images.to(device)
    outputs = model(images)
    print("모델 출력 shape:", outputs.shape)    # (B, num_classes) 예상
