# 데이터 증강 파일
"""
학습(train) / 검증(val) / 테스트(test)용 transform 묶어서 반환.

- train: Resize → RandomCrop → RandomHorizontalFlip → RandomRotation →
        ColorJitter → ToTensor → Normalize
- val/test: Resize → CenterCrop → ToTensor → Normalize
"""
# src/datasets/transform.py

from typing import Tuple, Dict
from torchvision import transforms


IMAGE_MEAN = (0.460, 0.534, 0.274)
IMAGE_STD  = (0.187, 0.176, 0.226)

def get_transforms(
    resize_size: int = 256,
    crop_size: int = 224,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> Dict[str, transforms.Compose]:


    train_transform = transforms.Compose([
        # 먼저 전체 이미지를 적당히 리사이즈
        transforms.Resize((resize_size, resize_size)),
        # 랜덤 크롭으로 위치 변화
        transforms.RandomCrop(crop_size),
        # 좌우 반전
        transforms.RandomHorizontalFlip(p=0.5),
        # 약간의 회전 
        transforms.RandomRotation(degrees=15),
        # 밝기/대비/채도/색조 살짝 랜덤하게
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
      # PIL → Tensor, [0,1] 범위
        transforms.ToTensor(),
        # 정규화
        transforms.Normalize(mean=mean, std=std),
    ])

    # 검증/테스트는 랜덤성 최소화
    eval_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {
        "train": train_transform,
        "test": eval_transform,
    }
