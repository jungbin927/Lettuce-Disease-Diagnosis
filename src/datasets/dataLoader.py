# src/datasets/dataloader.py

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transform import get_transforms  # 너가 만든 transform.py


def get_dataloaders(
    processed_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    crop_processed_data/lettuce_v1/train, test 폴더 구조를 기반으로
    ImageFolder + DataLoader를 생성하여 (train_loader, test_loader, meta) 반환.

    parameters
    ----------
    processed_root : str
        crop_processed_data/lettuce_v1 의 경로
    batch_size : int
        배치 크기
    num_workers : int
        병렬 데이터 로딩에 사용할 worker 수
    pin_memory : bool
        GPU 학습 시 성능 최적화를 위한 옵션
    """

    # 1) 경로 설정
    train_root = os.path.join(processed_root, "train")
    test_root = os.path.join(processed_root, "test")

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"[에러] train 폴더가 없습니다: {train_root}")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"[에러] test 폴더가 없습니다: {test_root}")

    # 2) transforms 불러오기
    transforms_dict = get_transforms()
    train_tf = transforms_dict["train"]
    test_tf = transforms_dict["test"]

    # 3) ImageFolder로 Dataset 생성
    train_dataset = ImageFolder(train_root, transform=train_tf)
    test_dataset = ImageFolder(test_root, transform=test_tf)

    # 4) DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,     # 배치 누락 방지
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # 5) 부가 정보(meta)
    meta = {
        "num_classes": len(train_dataset.classes),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_len": len(train_dataset),
        "test_len": len(test_dataset),
    }

    print(f"[DataLoader 준비 완료]")
    print(f" - Train: {meta['train_len']} images")
    print(f" - Test : {meta['test_len']} images")
    print(f" - Classes: {meta['classes']}")

    return train_loader, test_loader, meta
