import torch
from torch.utils.data import WeightedRandomSampler

def make_oversampler(dataset):
    """
    ImageFolder 기반 dataset에서 각 클래스 비율로 oversampling sampler 생성.
    dataset.targets 또는 dataset.samples 중 더 빠른 방식 사용.
    """

    # 1) 가장 빠르고 깔끔한 방식: dataset.targets
    if hasattr(dataset, "targets"):
        targets = dataset.targets

    # 2) fallback: dataset.samples (path, label)
    elif hasattr(dataset, "samples"):
        targets = [s[1] for s in dataset.samples]

    # 3) 최후 fallback: 느림 (피해야 함)
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]

    targets = torch.tensor(targets, dtype=torch.long)

    # 클래스별 개수 계산
    class_count = torch.bincount(targets)
    num_classes = len(class_count)

    # 클래스별 weight = 1 / class_count
    class_weights = 1.0 / class_count.float()

    # 각 샘플별 weight 생성
    sample_weights = class_weights[targets]

    # WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42)
    )

    return sampler
