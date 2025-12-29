import torch
import torch.nn as nn
from collections import Counter

def make_weighted_ce_from_dataset(dataset, device="cpu"):
    """
    ImageFolder 같은 dataset에서 label 분포를 보고
    class_weight를 계산한 뒤 weighted CrossEntropyLoss를 반환
    """

    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "samples"):
        # (path, label) 리스트가 있는 경우
        labels = [s[1] for s in dataset.samples]
    else:
        # 최악의 fallback: __getitem__ 돌면서 label만 뽑기 (느림)
        labels = [dataset[i][1] for i in range(len(dataset))]

    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)

    class_weights = []
    for c in range(num_classes):
        count_c = class_counts[c]
        w_c = total_samples / (num_classes * count_c)  # balanced 방식
        class_weights.append(w_c)

    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    print("[Weighted CE] class_counts:", class_counts)
    print("[Weighted CE] class_weights:", class_weights)

    return nn.CrossEntropyLoss(weight=class_weights)
