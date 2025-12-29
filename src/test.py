# src.test.py

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt 

from src.models.CNN import LettuceCNN    # or ResNet version
from src.models.RESNET18_pretrained import ResNet18_Lettuce
from src.models.ResSEAttnCNN import LettuceResSEAttnCNN
from src.datasets.transform import get_transforms
from src.utils.heatmap import save_confusion_matrix

# --------------------------------------------------
# 1) Argument parsing
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Test a lettuce disease model")

    parser.add_argument("--data_root", type=str, required=True,
                        help="테스트 이미지 폴더 (ImageFolder 구조)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="모델 체크포인트 (.pth)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "resnet18","resseattn"],
        help="사용할 모델 종류 (cnn, resnet18, resseattn)"
    )
    parser.add_argument(
        "--cm_path",
        type=str,
        default="confusion_matrix.png",
        help="혼동행렬(heatmap) 저장 경로"
    )
    return parser.parse_args()


# --------------------------------------------------
# 2) Build dataloader for test
# --------------------------------------------------
def load_test_loader(args):
    tf = get_transforms()
    test_tf = tf["test"]

    dataset = ImageFolder(root=args.data_root, transform=test_tf)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)

    return loader, dataset.classes


# --------------------------------------------------
# 3) Load model + checkpoint
# --------------------------------------------------
def build_model(args):
    if args.model == "cnn":
        model = LettuceCNN(num_classes=args.num_classes)

    elif args.model == "resnet18":
        model = ResNet18_Lettuce(
            num_classes=args.num_classes,
            pretrained=False,   # test 때는 ckpt로 덮어써서 굳이 pretrained 필요 없음
            head_only=False
        )
    elif args.model == "resseattn":              # ★ 추가
        model = LettuceResSEAttnCNN(
            num_classes=args.num_classes
        )
        
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    return model
        
def load_model(args):
    device = args.device

    model = build_model(args)
    ckpt = torch.load(args.ckpt_path, map_location=device)

   # checkpoint 구조에 따라 분기
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        # 어떤 코드는 {"model": state_dict} 이렇게 저장하기도 함
        state_dict = ckpt["model"]
    else:
        # 바로 state_dict만 저장한 경우
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# 4) Evaluation loop (NO grad)
# --------------------------------------------------
def evaluate(model, loader, device):
    preds, labels = [], []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            out = model(imgs)
            pr = softmax(out).argmax(dim=1)

            preds.append(pr.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "cm": cm
    }


# --------------------------------------------------
# 5) Main execution
# --------------------------------------------------
def main():
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"Model : {args.model}")

    loader, class_names = load_test_loader(args)
    model = load_model(args)
    
    results = evaluate(model, loader, args.device)

    acc = results["acc"]
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    report = results["report"]
    cm = results["cm"]


    print("===== TEST RESULTS =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")

    print("\n--- Classification Report ---")
    print(report)

    print("\n--- Confusion Matrix ---")
    print(cm)

    print("\nClass Index Mapping:")
    for idx, name in enumerate(class_names):
        print(f"{idx} → {name}")

    # 🔥 Heatmap 저장
    save_confusion_matrix(cm, class_names, args.cm_path)


if __name__ == "__main__":
    main()