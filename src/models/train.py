# src/models/train.py

import argparse
import os
import csv
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.datasets.transform import get_transforms
from src.models.CNN import LettuceCNN  # Base
from src.models.ResSEAttnCNN import LettuceResSEAttnCNN # 실험3
from src.utils.weight_loss import make_weighted_ce_from_dataset # 실험1
from src.utils.oversampling import make_oversampler # 실험2
from src.utils.seed import set_seed
from src.utils.plot import plot_curves

# ---------------------------
# [후보] utils/metrics.py 로 뺄 함수
# ---------------------------
def accuracy(logits, targets):
    """배치 단위 정확도 계산"""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------
# [후보] engine/train_fn.py 로 뺄 함수
# ---------------------------

def train_one_epoch(loader, device, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    size = len(loader.dataset)

    # tqdm 추가
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / size

def evaluate(loader, device, model, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    size = len(loader.dataset)
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)

            bsz = images.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy(logits, targets) * bsz

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
        
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # macro F1 (클래스 불균형 고려한 지표)
    val_f1 = f1_score(all_targets, all_preds, average="macro")

    return total_loss / size, total_acc / size, val_f1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="runs_lettuce")
    parser.add_argument("--num_workers", type=int, default=2)

    # 이미지 폴더 루트: crop_processed_data/lettuce_v1/train 만 사용
    parser.add_argument(
        "--processed_root",
        type=str,
        default="crop_processed_data/lettuce_v1",
        help="crop_processed_data/lettuce_v1 경로 (train 폴더 포함)",
    )
    
    parser.add_argument("--resume", type=str, default=None,
                    help="이전 체크포인트 경로(.pt)를 불러와 이어서 학습")
    
    args = parser.parse_args()

    # 1) 시드 고정
    set_seed(args.seed)

    # 2) 디바이스
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] {device}")

    # 3) 결과 폴더
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # [ABLATION] 실험 시 마다, 이름 변경
    run_name = f"ABL_reduce_batch_bs{args.batch_size}_lr{args.lr}_{stamp}"
    run_dir = os.path.join(args.save_dir, "ablation_test", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Run dir] {run_dir}")

    # 4) config 저장
    config = {
        "seed": args.seed,
        "device": device,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "processed_root": args.processed_root,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------
    # 5) Dataset / DataLoader 구성 (train만 사용)
    # -------------------------------------------------
    train_root = os.path.join(args.processed_root, "train")
    val_root = os.path.join(args.processed_root, "val")
    
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train 폴더 없음: {train_root}")
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"train 폴더 없음: {val_root}")
    
    transforms_dict = get_transforms()
    train_tf = transforms_dict["train"]
    val_tf = transforms_dict["test"]  # 검증은 test용 transform 그대로 사용

    # 전체 train dataset (ImageFolder)
    train_ds = ImageFolder(train_root, transform=train_tf)
    val_ds = ImageFolder(val_root, transform=val_tf)
    
    num_classes = len(train_ds.classes)

    pin = (device == "cuda")

    sampler = make_oversampler(train_ds) # 실험 2
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        # shuffle=True, # base code
        sampler=sampler, # 실험2
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    # -------------------------------------------------
    # 6) 모델 / 손실 / 옵티마이저
    # -------------------------------------------------
    # model = LettuceCNN(num_classes=num_classes).to(device) # BASE
    model = LettuceResSEAttnCNN(num_classes=num_classes).to(device) # 실험3
  
    loss_fn = nn.CrossEntropyLoss() # base 코드
    # loss_fn = make_weighted_ce_from_dataset(train_ds, device=device) # 실험1
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # -------------------------
    # Resume from checkpoint
    # -------------------------
    start_epoch = 1   # 기본은 1 epoch부터

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Loaded checkpoint from {args.resume} (start from epoch {start_epoch})")
        
    # -------------------------------------------------
    # 7) 학습 루프 + 로그/그래프/체크포인트
    # -------------------------------------------------
    csv_path = os.path.join(run_dir, "metrics.csv")
    epochs_list = []
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    best_acc = 0.0
    best_path = None
    last_val_acc = 0.0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "lr"])

        for ep in range(start_epoch, args.epochs + 1):
            start = time.time()

            tr_loss = train_one_epoch(train_loader, device, model, loss_fn, optimizer)
            val_loss, val_acc, val_f1 = evaluate(val_loader, device, model, loss_fn)

            curr_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            print(
                f"Epoch {ep:03d} | "
                f"train_loss={tr_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc*100:.2f}% "
                f"val_f1={val_f1:.4f} "
                f"lr={curr_lr:.3e} ({elapsed:.1f}s)"
            )

            # 기록
            epochs_list.append(ep)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            
            writer.writerow([
                ep,
                f"{tr_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{val_f1:.6f}",
                f"{curr_lr:.6e}",
            ])
            f.flush()

            # 그래프 저장 (매 epoch마다 업데이트)
            plot_curves(
                epochs=epochs_list,
                train_losses=train_losses,
                val_losses=val_losses,
                val_accs=val_accs,
                val_f1s=val_f1s,
                save_dir=run_dir,
            )

            # best 모델 저장 (val 기준)
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(run_dir, f"best_ep{ep:03d}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": ep,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                    },
                    best_path,
                )

            last_val_acc = val_acc
            last_val_f1 = val_f1

    # 마지막 에포크 모델 저장
    final_path = os.path.join(run_dir, "last.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "val_acc": last_val_acc,
            "val_f1": last_val_f1,
        },
        final_path,
    )

    print(f"\nDone. CSV saved to: {csv_path}")
    print(f"Best model: {best_path}")
    print(f"Curves: loss_curve.png, acc_curve.png")


if __name__ == "__main__":
    main()
