# src/datasets/make_val_from_train.py

import os
import random
import shutil
import argparse


def split_train_to_val(
    root_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    move: bool = True,  # True면 train에서 빼서 val로 "이동", False면 "복사"
):
    """
    root_dir/train 아래 클래스별 폴더에서 일부 이미지를 뽑아
    root_dir/val 아래 동일한 클래스 구조로 옮기거나 복사하는 함수.

    예)
    root_dir/
      ├ train/normal/*.jpg
      └ train/disease/*.jpg

    -> 실행 후

    root_dir/
      ├ train/normal/...
      ├ train/disease/...
      ├ val/normal/...
      └ val/disease/...
    """
    random.seed(seed)

    train_root = os.path.join(root_dir, "train")
    val_root = os.path.join(root_dir, "val")
    os.makedirs(val_root, exist_ok=True)

    # train/ 아래 클래스별 폴더 순회
    for class_name in sorted(os.listdir(train_root)):
        class_train_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_train_dir):
            continue  # 폴더 아닌 건 무시

        class_val_dir = os.path.join(val_root, class_name)
        os.makedirs(class_val_dir, exist_ok=True)

        # 이미지 파일 목록 수집
        filenames = [
            f for f in os.listdir(class_train_dir)
            if f.lower().endswith(exts)
        ]
        filenames.sort()  # 정렬 후 샘플링하면 디버깅 편함

        n_total = len(filenames)
        n_val = int(n_total * val_ratio)

        if n_val == 0:
            print(f"[경고] {class_name} 클래스는 이미지가 너무 적어서 val로 안 뽑음 (총 {n_total}장)")
            continue

        val_files = random.sample(filenames, n_val)

        print(f"[{class_name}] total={n_total}, val={n_val}")

        for fname in val_files:
            src = os.path.join(class_train_dir, fname)
            dst = os.path.join(class_val_dir, fname)
            if move:
                shutil.move(src, dst)  # train → val 이동
            else:
                shutil.copy2(src, dst)  # train 유지 + val에 복사


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="crop_processed_data/lettuce_v1",
        help="train/ 폴더가 들어있는 루트 경로",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="train에서 validation으로 보낼 비율 (0.2면 8:2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="항상 같은 애들이 val로 가도록 난수 고정",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="기본은 move(이동). 이 옵션을 켜면 copy(복사)로 동작",
    )
    args = parser.parse_args()

    split_train_to_val(
        root_dir=args.root_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        move=not args.copy,
    )

    print("✅ train → val 분리 완료")
