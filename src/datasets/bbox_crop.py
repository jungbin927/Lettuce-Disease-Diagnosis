# 원본 이미지를 상추 부위의 bbox로 자르는 작업

# src/datasets/bbox_crop.py

from pathlib import Path
import os
from typing import Dict

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===============================
# 경로 설정
# ===============================
this_dir = Path(__file__).resolve().parent      # .../src/datasets
project_root = this_dir.parent.parent           # .../Lettuce_Disease_Diagnosis

PROCESSED_DIR = project_root / "processed_data" / "lettuce_v1"
CROP_ROOT = project_root / "crop_processed_data" / "lettuce_v1"

# CSV 컬럼명
IMG_COL = "image"
LABEL_COL = "disease"
BBOX_COLS = ["xtl", "ytl", "xbr", "ybr"]

# 원본 이미지 루트
data_root = project_root / "original_data"

# 사용할 이미지 확장자
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


# ==========================================
# 1) original_data 전체를 스캔해서 매핑 생성
# ==========================================
def build_id_to_path_map(data_root: Path) -> Dict[str, Path]:
    """
    original_data/train/test/picture/normal,disease 하위(배치 폴더 포함)를
    전부 훑어서
      stem(확장자 제거, 소문자) -> 실제 이미지 경로(Path)
    로 매핑하는 함수
    """
    img_dirs = [
        data_root / "train" / "picture" / "normal",
        data_root / "train" / "picture" / "disease",
        data_root / "test" / "picture" / "normal",
        data_root / "test" / "picture" / "disease",
    ]

    id_to_path: Dict[str, Path] = {}

    for root_dir in img_dirs:
        if not root_dir.is_dir():
            print(f"[경고] 폴더 없음, 건너뜀: {root_dir}")
            continue

        #  하위 batch 폴더까지 재귀적으로 모두 탐색
        for cur_dir, _, files in os.walk(root_dir):
            cur_dir = Path(cur_dir)
            for fname in files:
                fpath = cur_dir / fname
                if not fpath.is_file():
                    continue

                stem, ext = os.path.splitext(fname)
                if ext not in IMG_EXTS:
                    continue

                key = stem.lower()  # 확장자 제거 + 소문자 통일

                # 동일 stem이 여러 번 나와도 첫 번째만 사용
                if key in id_to_path:
                    continue

                id_to_path[key] = fpath

    print(f"[bbox_crop] id_to_path 매핑 완료: 총 {len(id_to_path)}개 이미지")
    return id_to_path


# 모듈 로드 시 한 번만 생성
ID_TO_PATH: Dict[str, Path] = build_id_to_path_map(data_root)


# ==========================================
# 2) CSV 기준으로 bbox crop
# ==========================================
def crop_split(split: str):
    """
    split: 'train' or 'test'
    PROCESSED_DIR/{split}.csv 를 읽어서
    각 행의 image, disease, bbox 좌표를 기준으로
    crop_processed_data/lettuce_v1/{split}/{label}/ 아래에 잘라서 저장
    """
    csv_path = PROCESSED_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} 없음")

    df = pd.read_csv(csv_path)
    out_root = CROP_ROOT / split
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split}] bbox crop 시작...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = str(row[IMG_COL])  # 예: V006_..._S01_1.JPG, .jpeg 등등

        # 🔥 확장자 제거 + 소문자 → 매핑 키
        stem, _ = os.path.splitext(img_name)
        stem = stem.lower()

        img_path = ID_TO_PATH.get(stem)
        if img_path is None:
            print(f"[경고] 원본 이미지 매칭 실패: {img_name} (stem={stem})")
            continue

        label = str(row[LABEL_COL])

        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[경고] 이미지 불러오기 실패: {img_path} ({e})")
            continue

        try:
            x1, y1, x2, y2 = map(int, [row[c] for c in BBOX_COLS])
        except Exception as e:
            print(f"[경고] bbox 좌표 파싱 실패: {row[BBOX_COLS]} ({e})")
            continue

        cropped = img.crop((x1, y1, x2, y2))

        # 원본 파일명 유지
        save_name = Path(img_path).name
        save_path = out_dir / save_name

        try:
            cropped.save(save_path)
        except Exception as e:
            print(f"[경고] 저장 실패: {save_path} ({e})")
            continue

    print(f"[{split}] 완료 → {out_root}")


if __name__ == "__main__":
    for sp in ["train", "test"]:
        crop_split(sp)
    print("\n✅ 모든 bbox crop 완료!")
