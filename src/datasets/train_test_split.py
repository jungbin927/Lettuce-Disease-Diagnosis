# json을 변환한 data.csv를 이용해 train/test를 클래스별 비율에 맞게 8:2 분할

import os
import shutil
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

print("현재 작업 경로(CWD):", os.getcwd())
print("스크립트 위치:", os.path.abspath(__file__))


# original_data/train/test/picture/normal,disease 전체를 스캔해서
# 파일명 stem(확장자 제거, 소문자) → 실제 이미지 경로 로 매핑
def build_id_to_path_map(data_root: str) -> Dict[str, str]:
    img_dirs = [
        os.path.join(data_root, "train", "picture", "normal"),
        os.path.join(data_root, "train", "picture", "disease"),
        os.path.join(data_root, "test", "picture", "normal"),
        os.path.join(data_root, "test", "picture", "disease"),
    ]

    # 사용할 이미지 확장자 (대소문자 모두 처리)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    id_to_path: Dict[str, str] = {}

    for root_dir in img_dirs:
        if not os.path.isdir(root_dir):
            print(f"[경고] 폴더 없음, 건너뜀: {root_dir}")
            continue

        # 🔥 하위 batch 폴더까지 모두 탐색
        for cur_dir, _, files in os.walk(root_dir):
            for fname in files:
                fpath = os.path.join(cur_dir, fname)
                if not os.path.isfile(fpath):
                    continue

                stem, ext = os.path.splitext(fname)
                if ext not in exts:
                    continue

                key = stem.lower()  # 🔥 확장자 제거 + 소문자로 통일

                # 동일한 stem이 여러 번 나와도 첫 번째만 사용
                if key in id_to_path:
                    continue

                id_to_path[key] = fpath

    print(f"id_to_path 매핑 완료: 총 {len(id_to_path)}개 이미지")
    return id_to_path


# CSV 읽고, 라벨별 폴더 구조로 이미지를 배치하는 함수
def build_processed_from_csv(
    csv_path: str,           # image, disease 컬럼이 있는 CSV 경로
    dst_split_dir: str,      # 결과물을 넣을 클래스별 폴더의 루트 (예: processed/train)
    id_to_path: Dict[str, str],  # stem → 원본 이미지 실제 경로
    use_hardlink: bool = True,
) -> Tuple[int, int]:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다.: {csv_path}")

    os.makedirs(dst_split_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 필요한 image, disease 컬럼이 없으면 에러
    if not {"image", "disease"}.issubset(df.columns):
        raise ValueError(f"CSV에는 'image', 'disease' 컬럼이 필요합니다. {csv_path}")

    # label 고유값 리스트
    labels = df["disease"].astype(int).unique().tolist()

    # CSV에 나타난 라벨 고유값들로 dst_split_dir 하위 폴더 미리 생성
    for y in labels:
        os.makedirs(os.path.join(dst_split_dir, str(int(y))), exist_ok=True)

    created = 0
    missing = 0

    # CSV의 각 행 기준으로 이미지 배치
    for r in df.itertuples(index=False):
        img_name = str(r.image)        # 예: V006_..._S01_1.JPG, .jpeg, .jpg 등
        label = str(int(r.disease))

        #  확장자 제거 + 소문자로 통일해서 매핑 키로 사용
        stem, _ = os.path.splitext(img_name)
        stem = stem.lower()

        if stem not in id_to_path:
            print(f"[경고] 원본 이미지 없음, 건너뜀: {img_name}")
            missing += 1
            continue

        src_path = id_to_path[stem]
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_split_dir, label, fname)

        if os.path.exists(dst_path):
            continue

        if use_hardlink:
            try:
                os.link(src_path, dst_path)  # 하드링크
            except OSError:
                shutil.copy2(src_path, dst_path)  # 안 되면 복사
        else:
            shutil.copy2(src_path, dst_path)

        created += 1

    print(f"{csv_path} 기준 배치 완료: 생성 {created}개, 매칭 실패(누락) {missing}개")
    return len(df), len(labels)


# 전체 CSV를 train/test 로 8:2 분할하는 함수
def split_train_test_csv(
    full_csv: str,
    train_csv: str,
    test_csv: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, int]:

    if not os.path.exists(full_csv):
        raise FileNotFoundError(f"전체 CSV 파일이 없습니다.: {full_csv}")

    df = pd.read_csv(full_csv)

    if not {"image", "disease"}.issubset(df.columns):
        raise ValueError(f"CSV에는 'image', 'disease' 컬럼이 필요합니다. {full_csv}")

    # stratify 
    vc = df["disease"].value_counts()
    min_count = vc.min()
    use_stratify = min_count >= 2

    if not use_stratify:
        print(f"[경고] 일부 클래스 샘플 수가 2개 미만(min={min_count}) → stratify 없이 분할합니다.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["disease"] if use_stratify else None,   # 클래스 비율 유지
        random_state=random_state,
    )

    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"CSV split 완료 → train: {len(train_df)}, test: {len(test_df)}")
    return {"train": len(train_df), "test": len(test_df)}


# split된 CSV를 이용해서 processed/train, processed/test 폴더 만들기
def prepare_processed_from_csv(
    *,
    processed_root: str,   # 가공된 데이터 최상위 폴더 (예: /.../processed/lettuce_v1)
    train_csv: str,        # split된 train CSV
    test_csv: str,         # split된 test CSV
    id_to_path: Dict[str, str],
    use_hardlink: bool = True,
) -> Dict[str, int]:

    os.makedirs(processed_root, exist_ok=True)
    proc_train = os.path.join(processed_root, "train")
    proc_test = os.path.join(processed_root, "test")

    print(f"가공 시작 --> {processed_root}")

    n_train, _ = build_processed_from_csv(
        csv_path=train_csv,
        dst_split_dir=proc_train,
        id_to_path=id_to_path,
        use_hardlink=use_hardlink,
    )

    n_test, _ = build_processed_from_csv(
        csv_path=test_csv,
        dst_split_dir=proc_test,
        id_to_path=id_to_path,
        use_hardlink=use_hardlink,
    )

    print(f"가공 완료 - train: {n_train}, test: {n_test}")
    return {"train": n_train, "test": n_test}


def main():
    # 현재 파일(src/datasets/train_test_split.py) 기준으로 프로젝트 루트 계산
    this_dir = os.path.dirname(os.path.abspath(__file__))           # .../src/datasets
    project_root = os.path.dirname(os.path.dirname(this_dir))       # .../ (프로젝트 루트)

    # original_data 폴더
    data_root = os.path.join(project_root, "original_data")

    # 전체 CSV (image, disease가 들어 있는 파일)
    full_csv = os.path.join(data_root, "data.csv")

    # split 결과 CSV 저장 위치
    # split_dir = os.path.join(data_root, "splits")
    train_csv = os.path.join(project_root, "processed_data", "lettuce_v1", "train.csv")
    test_csv = os.path.join(project_root, "processed_data", "lettuce_v1", "test.csv")

    # 원본 이미지 위치를 전부 스캔해서 stem → 경로 매핑
    id_to_path = build_id_to_path_map(data_root)

    # 최종 processed 폴더 
    processed_root = os.path.join(project_root, "processed_data", "lettuce_v1")

    # 1) CSV 8:2로 분할
    split_train_test_csv(
        full_csv=full_csv,
        train_csv=train_csv,
        test_csv=test_csv,
        test_size=0.2,
        random_state=42,
    )

    # 2) 분할된 CSV 기준으로 새 train/test 폴더 구조 만들기
    prepare_processed_from_csv(
        processed_root=processed_root,
        train_csv=train_csv,
        test_csv=test_csv,
        id_to_path=id_to_path,
        use_hardlink=True,
    )


if __name__ == "__main__":
    main()
