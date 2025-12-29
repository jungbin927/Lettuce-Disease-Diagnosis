# 기존 이미지별 json 파일을 csv 형태로 변경


import os
import pandas as pd
import numpy as np
import json

# ---------------------------------------------------------
#  py 파일 기준으로 project_root / original_data 경로 설정
# ---------------------------------------------------------
this_file = os.path.abspath(__file__)
this_dir = os.path.dirname(this_file)                   # .../src/datasets
project_root = os.path.dirname(os.path.dirname(this_dir))  # .../ (프로젝트 루트)
base_dir = os.path.join(project_root, "original_data")

print("this_dir:", this_dir)
print("project_root:", project_root)
print("base_dir:", base_dir)

# ---------------------------------------------------------
# JSON → DataFrame 변환
# ---------------------------------------------------------
rows = []

# train, test 모두 순회
for dataset_type in ["train", "test"]:
    for label_name in ["normal", "disease"]:

        json_dir = os.path.join(base_dir, dataset_type, "label", label_name)
        print("scan:", json_dir, "| exists:", os.path.exists(json_dir))

        if not os.path.exists(json_dir):
            continue

        for fname in os.listdir(json_dir):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(json_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            desc = data.get("description", {})
            ann = data.get("annotations", {})

            # points 여러 개 있을 경우 모두 기록
            points_list = ann.get("points", [])
            if not points_list:
                points_list = [dict(xtl=None, ytl=None, xbr=None, ybr=None)]

            for p in points_list:
                row = {
                    # description
                    "image": desc.get("image"),
                    "date": desc.get("date"),
                    "worker": desc.get("worker"),
                    "height": desc.get("height"),
                    "width": desc.get("width"),
                    "task": desc.get("task"),
                    "type": desc.get("type"),
                    "region": desc.get("region"),

                    # annotations
                    "disease": ann.get("disease"),
                    "crop": ann.get("crop"),
                    "area": ann.get("area"),
                    "grow": ann.get("grow"),
                    "risk": ann.get("risk"),

                    # bounding box
                    "xtl": p.get("xtl"),
                    "ytl": p.get("ytl"),
                    "xbr": p.get("xbr"),
                    "ybr": p.get("ybr"),
                }
                rows.append(row)

print(f"총 {len(rows)}개 데이터 수집 완료")

# ---------------------------------------------------------
# DataFrame 후처리
# ---------------------------------------------------------
df = pd.DataFrame(rows)
print(df.columns)
# disease 2(기타), 7(잡초) 제거
df = df[df["disease"].isin([2, 7]) == False]

# 이미지 이름을 stem(lower)로 통일 (train_test_split과 동일하게)
df["image"] = df["image"].apply(lambda x: os.path.splitext(x)[0].lower())

# ---------------------------------------------------------
# CSV 저장
# ---------------------------------------------------------
save_path = os.path.join(base_dir, "data.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"CSV 저장 완료: {save_path}")
print("CSV shape:", df.shape)
