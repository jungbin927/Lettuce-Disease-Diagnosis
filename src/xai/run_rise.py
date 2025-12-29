# src/xai/run_rise_grid.py

import os
import glob
import argparse
import torch

from src.xai.rise import RISE, rise_single
from src.xai.grad_cam import save_grid
from src.xai.run_gradcam_grid import (
    get_true_label_from_name, load_model,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="resseattn",
        choices=["resseattn", "resnet"],
        help="사용할 모델 타입 (resseattn 또는 resnet)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 기존 load_model 재사용 (target_layer는 필요 없으니 버려도 됨)
    model, _ = load_model(args.model_type, device)

    # 2) RISE 객체 생성
    #    input_size=(224,224)는 get_transforms()["test"] 기준으로 맞춰줌
    rise = RISE(
        model=model,
        n_masks=4000,
        p1=0.5,
        input_size=(224, 224),
        initial_mask_size=(7, 7),
        n_batch=64,
        device=device,
        mask_path=None,   # 원하면 "runs_lettuce/rise_masks.pt" 같은 경로로 저장/재사용 가능
    )

    # 3) 경로 설정 (Grad-CAM과 동일한 입력 경로)
    xai_dir = r"crop_processed_data\xai"
    out_dir = os.path.join(
        r"result\xai_rise",
        "ResSEAttnCNN" if args.model_type == "resseattn" else "ResNet",
    )
    os.makedirs(out_dir, exist_ok=True)

    # jpg, jpeg, PNG 등 모두 포함
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        img_paths.extend(glob.glob(os.path.join(xai_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"❌ {xai_dir} 에서 이미지를 찾지 못했습니다.")
        return

    print(f"[{args.model_type}] 총 {len(img_paths)}장의 이미지를 RISE로 XAI 시각화합니다.")

    # 모델 출력 인덱스 → 클래스 이름 매핑
    class_names = ["정상", "상추노균병", "상추균핵병"]

    font_path = r"C:\Windows\Fonts\malgun.ttf"

    # 4) 한 장씩 RISE 생성 + side-by-side 저장
    for img_path in img_paths:
        fname = os.path.basename(img_path)
        true_label = get_true_label_from_name(fname)

        vis_img_np, cam_img, pred_label = rise_single(
            rise=rise,
            img_path=img_path,
            class_names=class_names,
            use_cuda=device.startswith("cuda"),
        )

        stem, _ = os.path.splitext(fname)
        save_path = os.path.join(out_dir, f"{stem}_rise.jpg")

        save_grid(
            orig_img=vis_img_np,
            cam_img=cam_img,
            save_path=save_path,
            pred_label=pred_label,
            true_label=true_label,
            font_path=font_path,
            font_size=20,
        )

        print(f"✅ Saved: {save_path}")

    print("\n🎉 모든 RISE XAI 시각화 저장 완료")


if __name__ == "__main__":
    main()
