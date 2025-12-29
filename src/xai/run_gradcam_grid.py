# run_gradcam_grid.py

import os
import glob
import argparse
import torch
import numpy as np
import cv2

from src.models.ResSEAttnCNN import LettuceResSEAttnCNN
from src.models.RESNET18_pretrained import ResNet18_Lettuce
from src.xai.grad_cam import grad_cam_single, save_grid

def get_true_label_from_name(filename: str) -> str:
    """
    파일 이름 prefix로 true label 추출
    예) normal1.JPG → "normal"
        disease1_2.jpeg → "disease1"
        disease2_3.jpeg → "disease2"
    """
    fname = filename.lower()
    if fname.startswith("normal"):
        return "정상"
    elif fname.startswith("disease1"):
        return "상추균핵병"
    elif fname.startswith("disease2"):
        return "상추노균병"
    else:
        return "unknown"
def load_model(model_type: str, variant: str, device: str):
    """
    model_type: "resseattn" 또는 "resnet"
    해당하는 모델 + target_layer + ckpt 로드해서 리턴
    """
    if model_type == "resseattn":
        model = LettuceResSEAttnCNN(num_classes=3).to(device)
        
        if variant == "full":
            ckpt_path = r"runs_lettuce\LETTUCE_bs256_lr0.001_20251115-133801\best_ep020.pt"
        elif variant == "no_se":
            ckpt_path = r"runs_lettuce/ablation_test/ABL_se_off_bs128_lr0.001_20251129-144330/best_ep015.pt"
        elif variant == "no_spatial":
            ckpt_path = r"runs_lettuce\ablation_test/ABL_sa_off_bs128_lr0.001_20251129-124645/best_ep020.pt"
        elif variant == "no_skip":
            ckpt_path = r"runs_lettuce/ablation_test/ABL_skip_connection_off_bs128_lr0.001_20251130-045250/best_ep018.pt"
        elif variant == "no_attn":      # SE+Spatial 둘 다 제거 버전
            ckpt_path = r"runs_lettuce/ablation_test/ABL_res_only_bs128_lr0.001_20251201-111058/best_ep015.pt"
        elif variant == "no_stage4":
            ckpt_path = r"runs_lettuce/ablation_test/ABL_reduce_depth_bs128_lr0.001_20251202-094430/best_ep017.pt"
        else:
            raise ValueError(f"지원하지 않는 variant: {variant}")
        
    elif model_type == "resnet":
        model = ResNet18_Lettuce(num_classes=3).to(device)
        ckpt_path = r"runs_lettuce\resnet18_full_bs256_lr0.001_20251114-105019\best_ep020.pt"
    else:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")
    
    ckpt = torch.load(ckpt_path, map_location=device)

    # 체크포인트 구조에 맞게 state_dict만 꺼내기
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    
    if model_type == "resseattn":
        if variant == "no_stage4":
            # Stage4 제거 모델이면 마지막 conv가 Stage3 
            target_layer = model.stage3_block.conv2
        else:
            # 원래 full 모델이랑 나머지 ablaiton(-se, -spatial, -skip 등)
            target_layer = model.stage4_block2.conv2
    else:  # "resnet"
        target_layer = model.layer4[-1].conv2

    return model, target_layer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="resseattn",
        choices=["resseattn", "resnet"],
        help="사용할 모델 타입 (resseattn 또는 resnet)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        help="ablation 버전 이름 (full, no_se, no_spatial, no_skip, no_attn, no_stage4 등)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 모델 + 타깃 레이어 로드
    model, target_layer = load_model(args.model_type, args.variant, device)

    # 2) 경로 설정
    xai_dir = r"crop_processed_data\xai"
    out_dir = os.path.join(r"result\xai", "ResSEAttnCNN" if args.model_type == "resseattn" else "ResNet")
    if args.model_type == "resseattn" and args.variant != "full":
    # ResSEAttn + ablation 버전 → xai_ablation/variant
        out_dir = os.path.join("result", "xai_ablation", args.variant)
    else:
    # resnet18 이거나, resseattn + full → 기존 경로 사용
        model_name = "ResSEAttnCNN" if args.model_type == "resseattn" else "ResNet"
        out_dir = os.path.join("result", "xai", model_name)
    os.makedirs(out_dir, exist_ok=True)

    # jpg, jpeg, JPG 등 모두 포함
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        img_paths.extend(glob.glob(os.path.join(xai_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"❌ {xai_dir} 에서 이미지를 찾지 못했습니다.")
        return

    print(f"[{args.model_type}] 총 {len(img_paths)}장의 이미지를 XAI 시각화합니다.")

    # 모델 출력 인덱스 → 클래스 이름 매핑
    # (학습 시 label 0,1,2 순서에 맞게 작성)
    class_names = ["정상", "상추노균병", "상추균핵병"]

    # 3) 한 장씩 Grad-CAM 생성 + side-by-side 저장
    for img_path in img_paths:
        fname = os.path.basename(img_path)
        true_label = get_true_label_from_name(fname)

        vis_img_np, cam_img, pred_label = grad_cam_single(
            model=model,
            img_path=img_path,
            target_layer=target_layer,
            class_names=class_names,
            use_cuda=device.startswith("cuda"),
        )

        stem, _ = os.path.splitext(fname)
        save_path = os.path.join(out_dir, f"{stem}_xai.jpg")

        font_path = r"C:\Windows\Fonts\malgun.ttf"
        
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

    print("\n🎉 모든 XAI 시각화 저장 완료")


if __name__ == "__main__":
    main()
