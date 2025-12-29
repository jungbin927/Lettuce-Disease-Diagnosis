# src/xai/rise.py

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from PIL import Image
from torchvision import transforms as T

from src.datasets.transform import get_transforms


class RISE(nn.Module):
    """
    RISE: Randomized Input Sampling for Explanation.

    - model: 학습된 분류 모델 (nn.Module)
    - n_masks: 생성할 랜덤 마스크 개수 (논문은 4000 정도 사용)
    - p1: 마스크에서 1이 나올 확률 (1이 클수록 더 많이 남기는 마스크)
    - input_size: (H, W) 모델 입력 크기 (예: 224x224)
    - initial_mask_size: (h, w) 저해상도 마스크 크기 (예: 7x7)
    - n_batch: 마스크를 몇 개씩 묶어서 모델에 넣을지 (메모리 조절용)
    """

    def __init__(
        self,
        model: nn.Module,
        n_masks: int = 4000,
        p1: float = 0.5,
        input_size: Tuple[int, int] = (224, 224),
        initial_mask_size: Tuple[int, int] = (7, 7),
        n_batch: int = 64,
        mask_path: Optional[str] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()

        self.n_masks = n_masks
        self.p1 = p1
        self.input_size = input_size
        self.initial_mask_size = initial_mask_size
        self.n_batch = n_batch
        self.device = device

        if mask_path is not None and os.path.exists(mask_path):
            self.masks = self.load_masks(mask_path)
        else:
            self.masks = self.generate_masks()
            if mask_path is not None:
                self.save_masks(mask_path)

    def generate_masks(self) -> torch.Tensor:
        """
        low-res 바이너리 마스크를 만든 뒤,
        bilinear upsample + random crop으로 (H,W) 크기의 마스크 생성.
        리턴 shape: (n_masks, 1, H, W)
        """
        H, W = self.input_size
        h0, w0 = self.initial_mask_size

        # upsample 전에 한 cell이 차지할 크기
        Ch = np.ceil(H / h0)
        Cw = np.ceil(W / w0)

        resize_h = int((h0 + 1) * Ch)
        resize_w = int((w0 + 1) * Cw)

        masks_list: List[torch.Tensor] = []

        for _ in range(self.n_masks):
            # p1 확률로 1인 저해상도 마스크 생성
            binary_mask = torch.rand(1, 1, h0, w0)
            binary_mask = (binary_mask < self.p1).float()  # (1,1,h0,w0)

            # bilinear upsample → (1,1,resize_h, resize_w)
            mask = F.interpolate(
                binary_mask,
                size=(resize_h, resize_w),
                mode="bilinear",
                align_corners=False,
            )

            # random crop → (1,1,H,W)
            i = np.random.randint(0, resize_h - H + 1)
            j = np.random.randint(0, resize_w - W + 1)
            mask = mask[:, :, i : i + H, j : j + W]

            masks_list.append(mask)

        masks = torch.cat(masks_list, dim=0)  # (N,1,H,W)
        return masks  # CPU에 둠, forward에서 device로 옮김

    @staticmethod
    def load_masks(filepath: str) -> torch.Tensor:
        return torch.load(filepath)

    def save_masks(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.masks, filepath)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        단일 이미지에 대해 RISE saliency map 계산.
        x: (1, 3, H, W)
        return: saliency, shape = (n_classes, H, W), 값 범위 [0,1]
        """
        self.model.eval()
        device = x.device

        masks = self.masks.to(device)  # (N,1,H,W)
        N, _, H, W = masks.shape

        probs_list: List[torch.Tensor] = []

        # 마스크를 여러 배치로 나눠서 처리 (메모리 절약)
        for i in range(0, N, self.n_batch):
            m = masks[i : i + self.n_batch]  # (b,1,H,W)
            b = m.size(0)

            # broadcasting: (1,3,H,W) * (b,1,H,W) -> (b,3,H,W)
            x_masked = x * m

            out = self.model(x_masked)           # (b, num_classes)
            probs = F.softmax(out, dim=1)        # (b, num_classes)
            probs_list.append(probs)

        probs_all = torch.cat(probs_list, dim=0)  # (N, num_classes)
        n_classes = probs_all.shape[1]

        # saliency = Σ (p_c,i * mask_i)
        masks_flat = masks.view(N, -1)                     # (N, H*W)
        saliency = torch.matmul(probs_all.T, masks_flat)   # (num_classes, H*W)
        saliency = saliency.view(n_classes, H, W)          # (C,H,W)

        # 1 / (N * p1) 로 정규화 (논문 공식)
        saliency = saliency / (self.n_masks * self.p1)

        # 0~1로 클래스별 정규화
        saliency = saliency - saliency.view(n_classes, -1).min(dim=1)[0].view(
            n_classes, 1, 1
        )
        saliency = saliency / (
            saliency.view(n_classes, -1).max(dim=1)[0].view(n_classes, 1, 1) + 1e-8
        )

        return saliency  # (num_classes, H, W), [0,1]


# -------------------------
# Grad-CAM과 비슷한 인터페이스의 헬퍼
# -------------------------

def _overlay_heatmap_on_image(
    vis_img_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    vis_img_np: (H,W,3), 0~1 float
    heatmap   : (H,W),   0~1 float
    return    : (H,W,3), 0~255 uint8
    """
    heatmap = np.clip(heatmap, 0, 1)
    hmap_uint8 = np.uint8(255 * heatmap)
    hmap_color = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)  # BGR
    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    hmap_color = np.float32(hmap_color) / 255.0

    vis = np.float32(vis_img_np)
    overlay = alpha * hmap_color + (1 - alpha) * vis
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = np.uint8(overlay * 255)
    return overlay_uint8


def rise_single(rise: RISE, img_path: str, class_names, use_cuda: bool = True):
    """
    Grad-CAM의 grad_cam_single과 같은 형태로 리턴:

      - vis_img_np: 원본 (0~1 float, HWC, RGB)
      - cam_image : RISE overlay (0~255 uint8, HWC, RGB)
      - pred_label_name: 예측 클래스 이름

    차이점은 target_layer 대신 RISE 객체를 받는 것 뿐.
    """
    model = rise.model
    model.eval()
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    model.to(device)

    # 1) transform dict 그대로 사용
    tf_dict = get_transforms()
    test_tf = tf_dict["test"]

    # 2) 원본 이미지 로드 (PIL)
    pil_img = Image.open(img_path).convert("RGB")

    # 시각화용 (Grad-CAM이랑 동일)
    vis_tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
    ])
    vis_img = vis_tf(pil_img)                # PIL
    vis_img_np = np.float32(vis_img) / 255.0 # (H,W,C), 0~1

    # 모델 입력용 텐서
    input_tensor = test_tf(pil_img).unsqueeze(0).to(device)  # [1, C, H, W]

    # 3) 예측 라벨
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
    pred_label_name = class_names[pred_idx]

    # 4) RISE saliency 계산
    saliency_all = rise(input_tensor)                    # (num_classes, H, W)
    heatmap = saliency_all[pred_idx].detach().cpu().numpy()  # (H,W), 0~1

    # 5) overlay 이미지 생성 (Grad-CAM 스타일)
    cam_image = _overlay_heatmap_on_image(vis_img_np, heatmap, alpha=0.4)

    return vis_img_np, cam_image, pred_label_name