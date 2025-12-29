import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.datasets.transform import get_transforms

def grad_cam_single(model, img_path, target_layer, class_names, use_cuda=True):
    """
    한 장의 이미지에 대해:
      - vis_img_np: 원본 (0~1 float, HWC, RGB)
      - cam_image : Grad-CAM overlay (0~255 uint8, HWC, RGB)
      - pred_label_name: 예측 클래스 이름 (class_names에서 뽑음)
    를 리턴
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    model.to(device)


    # 1)  transform dict 그대로 가져오기
    tf_dict = get_transforms()         
    test_tf = tf_dict["test"]        

    # 2) 원본 이미지 로드 (PIL)
    pil_img = Image.open(img_path).convert("RGB")

    # 시각화용 이미지 Normalize 제외
    vis_tf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
    ])
    vis_img = vis_tf(pil_img)          # PIL 이미지
    vis_img_np = np.float32(vis_img) / 255.0  # [0,1] 범위로 변환 (H, W, C)

    # 모델 입력용 텐서
    input_tensor = test_tf(pil_img).unsqueeze(0).to(device)  # [1, C, H, W]

    # 3) 예측 라벨
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
    pred_label_name = class_names[pred_idx]
    
    # 4) Grad-CAM 객체
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)

    # 5) overlay 이미지 생성y
    cam_image = show_cam_on_image(vis_img_np, grayscale_cam, use_rgb=True)
    cam_image = cam_image.astype(np.uint8)
    
    return vis_img_np, cam_image, pred_label_name


def save_grid(orig_img, cam_img, save_path,
                      pred_label, true_label,
                      margin=20, top_margin=40, 
                      font_path=None, font_size=20):
    """
    orig_img, cam_img: HWC, RGB
      - orig_img: 보통 0~1 float
      - cam_img : 보통 0~255 uint8
    왼쪽: 원본, 오른쪽: CAM, 위에 Pred/True 텍스트를 넣어 저장.
    """
    # 1) numpy → uint8 RGB로 정리
    if orig_img.dtype != np.uint8:
        orig = (np.clip(orig_img, 0, 1) * 255).astype(np.uint8)
    else:
        orig = orig_img.copy()

    if cam_img.dtype != np.uint8:
        cam = (np.clip(cam_img, 0, 1) * 255).astype(np.uint8)
    else:
        cam = cam_img.copy()

    h, w, _ = orig.shape

    # 전체 캔버스 크기
    canvas_h = h + top_margin
    canvas_w = w * 2 + margin

    # 2) 흰 배경 캔버스 생성 (PIL Image)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    orig_pil = Image.fromarray(orig)
    cam_pil = Image.fromarray(cam)

    # 3) 이미지 붙이기
    canvas.paste(orig_pil, (0, top_margin)) 
    canvas.paste(cam_pil, (w + margin, top_margin)) 
    
    # 4) 폰트 설정
    draw = ImageDraw.Draw(canvas)

    if font_path is not None:
        # 예: C:/Windows/Fonts/malgun.ttf
        font = ImageFont.truetype(font_path, font_size)
    else:
        # 폰트 경로를 못 줬을 때는 기본 폰트 (한글은 깨질 수 있음)
        font = ImageFont.load_default()

    text = f"Pred: {pred_label}  /  True: {true_label}"

    # 5) 텍스트 중앙 정렬 위치 계산
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (canvas_w - text_w) // 2
    text_y = (top_margin - text_h) // 2

    # 6) 텍스트 그리기 (검정색)
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    # 7) 저장
    canvas.save(save_path)