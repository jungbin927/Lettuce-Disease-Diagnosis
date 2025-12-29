# streamlit.py
import io
import pandas as pd
import numpy as np
import torch
import streamlit as st
from pathlib import Path
from PIL import Image
import altair as alt

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="상추 병해 진단 서비스",
    page_icon="🥬",
    layout="wide",
)

from src.models.ResSEAttnCNN import LettuceResSEAttnCNN
from src.datasets.transform import get_transforms
from src.xai.grad_cam import grad_cam_single 
from disease_info import DISEASE_INFO, get_confidence_comment
 

CLASS_NAMES = ["정상", "상추노균병", "상추균핵병"]

# -----------------------------
# 1) 모델 로드 함수
# -----------------------------
@st.cache_resource
def load_model(device: str = "cpu"):
    model = LettuceResSEAttnCNN(num_classes=3)
    ckpt_path = "runs_lettuce/LETTUCE_bs256_lr0.001_20251115-133801/best_ep020.pt"

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

# ======================================================
# 2) 예측 확률 계산 함수
# ======================================================
def predict_probs(model, pil_img, device):
    transforms_obj = get_transforms()  # 인자 없이 전체 가져온다고 가정

    # get_transforms가 dict를 리턴하는 경우 처리
    if isinstance(transforms_obj, dict):
        # test, val, valid 중에서 있으면 하나 골라서 사용
        transform = (
            transforms_obj.get("test")
            or transforms_obj.get("val")
            or transforms_obj.get("valid")
            or list(transforms_obj.values())[0]  # 그래도 없으면 첫번째 것
        )
    else:
        # dict가 아니면 그대로 사용
        transform = transforms_obj

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return probs

# -----------------------------
# 3) 메인 UI
# -----------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # =============================
    # 페이지 헤더 (HTML)
    # =============================
    st.markdown(
        """
        <style>
        .title-center {
            text-align: center; 
            font-size: 40px;
        }
        .subheader-center {
            text-align: center; 
            font-size: 18px; 
            color: #555555;
            margin-top: -10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="title-center">🥬 상추 병해 진단 서비스</h1>', unsafe_allow_html=True)
    st.markdown(
    """
    <p class="subheader-center">
    상추 잎 사진을 업로드하면, 
    <b>모델이 추정한 병해 유형</b>과 
    <b>Grad-CAM 기반의 의심 영역</b>을 시각적으로 보여주는 데모 서비스입니다.
    </p>
    """,
    unsafe_allow_html=True,
)
    st.write("")

    # =============================
    # 모델 로드
    # =============================
    model = load_model(device)

    # =============================
    # 이미지 업로드
    # =============================
    uploaded = st.file_uploader("상추 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        st.image(pil_img, caption="업로드된 이미지", width=200)

        tmp_path = "tmp_upload.jpg"
        pil_img.save(tmp_path)

        # Grad-CAM 레이어
        target_layer = model.stage4_block2.conv2

        # =============================
        # 버튼 클릭 → 분석 실행
        # =============================
        if st.button("예측 & 병해 의심 영역 확인하기"):
            
            # ⏳ 로딩 메시지 표시
            with st.spinner("🥬 상추 이미지를 분석 중입니다..."):
                
                # 1) 확률 예측
                probs = predict_probs(model, pil_img, device)
                pred_idx = int(np.argmax(probs))
                pred_label = CLASS_NAMES[pred_idx]
                pred_conf = probs[pred_idx]

                # 2) Grad-CAM 생성
                vis_img_np, cam_img, _ = grad_cam_single(
                    model=model,
                    img_path=tmp_path,
                    target_layer=target_layer,
                    class_names=CLASS_NAMES,
                    use_cuda=(device == "cuda"),
                )

            # =============================
            # 결과 표시
            # =============================
            st.markdown(f"### 🔍 예측 결과: **{pred_label}**")

            left_col, right_col = st.columns([3, 2])

            # ---- 왼쪽: 원본 + Grad-CAM ----
            with left_col:
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(
                        vis_img_np,
                        caption="입력 이미지",
                        width=260,
                    )
                with img_col2:
                    st.image(
                        cam_img,
                        caption="Grad-CAM 병해 의심 영역",
                        width=260,
                    )

            # ---- 오른쪽: 클래스별 확률 막대그래프 ----
            with right_col:
                st.markdown("### 📊 클래스별 예측 확률")

                prob_df = pd.DataFrame({
                    "클래스": CLASS_NAMES,
                    "확률": probs,
                })
                prob_df["확률_퍼센트"] = prob_df["확률"] * 100

                base = (
                    alt.Chart(prob_df)
                    .encode(
                        x=alt.X("클래스:N", axis=alt.Axis(title=None)),
                        y=alt.Y(
                            "확률_퍼센트:Q",
                            axis=alt.Axis(title="확률 (%)"),
                            scale=alt.Scale(domain=[0, 100]),
                        ),
                    )
                )

                # 막대: 클래스별 색깔 다르게
                bars = base.mark_bar().encode(
                    color=alt.Color("클래스:N", legend=None)
                )

                # 막대 위에 숫자 표시
                text = base.mark_text(
                    dy=-8,  # 막대 위로 살짝 올리기
                    fontSize=12,
                ).encode(
                    text=alt.Text("확률_퍼센트:Q", format=".1f")
                )

                chart = (bars + text).properties(height=230)

                st.altair_chart(chart, use_container_width=True)
                st.caption("※ 각 클래스별 softmax 예측 확률(%)입니다.")
                
            info = DISEASE_INFO[pred_label]
            comment = get_confidence_comment(pred_conf)

            st.markdown(
                f"""
                #### 🔍 해당 상추는 **{pred_conf * 100:.2f}%** 확률로  **‘{pred_label}’** 병해로 의심됩니다.

                ##### 🧪 질병 설명
                """,
                unsafe_allow_html=True
            )

            # 2) 설명은 HTML 줄바꿈을 사용하여 별도로 출력
            st.markdown(info['설명'], unsafe_allow_html=True)

            # 3) 대처법 제목
            st.markdown("##### 🛠 추천 대처 방법")

            # 4) 대처법 본문 출력
            st.markdown(info['대처법'], unsafe_allow_html=True)

# ======================================================
if __name__ == "__main__":
    main()
