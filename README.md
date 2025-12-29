# Lettuce Disease Diagnosis Project

### *AI 기반 상추 병해 진단 시스템 (Custom CNN / ResNet18 / XAI / Streamlit Demo)*

## 🚀 Overview

본 프로젝트는 **상추 잎 이미지로 병해 여부를 자동 판별하는 AI 기반 진단시스템**을 구축하는 것을 목표로 합니다.
단순히 사전학습 모델을 적용하는 데 그치지 않고,  
**상추 병반의 시각적 특성을 반영한 Custom CNN 아키텍처를 직접 설계·구현**하고,  
이를 Pretrained ResNet18과 **성능 및 해석 관점에서 비교 분석**하였습니다.

또한 모델의 예측 결과를 단순한 분류 결과로 끝내지 않고,  
**Grad-CAM 기반 설명 가능한 AI(XAI)**를 적용하여  
모델이 어떤 시각적 근거를 바탕으로 판단하는지 검증하였습니다.  
마지막으로 Streamlit을 활용해 **웹 기반 데모 시스템**을 구현하였습니다.

전처리 → CNN/ResNet 모델 학습 → 성능 개선 → 설명 가능한 AI(Grad-CAM) →
Streamlit 데모 구축까지\
**End-to-End 파이프라인을 직접 구현**하였습니다.

## 🎯 Key Contributions

- 상추 병해 진단을 위한 **Custom CNN 아키텍처 직접 설계**
- Pretrained ResNet18과의 성능·해석 비교 실험
- 클래스 불균형 문제 대응을 통한 모델 안정화
- Grad-CAM을 활용한 모델 판단 근거 시각화
- Streamlit 기반 End-to-End 데모 시스템 구축

---

## 🖼 Dataset

본 프로젝트에서는 **직접 전처리한 상추 잎 이미지 데이터셋**을 사용하였습니다.  
이미지는 다음과 같이 **3개 클래스**로 구성됩니다.

-   `0` 정상 (Normal)\
-   `9` 질병A\
-   `10` 질병B


### 3️⃣ 성능 개선 기법

  Technique               효과
  ----------------------- -------------------------------
  **Class Weighting**     불균형 클래스 F1-score 상승
  **Augmentation 강화**   질병 이미지 일반화 향상
  **Oversampling**        minority class recall 개선

## 📊 Experimental Results

-   Accuracy\
-   Precision\
-   Recall\
-   F1-score (핵심)

Confusion Matrix는 heatmap 형태로 자동 저장됩니다.

## 🔍 Explainability --- Grad-CAM

모델이 어떤 부분을 보고 판단했는지 확인하기 위해\
**Grad-CAM을 적용해 Class Activation Map을 생성했습니다.**

## 🌐 Streamlit Web Demo

웹에서 이미지를 업로드하면:

1.  모델 예측 출력\
2.  확률(score) 표시\
3.  Grad-CAM 시각화\
4.  결과 UI 제공



