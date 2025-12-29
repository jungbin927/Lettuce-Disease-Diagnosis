# Lettuce Disease Diagnosis Project

### *AI 기반 상추 병해 진단 시스템 (CNN / ResNet18 / Grad-CAM / Streamlit Demo)*

## 🚀 Overview

본 프로젝트는 **상추 잎 이미지로 병해 여부를 자동 판별하는 AI 기반 진단
시스템**입니다.\
전처리 → CNN/ResNet 모델 학습 → 성능 개선 → 설명 가능한 AI(Grad-CAM) →
Streamlit 데모 구축까지\
**End-to-End 파이프라인을 직접 구현**하였습니다.

프로젝트 주요 목표는 다음과 같습니다:

-   상추 병해 이미지 분류 모델 개발\
-   Pretrained ResNet18 기반 전이학습 적용\
-   CNN 모델 성능 향상을 위한 클래스 불균형 처리 및 데이터 증강\
-   Grad-CAM을 통한 모델의 **판단 근거 시각화(XAI)**\
-   Streamlit 기반 간단한 **Web Service Prototype** 제작

## 🖼 Dataset

데이터는 직접 수집 및 전처리한 상추 잎 이미지이며,\
각 이미지는 다음과 같이 **3개 클래스**로 구성되어 있습니다:

-   `0` 정상 (Normal)\
-   `9` 질병A\
-   `10` 질병B

### 🔧 폴더 구조

    crop_processed_data/
        lettuce_v1/
            train/
                0/
                9/
                10/
            val/
            test/

## 🧠 Model Architecture

### 1️⃣ Custom CNN (Baseline)

간단한 CNN 구조를 직접 설계하여 baseline 모델로 사용했습니다.

### 2️⃣ ResNet18 (Pretrained)

ImageNet 사전학습된 ResNet18을 로드하여\
❗ "Fully Fine-tuning" or "Head-only Fine-tuning" 선택 가능.

### 3️⃣ 성능 개선 기법

  Technique               효과
  ----------------------- -------------------------------
  **Class Weighting**     불균형 클래스 F1-score 상승
  **Augmentation 강화**   질병 이미지 일반화 향상
  **Oversampling**        minority class recall 개선
  **Threshold Tuning**    precision--recall 밸런스 조절

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

## 🏗 Project Structure

    src/
     ├── datasets/
     │    ├── transform.py
     │    └── custom_dataset.py
     ├── models/
     │    ├── CNN.py
     │    └── RESNET18_pretrained.py
     ├── utils/
     │    └── heatmap.py
     ├── train.py
     └── test.py

## ⚙️ Installation

``` bash
pip install -r requirements.txt
```

## 🏃‍♂️ Training

``` bash
python src/train.py   --data_root crop_processed_data/lettuce_v1   --model resnet18   --epochs 20   --batch_size 256
```

## 🧪 Testing

``` bash
python -m src.test   --model cnn   --data_root crop_processed_data/lettuce_v1/test   --ckpt_path runs_lettuce/best_ep016.pt   --cm_path results/cnn_confusion_matrix.png
```

## 🎯 Conclusion

End-to-End 파이프라인 구축 완료:

-   AI 모델 개발\
-   성능 개선(전이학습 + 불균형 처리)\
-   설명 가능성 확보(XAI)\
-   Streamlit 데모 구현
