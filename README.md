# Lettuce Disease Diagnosis Project  
AI 기반 상추 병해 진단 시스템  
(Custom CNN 설계 · ResNet18 비교 · XAI · Ablation · Streamlit Demo)

---

## 1. 프로젝트 개요

본 프로젝트는 **상추 잎 이미지를 기반으로 병해 여부를 자동으로 분류하는 AI 기반 병해 진단 시스템**을 구축하는 것을 목표로 합니다.  
단순히 사전학습(pretrained) 모델을 적용하는 수준을 넘어,  
**상추 병반의 시각적 특성을 반영한 Custom CNN 아키텍처를 직접 설계·구현**하고,  
이를 Pretrained ResNet18과 **성능 및 해석 관점에서 비교 분석**하였습니다.

또한 모델의 예측 결과를 단순한 분류 결과로만 제시하지 않고,  
**Grad-CAM 및 RISE 기반 설명 가능한 AI(XAI)** 기법을 적용하여  
모델이 어떤 시각적 근거를 바탕으로 판단했는지를 분석하였습니다.  
마지막으로 Streamlit을 활용해 **웹 기반 데모 서비스**를 구현하였습니다.

본 프로젝트는  
**데이터 전처리 → 모델 설계 → 성능 개선 → XAI 분석 → Ablation Test → 웹 데모 구현**  
까지 포함하는 **End-to-End 딥러닝 비전 파이프라인**을 직접 구현한 프로젝트입니다.

---

## 2. 주요 기여 사항 (Key Contributions)

- 상추 병해 진단을 위한 **Custom CNN 아키텍처 직접 설계**
- Custom CNN과 **Pretrained ResNet18 간 성능 및 표현 특성 비교**
- 클래스 불균형 문제 해결을 위한 성능 개선 실험 수행
- **Grad-CAM, RISE 기반 모델 해석(XAI) 분석**
- 모델 구조 타당성 검증을 위한 **Ablation Test 수행**
- Streamlit 기반 **End-to-End 웹 데모 시스템 구현**

---

## 3. 데이터셋 설명

- **데이터 출처**: AI Hub – 시설 작물 질병 진단 이미지
- **분류 클래스 (3-class)**  
  - `0`: 정상  
  - `9`: 상추 균핵병  
  - `10`: 상추 노균병  

### 데이터 특징
- 이미지별 JSON 메타데이터 제공 (병해 정보, bounding box 등)
- 정상 대비 병해 클래스 비율이 낮은 **심각한 클래스 불균형 구조**
- Bounding-box 정보를 활용하여 **상추 잎 영역 중심으로 Crop 전처리 수행**

---

## 4. 데이터 전처리

다음과 같은 전처리 과정을 수행하였습니다.

1. JSON → CSV 변환을 통한 메타데이터 일괄 관리
2. Bounding-box 기반 Crop으로 불필요한 배경 제거
3. 클래스 비율을 고려한 **Stratified Train / Validation / Test 분할**
4. 이미지 Resize 및 정규화
5. 데이터 증강 (Train 데이터만 적용)
   - Random Horizontal Flip
   - Random Rotation
   - Color Jitter (밝기, 대비, 채도, 색조)

---

## 5. 모델 설계

### 5.1 Baseline CNN

- 4개의 Convolution Block과 Fully Connected Layer로 구성
- 입력 크기: 224 × 224 RGB 이미지
- 병해 분류 문제에 대한 **기초 성능 기준(Baseline)** 확보 목적

---

### 5.2 Custom CNN (핵심 모델)

본 프로젝트의 핵심 모델로,  
상추 병반의 **국소적인 색상·텍스처 변화**를 효과적으로 학습하도록  
CNN 구조를 직접 설계하였습니다.

**주요 구성 요소**
- Residual Block (Skip Connection)
- SE Block (채널 어텐션)
- Spatial Attention (공간 어텐션)
- Adaptive Average Pooling

해당 모델은 프로젝트 전반에서 **주요 분석 대상 모델**로 사용되었습니다.

---

### 5.3 ResNet18 (비교 모델)

- ImageNet 사전학습된 ResNet18 사용
- 동일한 학습 조건에서 **전체 레이어 Fine-tuning**
- Custom CNN과의 **성능 상한선 및 표현 차이 비교 목적**

---

## 6. 성능 개선 기법

클래스 불균형 문제를 해결하기 위해 다음 기법을 적용하였습니다.

| 기법 | 설명 |
|----|----|
| Weighted Loss | 소수 클래스에 더 큰 손실 가중치 부여 |
| Random Oversampling | 소수 클래스 샘플 선택 빈도 증가 |
| Data Augmentation | 질병 이미지 일반화 성능 향상 |

---

## 7. 실험 결과

모델 성능 평가는 다음 지표를 사용하였습니다.

- Accuracy
- Precision
- Recall
- **F1-score (주요 성능 지표)**

| 모델 | Accuracy | Precision | Recall | F1-score |
|----|----|----|----|----|
| Base CNN | 0.931 | 0.889 | 0.827 | 0.854 |
| Base CNN + Weighted Loss | 0.933 | 0.873 | 0.880 | 0.866 |
| Base CNN + Oversampling | 0.941 | 0.877 | 0.895 | 0.879 |
| **Custom CNN + Oversampling** | **0.950** | **0.886** | **0.903** | **0.894** |
| ResNet18 | 0.974 | 0.941 | 0.933 | 0.937 |

Confusion Matrix를 통해 클래스별 오분류 패턴을 분석하였습니다.

---

## 8. 설명 가능한 AI (XAI)

모델의 예측 근거를 해석하기 위해 두 가지 XAI 기법을 적용하였습니다.

### Grad-CAM
- CNN의 마지막 합성곱 계층의 기울기를 활용
- 병반이 위치한 **국소 영역 강조**

### RISE
- 입력 이미지를 무작위 마스크로 가린 후 예측 변화 분석
- 모델 구조와 무관한 전역적 중요도 맵 생성

**분석 결과**
- Custom CNN은 병반 영역에 비교적 **집중적으로 반응**
- ResNet18은 더 넓은 영역을 기반으로 판단하는 경향 확인
- 오분류 사례 분석을 통해 모델 취약점 확인

---

## 9. Ablation Test

Custom CNN 구조의 타당성을 검증하기 위해 Ablation Test를 수행하였습니다.

### 기능 단위 Ablation
- SE Block 제거
- Spatial Attention 제거
- Skip Connection 제거
- Attention 모듈 전체 제거

### 구조 단위 Ablation
- Stage 4 제거 (모델 깊이 감소)

**주요 분석 결과**
- 충분한 모델 깊이가 성능 향상에 중요
- Attention 모듈은 성능 안정성에 기여
- Skip Connection은 데이터 및 깊이에 따라 성능에 상이한 영향 확인

---

## 10. Streamlit 웹 데모

Streamlit을 활용하여 **상추 병해 진단 웹 데모**를 구현하였습니다.

**기능**
1. 상추 이미지 업로드
2. 병해 예측 결과 출력
3. 클래스별 확률 표시
4. Grad-CAM 기반 시각화 결과 제공

---

## 11. 프로젝트 구조

```text
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

