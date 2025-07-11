# 이미지 분류 경진대회
## Team : CV-Team9

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [홍정민](https://github.com/UpstageAILab)             |            [최지희](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [김효석](https://github.com/UpstageAILab)             |
|                            팀장, 데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |

## 0. Overview
### Environment
- OS: Ubuntu 20.04
- Python: 3.10
- GPU (Upstage Server)
- Library:
    PyTorch 2.0
    albumentations
    timm
    pandas, scikit-learn, matplotlib

### Requirements
- requirements.txt

## 1. Competiton Info

### Overview

- Document Type Classification | 문서 타입 분류
- 문서 이미지를 분류하여 17개 카테고리 중 하나로 분류하는 이미지 분류 모델 개발
- 총 17개 문서 클래스

### Timeline

 - 2025.06.30 ~ 2025.07.10

## 2. Components

### Directory

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- 학습 이미지 수: 1,570장
- 테스트 이미지 수: 3,140장
- 메타 정보: meta.csv로 클래스 라벨 확인

### EDA
<img width="776" height="488" alt="스크린샷 2025-07-11 오후 1 14 40" src="https://github.com/user-attachments/assets/11408ac4-8bdf-4df8-a30b-2d4b73c02a52" />
<img width="589" height="594" alt="스크린샷 2025-07-11 오후 1 14 56" src="https://github.com/user-attachments/assets/eeb92fe7-109f-4b82-a070-d6dd7d8ca2a0" />
<img width="811" height="824" alt="스크린샷 2025-07-11 오후 1 15 15" src="https://github.com/user-attachments/assets/72c77a46-791a-4a45-9499-fece77a5adf1" />

- 클래스 불균형 존재
- 테스트 이미지는 흐릿하거나 회전됨. 증강 처리 필요
- 학습 이미지 수가 적은 편.

### Data Processing

- Resize
- Normalize
- Augmentation : rotate, flip, zoom, GaussNoise, GaussianBlur, RandomBrightnessContrast, sharpen, CLAHE, coarse dropout 등 37종 적용
  

## 4. Modeling

### Model descrition
<img width="949" height="680" alt="스크린샷 2025-07-11 오후 1 16 16" src="https://github.com/user-attachments/assets/42bcd31d-2af2-4abb-b62b-86af963c477f" />
<img width="949" height="685" alt="스크린샷 2025-07-11 오후 1 16 51" src="https://github.com/user-attachments/assets/d56da7fd-b99e-4f6e-8cb4-f972387af1a0" />

- ConvNeXt Base

### Modeling Process

- 5-Fold Stratified KFold

- Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-6)

- Mixup + CutMix
  
- Loss: Label Smoothing CrossEntropy (eps=0.1)
  
- TTA (sharpen, CLAHE)

## 5. Result
<img width="942" height="803" alt="스크린샷 2025-07-11 오후 1 19 33" src="https://github.com/user-attachments/assets/958682a0-3a61-43e8-b5e0-d7b64bd78fef" />

### Leader Board




### Presentation

- [Link] (https://docs.google.com/presentation/d/1eVMUZeacCLKNIXcOJ3W-VZHJXaa8LcjP/edit?slide=id.g36dabd66d17_2_6#slide=id.g36dabd66d17_2_6)


