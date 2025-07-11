# 이미지 분류 경진대회
## Team : CV-Team9

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [홍정민](https://github.com/UpstageAILab)             |            [최지희](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [김효석](https://github.com/UpstageAILab)             |
|                            팀장, 데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            담당 역할                             |

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

2025. 06. 30 ~ 2025. 07.10

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

- 클래스 불균형 존재
- 테스트 이미지는 흐릿하거나 회전됨. 증강 처리 필요
- 학습 이미지 수가 적은 편.

### Data Processing

- Resize
- Normalize
- Augmentation : rotate, flip, zoom, GaussNoise, GaussianBlur, RandomBrightnessContrast, sharpen, CLAHE, coarse dropout 등 37종 적용
  

## 4. Modeling

### Model descrition

- ConvNeXt Base

### Modeling Process

- 5-Fold Stratified KFold

- Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-6)

- Mixup + CutMix
  
- Loss: Label Smoothing CrossEntropy (eps=0.1)
  
- TTA (sharpen, CLAHE)

## 5. Result

### Leader Board

<img width="817" height="647" alt="image" src="https://github.com/user-attachments/assets/4dc21f92-3d62-4f2f-817e-54100a7283e2" />


### Presentation

- [Link] (https://docs.google.com/presentation/d/1eVMUZeacCLKNIXcOJ3W-VZHJXaa8LcjP/edit?slide=id.g36dabd66d17_2_6#slide=id.g36dabd66d17_2_6)


