# 이미지 분류 경진대회
## Team : CV-Team9

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [홍정민](https://github.com/UpstageAILab)             |            [최지희](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [김효석](https://github.com/UpstageAILab)             |
|      용

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


