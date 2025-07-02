# 이미지 분류 프로젝트

문서 이미지 분류를 위한 딥러닝 프로젝트입니다.

## 프로젝트 구조

```
CV/
├── data/                   # 데이터 폴더
│   ├── train_aug/          # 학습 이미지 (원본 + 증강)
│   ├── test/               # 테스트 이미지
│   ├── train.csv           # 학습 데이터 메타
│   ├── meta.csv            # 클래스 메타데이터
│   └── train_augmented.csv # 증강된 데이터 메타
├── models/                 # 학습된 모델 저장
├── src/                    # 소스 코드
│   ├── data/               # 데이터 처리
│   │   ├── __init__.py          # 패키지 초기화
│   │   ├── EDA.py               # 탐색적 데이터 분석
│   │   ├── augmentation.py      # 데이터 증강
│   │   └── data_preprocessing.py # 데이터 전처리
│   ├── model/                   # 모델 관련
│   │   ├── __init__.py          # 패키지 초기화
│   │   └── model_architecture.py # 모델 아키텍처
│   ├── train/                   # 학습 관련
│   │   ├── __init__.py          # 패키지 초기화
│   │   ├── training_pipeline.py # 학습 파이프라인
│   │   └── run_training.py      # 학습 실행 스크립트
│   └── inference/               # 추론 관련
│       ├── __init__.py          # 패키지 초기화
│       └── inference.py         # 추론 모듈
└── requirements.txt             # 의존성 패키지
```

## 설치 및 설정

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **데이터 준비**
- `data/train_aug/`: 학습 이미지 파일들
- `data/test/`: 테스트 이미지 파일들
- `data/train_augmented.csv`: 학습 데이터 메타데이터
- `data/meta.csv`: 클래스 정보

## 사용법

### 1. 데이터 증강

```bash
cd src
python data/augmentation.py
```

증강된 이미지는 `data/train_aug/` 폴더에 저장되고, 메타데이터는 `data/train_augmented.csv`에 저장됩니다.

### 2. 모델 학습

#### 2.1 기본 학습
```bash
cd src
python train/run_training.py
```

#### 2.2 다양한 학습 옵션
```bash
# EfficientNet B0 모델로 학습
python train/run_training.py --model efficientnet_b0 --epochs 50

# ResNet50 모델로 학습
python train/run_training.py --model resnet50 --batch-size 16 --epochs 30

# 특정 단계 건너뛰기
python train/run_training.py --skip-eda --skip-evaluation

# 평가만 실행 (학습 완료 후)
python train/run_training.py --evaluation-only --model-path models/final_model.pth

# wandb 로깅과 함께 학습
python train/run_training.py --use-wandb --wandb-project my-project --wandb-run-name experiment-1

# wandb 사용 예시
python train/run_training.py --model efficientnet_b0 --epochs 30 --use-wandb --wandb-project document-classification
```

학습된 모델은 `models/` 폴더에 저장됩니다.

#### 2.3 wandb 설정
wandb를 사용하려면 먼저 로그인해야 합니다:
```bash
pip install wandb
wandb login
```

### 3. 추론 (Inference)

#### 3.1 기본 추론 실행

```bash
cd src
python inference/inference.py
```

이 명령어는 학습된 모델을 사용하여 테스트 데이터셋에 대해 예측을 수행하고 `submission.csv` 파일을 생성합니다.

```python
from src.inference.inference import run_inference

# 추론 실행
run_inference(
    model_path="models/best_model.pth",
    test_csv_path="data/sample_submission.csv",
    test_img_dir="data/test",
    submission_path="submission.csv",
    batch_size=32
)
```

## 주요 기능

### 데이터 증강
- 다양한 증강 기법 (회전, 뒤집기, 밝기 조정, 노이즈 등)
- 클래스별 균형 맞추기
- 문서 특화 증강

### 모델 아키텍처
- EfficientNet 기반
- 전이학습 지원
- 다양한 옵티마이저 및 스케줄러

### 추론 기능
- 테스트 데이터셋 배치 예측
- 모델 아키텍처 자동 감지
- submission.csv 자동 생성
- 예측 결과 통계 출력

## 출력 파일

### 학습 결과
- `models/best_model.pth`: 최고 성능 모델
- `models/training_curves.png`: 학습 곡선
- `models/final_model.pth`: 최종 모델

### 추론 결과
- `submission.csv`: 대회 제출용 예측 결과

## 설정 옵션

### 학습 설정 (`training_pipeline.py`)
```python
config = {
    'model_name': 'efficientnet_b0',
    'num_classes': 17,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-3,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'loss_function': 'cross_entropy',
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    'early_stopping_patience': 10,
    # wandb 설정
    'use_wandb': False,
    'wandb_project': 'image-classification',
    'wandb_run_name': None
}
```

### 추론 설정 (`inference.py`)
```python
# 기본 설정
model_path = "models/best_model.pth"
test_csv_path = "data/sample_submission.csv"
test_img_dir = "data/test"
submission_path = "submission.csv"
batch_size = 32
```

## 문제 해결

### 일반적인 오류

1. **모델 파일 없음**
   - 먼저 모델을 학습하세요: `python run_training.py`

2. **CUDA 메모리 부족**
   - 배치 크기를 줄이세요
   - CPU 모드로 실행하세요

3. **이미지 로드 실패**
   - 이미지 경로가 올바른지 확인하세요
   - 지원되는 형식: jpg, jpeg, png, bmp

4. **wandb 관련 오류**
   - wandb가 설치되어 있는지 확인: `pip install wandb`
   - wandb 로그인이 되어 있는지 확인: `wandb login`
   - 인터넷 연결이 안정적인지 확인

5. **추론 관련 오류**
   - 모델 파일이 존재하는지 확인: `models/best_model.pth`
   - 메타데이터 파일이 존재하는지 확인: `data/meta.csv`
   - 이미지 파일 경로가 올바른지 확인
   - GPU 메모리 부족 시 배치 크기를 줄이세요

### 성능 최적화

1. **GPU 사용**
   - CUDA가 설치된 환경에서 실행하면 자동으로 GPU를 사용합니다

2. **배치 크기 조정**
   - 메모리에 맞게 배치 크기를 조정하세요

3. **이미지 전처리**
   - 이미지 크기는 자동으로 224x224로 조정됩니다
   - 비율 유지 패딩이 적용됩니다

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 