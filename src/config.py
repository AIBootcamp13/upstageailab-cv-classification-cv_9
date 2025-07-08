import torch
import os

# --- 프로젝트 루트 경로 설정 ---
# 이 config.py 파일의 위치를 기준으로 프로젝트 루트를 동적으로 찾습니다.
# 현재 파일 경로: /path/to/project/src/config.py
# 프로젝트 루트: /path/to/project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # /path/to/project/src
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT) # /path/to/project

class CFG:
    # --- 기본 경로 ---
    PROJECT_ROOT_DIR = PROJECT_ROOT

    # --- 데이터 경로 ---
    # generate_augmentations.py로 생성한 증강 데이터셋 경로를 사용합니다.
    DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
    SRC_DIR = os.path.join(PROJECT_ROOT_DIR, 'src')
    
    TRAIN_CSV_PATH = os.path.join(SRC_DIR, 'data_aug', 'train_aug.csv')
    META_CSV_PATH = os.path.join(DATA_DIR, 'meta.csv')
    TRAIN_IMG_PATH = os.path.join(SRC_DIR, 'data_aug')
    TEST_IMG_PATH = os.path.join(DATA_DIR, 'test')
    SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

    # --- 모델 저장 경로 ---
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')

    # --- 모델 및 학습 파라미터 ---
    # train.py 실행 시 --model 인자로 다른 모델을 지정할 수 있습니다.
    # 예: 'convnext_tiny', 'resnet50', 'vit_base_patch16_224' 등
    # 인자를 주지 않을 경우 아래 모델이 기본값으로 사용됩니다.
    # ConvNeXt-Large 모델을 위한 추천 파라미터입니다.
    DEFAULT_MODEL = 'convnext_large.fb_in22k_ft_in1k_384'
    IMG_SIZE = 384
    # 대형 모델은 메모리 사용량이 크므로 배치 크기를 줄여야 합니다. (GPU 사양에 따라 4 또는 2로 더 줄여야 할 수 있습니다.)
    BATCH_SIZE = 8
    EPOCHS = 15 # 대형 모델은 더 많은 학습이 필요할 수 있습니다. Early Stopping으로 최적점을 찾습니다.
    LEARNING_RATE = 1e-5 # 대형 모델 파인튜닝 시에는 더 작은 학습률이 효과적입니다.
    PATIENCE = 3 # Early stopping을 위한 patience 값 (3 에포크 동안 개선이 없으면 중단)
    NUM_CLASSES = 17
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")