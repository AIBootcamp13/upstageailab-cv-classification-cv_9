import torch

class CFG:
    # --- 데이터 경로 ---
    # generate_augmentations.py로 생성한 증강 데이터셋 경로를 사용합니다.
    TRAIN_CSV_PATH = 'src/data_aug/train_aug.csv'
    META_CSV_PATH = 'data/meta.csv'
    TRAIN_IMG_PATH = 'src/data_aug'
    TEST_IMG_PATH = 'data/test'
    SAMPLE_SUBMISSION_PATH = 'data/sample_submission.csv'

    # --- 모델 및 학습 파라미터 ---
    # train.py 실행 시 --model 인자로 다른 모델을 지정할 수 있습니다.
    # 예: 'tf_efficientnetv2_s_in21k', 'resnet50', 'vit_base_patch16_224' 등
    # 인자를 주지 않을 경우 아래 모델이 기본값으로 사용됩니다.
    DEFAULT_MODEL = 'tf_efficientnetv2_s_in21k'
    IMG_SIZE = 224
    # 모델이 커지면 메모리 부족 오류가 발생할 수 있습니다.
    # 그럴 경우 배치 사이즈를 16이나 8로 줄여보세요.
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 17
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")