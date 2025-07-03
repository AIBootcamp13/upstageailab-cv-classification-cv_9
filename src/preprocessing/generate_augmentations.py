import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm
import random

def generate_augmented_data():
    """
    원본 학습 데이터셋에 다양한 증강 기법을 적용하여
    새로운 데이터셋을 생성하고 별도의 폴더에 저장합니다.
    """
    # --- 경로 설정 (절대 경로 사용) ---
    # 현재 스크립트가 /workspace/ 에 있다고 가정합니다.
    BASE_DIR = "/workspace"
    ORIGINAL_TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
    ORIGINAL_TRAIN_CSV = os.path.join(BASE_DIR, "data/train.csv")
    
    AUG_DIR = os.path.join(BASE_DIR, "src", "data_aug")
    AUG_CSV_PATH = os.path.join(AUG_DIR, "train_aug.csv")

    # --- 출력 폴더 생성 ---
    os.makedirs(AUG_DIR, exist_ok=True)
    print(f"증강된 이미지는 '{AUG_DIR}' 폴더에 저장됩니다.")

    # --- 원본 데이터 정보 로드 ---
    df_train = pd.read_csv(ORIGINAL_TRAIN_CSV)

    # --- 증강 기법 정의 ---
    # 기하학적 변환 (회전, 뒤집기 등). 이 중 하나는 반드시 적용됩니다.
    geometric_transforms = [
        A.Rotate(limit=270, p=1.0, border_mode=cv2.BORDER_CONSTANT, border_value=0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
    ]

    # 색상 및 기타 변환 (선택적으로 적용됩니다).
    other_transforms = [
        A.Blur(blur_limit=(1, 3), p=1.0),
        A.RandomSunFlare(p=1.0, flare_roi=(0, 0, 1, 0.5), angle_lower=0.3, src_radius=150),
        A.ToGray(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.SaltAndPepper(p=1.0),
        # A.GridDistortion(p=1.0),
        # A.OpticalDistortion(distort_limit=0.1, shift_limit=0.5, p=1.0),
        # A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
    ]

    augmented_data_records = []
    NUM_AUGMENTATIONS_PER_IMAGE = 7

    print("데이터 증강을 시작합니다...")
    # --- 각 이미지에 대해 증강 적용 및 저장 ---
    for _, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        img_id = row['ID']
        target = row['target']
        
        # 원본 이미지 로드
        img_path = os.path.join(ORIGINAL_TRAIN_DIR, img_id)
        image = cv2.imread(img_path)
        if image is None:
            print(f"경고: {img_path} 이미지를 읽을 수 없습니다. 건너뜁니다.")
            continue
        # Albumentations는 RGB 이미지를 사용하므로 변환합니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. 원본 이미지도 새로운 데이터셋에 포함
        original_save_path = os.path.join(AUG_DIR, img_id)
        cv2.imwrite(original_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        augmented_data_records.append({'ID': img_id, 'target': target})

        # 2. 랜덤하게 증강 기법을 조합하여 5개의 새로운 이미지 생성
        base_name, ext = os.path.splitext(img_id)
        
        used_combinations = set()
        generated_count = 0
        while generated_count < NUM_AUGMENTATIONS_PER_IMAGE:
            # 1. 기하학적 변환 중 하나를 반드시 선택
            base_augs = [random.choice(geometric_transforms)]

            # 2. 다른 변환들을 0개에서 2개까지 랜덤하게 추가 선택
            num_others_to_apply = random.randint(0, 2)
            other_augs = []
            if num_others_to_apply > 0:
                other_augs = random.sample(other_transforms, num_others_to_apply)
            
            augs_to_apply = base_augs + other_augs

            # 중복된 조합을 피하기 위해 사용된 증강 기법 조합의 시그니처를 생성
            # frozenset을 사용하여 순서에 상관없이 동일한 조합인지 확인
            combination_signature = frozenset(aug.__class__.__name__ for aug in augs_to_apply)
            if combination_signature in used_combinations:
                continue # 이미 사용된 조합이면 다시 시도
            
            used_combinations.add(combination_signature)

            # 선택된 변환들을 섞어서 순서를 무작위로 만듭니다.
            random.shuffle(augs_to_apply)

            # 증강 파이프라인 생성
            transform = A.Compose(augs_to_apply)
            augmented_image = transform(image=image)['image']
            
            new_img_id = f"{base_name}_aug_{generated_count}{ext}"
            new_img_path = os.path.join(AUG_DIR, new_img_id)
            
            # 증강된 이미지 저장
            cv2.imwrite(new_img_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            augmented_data_records.append({'ID': new_img_id, 'target': target})
            generated_count += 1

    # --- 새로운 CSV 파일 생성 ---
    aug_df = pd.DataFrame(augmented_data_records)
    aug_df.to_csv(AUG_CSV_PATH, index=False)
    print(f"증강된 데이터 정보가 '{AUG_CSV_PATH}'에 저장되었습니다.")
    print(f"총 {len(aug_df)}개의 학습 이미지가 생성되었습니다.")

if __name__ == "__main__":
    generate_augmented_data()
