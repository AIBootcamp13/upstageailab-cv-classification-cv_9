import os
import sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast,
    HueSaturationValue, GaussNoise, GaussianBlur, ElasticTransform,
    Perspective, GridDistortion, RandomRotate90, RandomScale,
    Compose, Resize, CoarseDropout, Sharpen, CLAHE, Emboss, MotionBlur, MedianBlur, RandomCrop
)

# === 경로 설정 ===
# 현재 파일의 위치 (예: /root/CV/src/data/augmentation.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 경로 (예: /root/CV)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 루트 경로를 PYTHONPATH에 추가
sys.path.append(project_root)

INPUT_DIR = os.path.join(project_root, "data/train")            # 원본 이미지 디렉토리
OUTPUT_DIR = os.path.join(project_root, "data/train_aug")       # 증강 이미지 저장 경로
CSV_PATH = os.path.join(project_root, "data/train.csv")         # 원본 메타
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 데이터 로드 ===
df = pd.read_csv(CSV_PATH)
target_per_class = 1000  # 클래스당 최소 목표 수

# === 증강 기법 정의 ===
AUGMENTATIONS = {
    # "flip": Compose([HorizontalFlip(p=1.0)]),
    # "rotate10": Compose([Rotate(limit=10, p=1.0)]),
    # "rotate20": Compose([Rotate(limit=20, p=1.0)]),
    # "rotate30": Compose([Rotate(limit=30, p=1.0)]),
    # "rotate45": Compose([Rotate(limit=45, p=1.0)]),
    # "rotate60": Compose([Rotate(limit=60, p=1.0)]),
    # "rotate90": Compose([RandomRotate90(p=1.0)]),
    # "rotate180": Compose([Rotate(limit=180, p=1.0)]),
    # "zoom": Compose([RandomScale(scale_limit=0.2, p=1.0), Resize(224, 224)]),
    # "translate": Compose([ShiftScaleRotate(shift_limit=0.2, scale_limit=0.0, rotate_limit=0, p=1.0)]),
    # "scale_shift": Compose([ShiftScaleRotate(p=1.0)]),
    # "color_jitter": Compose([RandomBrightnessContrast(p=1.0)]),
    # "hsv": Compose([HueSaturationValue(p=1.0)]),
    # "noise": Compose([GaussNoise(p=1.0)]),
    # "blur": Compose([GaussianBlur(p=1.0)]),
    # "elastic": Compose([ElasticTransform(p=1.0)]),
    # "perspective": Compose([Perspective(p=1.0)]),
    # "grid_distort": Compose([GridDistortion(p=1.0)]),
    # "combined1": Compose([HorizontalFlip(p=1.0), Rotate(limit=15, p=1.0), RandomBrightnessContrast(p=1.0)]),
    # "combined2": Compose([Rotate(limit=45, p=1.0), GaussNoise(p=1.0), GaussianBlur(p=1.0)]),
    # "combined3": Compose([Rotate(limit=75, p=1.0), GaussNoise(p=1.0), GaussianBlur(p=1.0)]),
    # "rotate_heavy": Compose([Rotate(limit=90, p=1.0), RandomBrightnessContrast(p=1.0)]),
    # "rotate_light": Compose([Rotate(limit=5, p=1.0), HorizontalFlip(p=0.5)]),
    # 증강 추가
    "combined1": Compose([Rotate(limit=90, p=1.0), GaussNoise(p=1.0), GaussianBlur(p=1.0)]),
    "combined2": Compose([Rotate(limit=180, p=1.0), GaussNoise(p=1.0), GaussianBlur(p=1.0)]),
    "combined3": Compose([Rotate(limit=90, p=1.0), GaussNoise(p=1.0)]),
    "combined4": Compose([Rotate(limit=180, p=1.0), GaussianBlur(p=1.0)]),
    "combined5": Compose([Rotate(limit=180, p=1.0), VerticalFlip(p=1.0)]),
    "combined6": Compose([Rotate(limit=180, p=1.0), HorizontalFlip(p=1.0)]),
    "combined7": Compose([VerticalFlip(p=1.0), HorizontalFlip(p=1.0)]),
    "combined8": Compose([VerticalFlip(p=1.0), GaussianBlur(p=1.0)]),
    "combined9": Compose([VerticalFlip(p=1.0), GaussNoise(p=1.0)]),
    "combined10": Compose([VerticalFlip(p=1.0), Rotate(limit=15, p=1.0), RandomBrightnessContrast(p=1.0)]),
    "combined11": Compose([VerticalFlip(p=1.0), Rotate(limit=45, p=1.0), GaussianBlur(p=1.0)]),
    "combined12": Compose([HorizontalFlip(p=1.0), GaussianBlur(p=1.0)]),
    "combined13": Compose([HorizontalFlip(p=1.0), GaussNoise(p=1.0)]),
    "combined14": Compose([HorizontalFlip(p=1.0), Rotate(limit=15, p=1.0), RandomBrightnessContrast(p=1.0)]),
    "combined15": Compose([HorizontalFlip(p=1.0), Rotate(limit=45, p=1.0), GaussNoise(p=1.0)]),
    # 증강 추가 2 - 문서 이미지 최적화
    # "Sharpen1": Compose([Sharpen(alpha=(0.2, 0.4), lightness=(0.8, 1.2), p=1.0)]),  # 약한 선명화
    # "Sharpen2": Compose([Sharpen(alpha=(0.4, 0.7), lightness=(0.7, 1.0), p=1.0)]),  # 강한 선명화
    # "CLAHE1": Compose([CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0)]),      # 약한 대비 향상
    # "CLAHE2": Compose([CLAHE(clip_limit=3.0, tile_grid_size=(16, 16), p=1.0)]),    # 강한 대비 향상
    # "emboss1": Compose([Emboss(alpha=(0.2, 0.4), p=1.0)]),   # 약한 엠보싱
    # "emboss2": Compose([Emboss(alpha=(0.4, 0.7), p=1.0)]),   # 강한 엠보싱
    # "motionblur1": Compose([MotionBlur(blur_limit=3, p=1.0)]),                     # 약한 모션 블러
    # "motionblur2": Compose([MotionBlur(blur_limit=7, p=1.0)]),                     # 강한 모션 블러
    # "medianblur1": Compose([MedianBlur(blur_limit=3, p=1.0)]),                     # 약한 중간값 블러
    # "medianblur2": Compose([MedianBlur(blur_limit=5, p=1.0)]),                     # 강한 중간값 블러
    # "sharpCLAHE": Compose([Sharpen(alpha=(0.3, 0.5), lightness=(0.8, 1.1), p=1.0), CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)]),
    # "randomcrop": Compose([RandomCrop(height=200, width=200, p=1.0)]),             # 200x200 크기로 자르기
    # "shiftscale": Compose([ShiftScaleRotate(shift_limit=0.2, scale_limit=0.0, rotate_limit=0, p=1.0)]),
    # "coarseDropout1": Compose([CoarseDropout(p=1.0)]),  # 작은 구멍
    # "coarseDropout2": Compose([CoarseDropout(p=1.0)]),  # 큰 구멍
}

# === 결과 저장용 리스트 ===
augmented_records = []

# === 증강 함수 ===
def apply_and_save(image, image_id, label, aug_name, transform, idx):
    augmented = transform(image=image)['image']
    filename = f"{image_id}_{aug_name}_{idx}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
    augmented_records.append({
        "ID": filename,
        "target": label,
        "is_augmented": True,
        "augmentation_method": aug_name
    })

if __name__ == "__main__":
    # === 증강 실행 ===
    for class_id in tqdm(df['target'].unique(), desc="클래스별 증강"):
        class_df = df[df['target'] == class_id]
        current_images = class_df['ID'].tolist()
        current_count = len(current_images)
        # needed = target_per_class - current_count
        needed = target_per_class

        if needed <= 0:
            continue

        # rotate 기법들을 더 자주 선택하도록 가중치 설정
        # rotate_augmentations = ["rotate10", "rotate20", "rotate30", "rotate45", "rotate60", "rotate90", "rotate180", "rotate_heavy", "rotate_light", "flip"]
        rotate_augmentations = []
        other_augmentations = [name for name in AUGMENTATIONS.keys() if name not in rotate_augmentations]
        
        # rotate 기법별 가중치 (더 다양한 각도가 선택되도록)
        rotate_weights = {
            "rotate10": 0.05,    # 작은 각도
            "rotate20": 0.05,    # 작은 각도
            "rotate30": 0.15,    # 중간 각도
            "rotate45": 0.15,    # 중간 각도
            "rotate60": 0.10,    # 큰 각도
            "rotate90": 0.15,    # 90도
            "rotate180": 0.10,   # 180도
            "rotate_heavy": 0.10, # 복합 rotate
            "rotate_light": 0.05,  # 가벼운 rotate
            "flip": 0.10
        }
        
        # rotate 기법이 70%, 다른 기법이 30% 비율로 선택되도록 조정
        rotate_weight = 0
        other_weight = 1
        
        image_idx = 0

        while needed > 0:
            row = class_df.iloc[image_idx % current_count]
            image_path = os.path.join(INPUT_DIR, row['ID'])
            
            # 파일 존재 여부 확인
            if not os.path.exists(image_path):
                print(f"파일을 찾을 수 없습니다: {image_path}")
                image_idx += 1
                continue
                
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                image_idx += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # rotate 기법 우선 선택 (가중치 적용)
            if np.random.random() < rotate_weight and rotate_augmentations:
                aug_name = np.random.choice(rotate_augmentations, p=[rotate_weights[name] for name in rotate_augmentations])
            else:
                aug_name = np.random.choice(other_augmentations)
            
            aug_transform = AUGMENTATIONS[aug_name]
            
            try:
                apply_and_save(image, row['ID'].split('.')[0], row['target'], aug_name, aug_transform, image_idx)
                needed -= 1
            except Exception as e:
                print(f"오류 - {row['ID']} {aug_name}: {e}")

            image_idx += 1

    # === 원본 + 증강된 데이터 통합 저장 ===
    original_records = df.copy()
    original_records['is_augmented'] = False
    original_records['augmentation_method'] = "-"

    aug_df = pd.DataFrame(augmented_records)
    final_df = pd.concat([original_records, aug_df], ignore_index=True)
    final_df.to_csv(os.path.join(project_root, "data/train_augmented.csv"), index=False)

    # 증강 결과 통계 출력
    print("\n=== 증강 결과 통계 ===")
    print(f"원본 데이터 수: {len(df)}")
    print(f"증강된 데이터 수: {len(augmented_records)}")
    print(f"총 데이터 수: {len(final_df)}")

    # rotate 기법별 통계
    rotate_counts = {}
    for record in augmented_records:
        aug_method = record['augmentation_method']
        if 'rotate' in aug_method:
            rotate_counts[aug_method] = rotate_counts.get(aug_method, 0) + 1

    print(f"\n=== Rotate 기법별 통계 ===")
    for method, count in sorted(rotate_counts.items()):
        print(f"{method}: {count}개")

    print(f"\n✅ 증강 완료. 저장 위치: {OUTPUT_DIR}, 메타: data/train_augmented.csv")
