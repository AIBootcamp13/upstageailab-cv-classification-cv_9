# -*- coding: utf-8 -*-
"""
이미지 분류를 위한 데이터 전처리 모듈
- 이미지 리사이징 및 정규화
- 클래스 불균형 해결을 위한 데이터 증강
- 문서 특화 증강 기법
- 시각화 기능
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import warnings
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')


# 현재 파일의 위치 (예: /root/CV/src/data/data_preprocessing.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 경로 (예: /root/CV)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 루트 경로를 PYTHONPATH에 추가
sys.path.append(project_root)

class ImagePreprocessor:
    """이미지 전처리 클래스"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def resize_image(self, image_path):
        """이미지 리사이징 (비율 유지 패딩)"""
        try:
            # 이미지 로드
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image_path
            
            # 원본 크기
            h, w = img.shape[:2]
            target_h, target_w = self.target_size
            
            # 비율 계산
            ratio = min(target_w / w, target_h / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            # 리사이징
            resized = cv2.resize(img, (new_w, new_h))
            
            # 패딩으로 정사각형 만들기
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            
            return padded
            
        except Exception as e:
            print(f"이미지 리사이징 실패: {e}")
            return None
    
    def balance_classes(self, train_df, meta_df):
        """클래스별 샘플 수를 균형있게 조정"""
        print("=== 클래스 불균형 해결을 위한 데이터 증강 ===")
        
        balanced_data = []
        class_name_mapping = dict(zip(meta_df['target'], meta_df['class_name']))
        
        for target in train_df['target'].unique():
            class_samples = train_df[train_df['target'] == target]
            current_count = len(class_samples)
            target_count = 100  # 각 클래스당 100개로 설정
            
            class_name = class_name_mapping.get(target, f'class_{target}')
            print(f"클래스 {target} ({class_name}): {current_count} → {target_count}")
            
            if current_count < target_count:
                # 부족한 샘플 수만큼 증강
                needed_samples = target_count - current_count
                augmented_samples = self._augment_class_samples(
                    class_samples, needed_samples, target
                )
                balanced_data.extend(augmented_samples)
            
            # 기존 샘플들도 추가
            balanced_data.extend(class_samples.to_dict('records'))
        
        balanced_df = pd.DataFrame(balanced_data)
        print(f"증강 후 총 샘플 수: {len(balanced_df)}")
        
        # 클래스별 샘플 수 출력
        print("\n=== 클래스별 샘플 수 ===")
        for class_id in sorted(balanced_df['target'].unique()):
            class_name = meta_df[meta_df['target'] == class_id]['class_name'].iloc[0]
            count = len(balanced_df[balanced_df['target'] == class_id])
            print(f"클래스 {class_id} ({class_name}): {count}")
        
        return balanced_df
    
    def _augment_class_samples(self, class_samples, needed_count, target):
        """특정 클래스의 샘플을 증강"""
        augmented_samples = []
        
        # 증강 기법별 가중치 설정
        augmentation_methods = ['light', 'medium', 'strong']
        weights = [0.3, 0.5, 0.2]
        
        for i in range(needed_count):
            # 랜덤하게 기존 샘플 선택
            sample = class_samples.sample(n=1).iloc[0]
            
            # 증강 기법 선택
            method = np.random.choice(augmentation_methods, p=weights)
            
            # 증강된 샘플 정보 생성
            augmented_sample = sample.copy()
            augmented_sample['ID'] = f"{sample['ID']}_aug_{method}_{i}"
            augmented_sample['is_augmented'] = True
            augmented_sample['augmentation_method'] = method
            
            augmented_samples.append(augmented_sample)
        
        return augmented_samples
    
    def create_dataloader(self, df, data_path, batch_size=32, shuffle=True, augmentation_level='medium'):
        """기본 데이터 로더 생성"""
        print(f"기본 데이터 로더 생성 (배치 크기: {batch_size}, 증강 레벨: {augmentation_level})")
        
        # 증강 기법 설정
        if augmentation_level == 'light':
            transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=5, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif augmentation_level == 'medium':
            transform = A.Compose([
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.1),
                A.Transpose(p=0.1),
                A.Rotate(limit=10, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # strong
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Transpose(p=0.2),
                A.Rotate(limit=15, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.4),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.4),
                A.RGBShift(p=0.4),
                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.2),
                A.MedianBlur(p=0.2),
                A.Perspective(p=0.2),
                A.ElasticTransform(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 데이터셋 생성
        dataset = DocumentImageDataset(df, data_path, transform=transform, target_size=self.target_size)
        
        # DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"데이터셋 크기: {len(dataset)}")
        print(f"배치 수: {len(dataloader)}")
        
        return dataloader
    
    def create_advanced_dataloader(self, df, data_path, batch_size=32, shuffle=True):
        """고급 데이터 로더 생성"""
        print(f"고급 데이터 로더 생성 (배치 크기: {batch_size})")
        
        # 고급 증강 기법 설정 (문서 특화)
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Transpose(p=0.2),
            A.Rotate(limit=20, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.RGBShift(p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.ISONoise(p=0.2),
            A.MultiplicativeNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(p=0.2),
            A.MedianBlur(p=0.2),
            A.Perspective(p=0.2),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터셋 생성
        dataset = DocumentImageDataset(df, data_path, transform=transform, target_size=self.target_size)
        
        # DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"고급 데이터셋 크기: {len(dataset)}")
        print(f"고급 배치 수: {len(dataloader)}")
        
        return dataloader

class DataAugmentation:
    """데이터 증강 클래스 - 문서 이미지 특성에 최적화"""
    
    def __init__(self, strength='medium'):
        self.strength = strength
        self.transforms = self._get_transforms()
    
    def _get_transforms(self):
        """증강 강도에 따른 변환 설정"""
        if self.strength == 'light':
            return A.Compose([
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=5, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.strength == 'medium':
            return A.Compose([
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.1),
                A.Transpose(p=0.1),
                A.Rotate(limit=10, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # strong
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Transpose(p=0.2),
                A.Rotate(limit=15, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.4),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.4),
                A.RGBShift(p=0.4),
                A.GaussNoise(p=0.3),
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.2),
                A.MedianBlur(p=0.2),
                A.Perspective(p=0.2),
                A.ElasticTransform(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

class DocumentImageDataset(Dataset):
    """문서 이미지 데이터셋 클래스"""
    
    def __init__(self, df, data_path, transform=None, target_size=(224, 224)):
        self.df = df
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_size = target_size
        
        # 이미지 파일 확장자 매핑
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['ID']
        label = row['target']
        
        # 이미지 파일 찾기 (증강된 이미지는 이미 올바른 파일명으로 저장됨)
        image_path = None
        
        # 먼저 정확한 파일명으로 찾기
        temp_path = self.data_path / image_id
        if temp_path.exists():
            image_path = temp_path
        else:
            # 확장자가 없는 경우 확장자 추가해서 찾기
            for ext in self.image_extensions:
                temp_path = self.data_path / f"{image_id}{ext}"
                if temp_path.exists():
                    image_path = temp_path
                    break
        
        if image_path is None:
            # 기본값으로 빈 이미지 생성
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
            print(f"이미지 파일을 찾을 수 없습니다.")
        else:
            # 이미지 로드
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    image = np.zeros((*self.target_size, 3), dtype=np.uint8)
                    print(f"이미지 로드 실패: {image_path}")
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                image = np.zeros((*self.target_size, 3), dtype=np.uint8)
                print(f"이미지 로드 오류 ({image_path}): {e}")
        
        # 이미지 리사이징 (비율 유지 패딩)
        image = self._resize_with_padding(image)
        
        # 변환 적용
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"변환 적용 실패: {e}")
                # 변환 실패 시 원본 이미지 사용
                pass
        
        # 텐서로 변환
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def _resize_with_padding(self, image):
        """이미지 리사이징 (비율 유지 패딩)"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # 비율 계산
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # 리사이징
        resized = cv2.resize(image, (new_w, new_h))
        
        # 패딩으로 정사각형 만들기
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded

def visualize_preprocessing_results(data_path, train_csv_path, meta_csv_path, num_samples=5):
    """전처리 결과 시각화"""
    print("=== 전처리 결과 시각화 ===")
    
    # 데이터 로드
    train_df = pd.read_csv(train_csv_path)
    meta_df = pd.read_csv(meta_csv_path)
    
    # 전처리기 생성
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # 실제 존재하는 이미지 파일 찾기
    actual_files = list(Path(data_path).glob("*.jpg"))
    if not actual_files:
        actual_files = list(Path(data_path).glob("*.jpeg"))
    if not actual_files:
        actual_files = list(Path(data_path).glob("*.png"))
    
    if not actual_files:
        print("시각화할 수 있는 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 이미지 파일 수: {len(actual_files)}")
    
    # 실제 파일 중에서 샘플 선택
    selected_files = random.sample(actual_files, min(num_samples, len(actual_files)))
    
    # 샘플 이미지 처리
    fig, axes = plt.subplots(2, len(selected_files), figsize=(15, 6))
    
    for i, image_path in enumerate(selected_files):
        print(f"처리 중: {image_path}")
        
        try:
            # 원본 이미지
            original = cv2.imread(str(image_path))
            if original is not None:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(original)
                axes[0, i].set_title(f"원본\n{image_path.name}")
                axes[0, i].axis('off')
            else:
                axes[0, i].text(0.5, 0.5, '로드 실패', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(f"로드 실패\n{image_path.name}")
                axes[0, i].axis('off')
            
            # 전처리된 이미지
            processed = preprocessor.resize_image(str(image_path))
            if processed is not None:
                axes[1, i].imshow(processed)
                axes[1, i].set_title(f"전처리\n224x224")
                axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, '처리 실패', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f"처리 실패\n{image_path.name}")
                axes[1, i].axis('off')
                
        except Exception as e:
            print(f"이미지 처리 실패 ({image_path.name}): {e}")
            axes[0, i].text(0.5, 0.5, '오류', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f"오류\n{image_path.name}")
            axes[0, i].axis('off')
            axes[1, i].text(0.5, 0.5, '오류', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f"오류\n{image_path.name}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # 파일 저장
    try:
        save_path = os.path.join(project_root, 'preprocessing_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"전처리 시각화 파일 저장 완료: {save_path}")
    except Exception as e:
        print(f"파일 저장 실패: {e}")
    
    plt.show()

def preprocess_data(use_kfold=True, n_splits=5, fold_idx=None):
    """데이터 전처리 메인 함수 - Stratified K-Fold 또는 단일 분할"""
    print("=== 데이터 전처리 시작 ===")
    
    if use_kfold:
        # Stratified K-Fold 검증 사용
        validator, dataloaders, final_train_df, meta_df = create_stratified_kfold_data(
            n_splits=n_splits, 
            random_state=42
        )
        
        if fold_idx is not None:
            # 특정 폴드만 반환
            if fold_idx >= len(dataloaders):
                raise ValueError(f"폴드 인덱스 {fold_idx}가 범위를 벗어났습니다. (0-{len(dataloaders)-1})")
            
            fold_data = dataloaders[fold_idx]
            print(f"\n=== Fold {fold_idx + 1} 데이터 로더 반환 ===")
            return fold_data['train_loader'], fold_data['val_loader'], validator, dataloaders
        else:
            # 첫 번째 폴드 반환 (기본값)
            fold_data = dataloaders[0]
            print(f"\n=== Fold 1 데이터 로더 반환 (기본값) ===")
            return fold_data['train_loader'], fold_data['val_loader'], validator, dataloaders
    else:
        # 기존 단일 분할 방식
        print("=== 단일 분할 방식 사용 ===")
        
        # 파일 경로 설정
        train_path = os.path.join(project_root, "data/train")
        train_csv_path = os.path.join(project_root, "data/train.csv")
        train_augmented_csv_path = os.path.join(project_root, "data/train_augmented.csv")
        meta_csv_path = os.path.join(project_root, "data/meta.csv")
        
        # 데이터 로드
        train_df = pd.read_csv(train_csv_path)
        meta_df = pd.read_csv(meta_csv_path)
        
        print(f"원본 데이터 크기: {len(train_df)}")
        
        # 증강된 데이터가 있는지 확인하고 로드
        if os.path.exists(train_augmented_csv_path):
            train_augmented_df = pd.read_csv(train_augmented_csv_path)
            print(f"증강된 데이터 크기: {len(train_augmented_df)}")
            
            # 원본과 증강된 데이터 통합
            final_train_df = train_augmented_df.copy()
            print(f"통합된 데이터 크기: {len(final_train_df)}")
            
            # 증강된 샘플 수 확인
            augmented_count = len(final_train_df[final_train_df['is_augmented'] == True])
            original_count = len(final_train_df[final_train_df['is_augmented'] == False])
            print(f"  - 원본 샘플: {original_count}")
            print(f"  - 증강 샘플: {augmented_count}")
        else:
            print("증강된 데이터가 없습니다. 원본 데이터만 사용합니다.")
            final_train_df = train_df.copy()
        
        # 학습/검증 분할 (80:20)
        from sklearn.model_selection import train_test_split
        
        train_data, val_data = train_test_split(
            final_train_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=final_train_df['target']
        )
        
        print(f"학습 데이터 크기: {len(train_data)}")
        print(f"검증 데이터 크기: {len(val_data)}")
        
        # 전처리기 생성
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # 데이터 로더 생성
        print("\n=== train 폴더의 모든 이미지 사용하여 데이터 로더 생성 ===")
        train_loader = preprocessor.create_dataloader(
            train_data, 
            train_path, 
            batch_size=32, 
            shuffle=True,
            augmentation_level='medium'
        )
        
        val_loader = preprocessor.create_dataloader(
            val_data, 
            train_path, 
            batch_size=32, 
            shuffle=False,
            augmentation_level='medium'
        )
        
        print(f"데이터 로더 생성 완료:")
        print(f"  - 학습 데이터셋 크기: {len(train_data)}")
        print(f"  - 검증 데이터셋 크기: {len(val_data)}")
        print(f"  - 학습 배치 수: {len(train_loader)}")
        print(f"  - 검증 배치 수: {len(val_loader)}")
        print(f"  - 사용 폴더: {train_path}")
        
        return train_loader, val_loader, None, None

class StratifiedKFoldValidator:
    """Stratified K-Fold 검증 클래스"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.folds = []
    
    def create_folds(self, df):
        """Stratified K-Fold 분할 생성"""
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # 클래스별 분포 확인
        class_counts = df['target'].value_counts().sort_index()
        print(f"=== 클래스별 분포 ===")
        for class_id, count in class_counts.items():
            print(f"클래스 {class_id}: {count}개")
        
        # K-Fold 분할 생성
        self.folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
            train_fold = df.iloc[train_idx].reset_index(drop=True)
            val_fold = df.iloc[val_idx].reset_index(drop=True)
            
            # 각 폴드의 클래스 분포 확인
            train_class_counts = train_fold['target'].value_counts().sort_index()
            val_class_counts = val_fold['target'].value_counts().sort_index()
            
            print(f"\n=== Fold {fold_idx + 1} ===")
            print(f"학습 데이터: {len(train_fold)}개")
            print(f"검증 데이터: {len(val_fold)}개")
            print(f"학습/검증 비율: {len(train_fold)}/{len(val_fold)} ({len(train_fold)/len(df)*100:.1f}%/{len(val_fold)/len(df)*100:.1f}%)")
            
            # 클래스별 분포 확인
            print("클래스별 분포:")
            for class_id in sorted(df['target'].unique()):
                train_count = train_class_counts.get(class_id, 0)
                val_count = val_class_counts.get(class_id, 0)
                print(f"  클래스 {class_id}: 학습 {train_count}개, 검증 {val_count}개")
            
            self.folds.append({
                'fold_idx': fold_idx,
                'train_data': train_fold,
                'val_data': val_fold,
                'train_class_counts': train_class_counts,
                'val_class_counts': val_class_counts
            })
        
        return self.folds
    
    def get_fold(self, fold_idx):
        """특정 폴드 데이터 반환"""
        if fold_idx >= len(self.folds):
            raise ValueError(f"폴드 인덱스 {fold_idx}가 범위를 벗어났습니다. (0-{len(self.folds)-1})")
        return self.folds[fold_idx]
    
    def get_all_folds(self):
        """모든 폴드 데이터 반환"""
        return self.folds
    
    def create_dataloaders_for_fold(self, fold_data, data_path, batch_size=32, augmentation_level='medium'):
        """특정 폴드에 대한 데이터 로더 생성"""
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # 학습 데이터 로더 (증강 적용)
        train_loader = preprocessor.create_dataloader(
            fold_data['train_data'],
            data_path,
            batch_size=batch_size,
            shuffle=True,
            augmentation_level=augmentation_level
        )
        
        # 검증 데이터 로더 (증강 없음)
        val_loader = preprocessor.create_dataloader(
            fold_data['val_data'],
            data_path,
            batch_size=batch_size,
            shuffle=False,
            augmentation_level='none'  # 검증에는 증강 적용하지 않음
        )
        
        return train_loader, val_loader
    
    def create_dataloaders_for_all_folds(self, data_path, batch_size=32, augmentation_level='medium'):
        """모든 폴드에 대한 데이터 로더 생성"""
        dataloaders = []
        
        for fold_data in self.folds:
            train_loader, val_loader = self.create_dataloaders_for_fold(
                fold_data, data_path, batch_size, augmentation_level
            )
            
            dataloaders.append({
                'fold_idx': fold_data['fold_idx'],
                'train_loader': train_loader,
                'val_loader': val_loader,
                'train_data': fold_data['train_data'],
                'val_data': fold_data['val_data']
            })
        
        return dataloaders


def create_stratified_kfold_data(n_splits=5, random_state=42):
    """Stratified K-Fold 데이터 생성"""
    print("=== Stratified K-Fold 검증 데이터 생성 ===")
    
    # 파일 경로 설정
    train_path = os.path.join(project_root, "data/train")
    train_csv_path = os.path.join(project_root, "data/train.csv")
    train_augmented_csv_path = os.path.join(project_root, "data/train_augmented.csv")
    meta_csv_path = os.path.join(project_root, "data/meta.csv")
    
    # 데이터 로드
    train_df = pd.read_csv(train_csv_path)
    meta_df = pd.read_csv(meta_csv_path)
    
    print(f"원본 데이터 크기: {len(train_df)}")
    
    # 증강된 데이터가 있는지 확인하고 로드
    if os.path.exists(train_augmented_csv_path):
        train_augmented_df = pd.read_csv(train_augmented_csv_path)
        print(f"증강된 데이터 크기: {len(train_augmented_df)}")
        
        # 원본과 증강된 데이터 통합
        final_train_df = train_augmented_df.copy()
        print(f"통합된 데이터 크기: {len(final_train_df)}")
        
        # 증강된 샘플 수 확인
        augmented_count = len(final_train_df[final_train_df['is_augmented'] == True])
        original_count = len(final_train_df[final_train_df['is_augmented'] == False])
        print(f"  - 원본 샘플: {original_count}")
        print(f"  - 증강 샘플: {augmented_count}")
    else:
        print("증강된 데이터가 없습니다. 원본 데이터만 사용합니다.")
        final_train_df = train_df.copy()
    
    # Stratified K-Fold 검증기 생성
    validator = StratifiedKFoldValidator(n_splits=n_splits, random_state=random_state)
    
    # K-Fold 분할 생성
    folds = validator.create_folds(final_train_df)
    
    # 모든 폴드에 대한 데이터 로더 생성
    dataloaders = validator.create_dataloaders_for_all_folds(
        train_path, 
        batch_size=32, 
        augmentation_level='medium'
    )
    
    print(f"\n=== K-Fold 검증 설정 완료 ===")
    print(f"폴드 수: {n_splits}")
    print(f"총 데이터 크기: {len(final_train_df)}")
    print(f"각 폴드 평균 크기: {len(final_train_df) // n_splits}")
    
    return validator, dataloaders, final_train_df, meta_df

if __name__ == "__main__":
    preprocess_data()