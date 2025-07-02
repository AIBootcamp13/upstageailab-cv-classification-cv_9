#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 분류 추론 스크립트
- 학습된 모델을 불러와서 테스트 데이터셋에 대해 예측
- submission.csv 형식으로 결과 저장
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# 커스텀 모듈 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_architecture import get_model

class TestDataset(Dataset):
    """테스트 데이터셋 클래스"""
    
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 이미지 파일명
        img_name = self.df.iloc[idx]['ID']
        img_path = os.path.join(self.image_dir, img_name)
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 전처리
        if self.transform:
            image = self.transform(image)
        
        return image

def get_test_transform():
    """테스트용 전처리 변환"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path, num_classes, device):
    """모델 로드"""
    print(f"모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 아키텍처 감지
    model_name = 'efficientnet_b0'  # 기본값
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
        print(f"감지된 모델 아키텍처: {model_name}")
    elif 'config' in checkpoint and 'model_name' in checkpoint['config']:
        model_name = checkpoint['config']['model_name']
        print(f"감지된 모델 아키텍처: {model_name}")
    else:
        print(f"모델 아키텍처를 감지할 수 없어 기본값 사용: {model_name}")
    
    # 모델 생성
    model = get_model(model_name, num_classes, pretrained=False)
    
    # 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("✅ 모델 로드 완료")
    return model

def get_test_loader(test_csv_path, test_img_dir, batch_size=32):
    """테스트 데이터 로더 생성"""
    print(f"테스트 데이터 로더 생성 중...")
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(test_csv_path)
    print(f"테스트 데이터 수: {len(test_df)}")
    
    # 전처리 변환
    test_transform = get_test_transform()
    
    # 데이터셋 생성
    test_dataset = TestDataset(test_df, test_img_dir, test_transform)
    
    # 데이터 로더 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ 테스트 데이터 로더 생성 완료 (배치 수: {len(test_loader)})")
    return test_loader, test_df

def predict(model, test_loader, device):
    """예측 실행"""
    print("예측 시작...")
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            
            # 진행률 출력
            if (batch_idx + 1) % 10 == 0:
                print(f"예측 진행률: {batch_idx + 1}/{len(test_loader)}")
    
    print(f"✅ 예측 완료 (총 {len(predictions)}개)")
    return predictions

def save_submission(test_df, predictions, submission_path):
    """submission.csv 저장"""
    print(f"submission.csv 저장 중: {submission_path}")
    
    # 결과 데이터프레임 생성
    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "target": predictions
    })
    
    # 저장
    submission.to_csv(submission_path, index=False)
    
    print(f"✅ submission.csv 저장 완료")
    
    # 예측 결과 통계
    print(f"\n=== 예측 결과 통계 ===")
    print(f"총 예측 수: {len(predictions)}")
    print(f"클래스별 예측 수:")
    for class_id, count in enumerate(np.bincount(predictions)):
        print(f"  클래스 {class_id}: {count}개")

def run_inference(model_path, test_csv_path, test_img_dir, submission_path, batch_size=32):
    """추론 실행 메인 함수"""
    print("=" * 50)
    print("이미지 분류 추론 시작")
    print("=" * 50)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 클래스 수 확인 (meta.csv에서)
    meta_path = "data/meta.csv"
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        num_classes = len(meta_df)
        print(f"클래스 수: {num_classes}")
    else:
        num_classes = 17  # 기본값
        print(f"meta.csv를 찾을 수 없어 기본 클래스 수 사용: {num_classes}")
    
    # 1. 모델 로드
    model = load_model(model_path, num_classes, device)
    
    # 2. 테스트 데이터 로더 생성
    test_loader, test_df = get_test_loader(test_csv_path, test_img_dir, batch_size)
    
    # 3. 예측 실행
    predictions = predict(model, test_loader, device)
    
    # 4. submission.csv 저장
    save_submission(test_df, predictions, submission_path)
    
    print("\n" + "=" * 50)
    print("🎉 추론 완료!")
    print("=" * 50)

if __name__ == "__main__":
    # 프로젝트 루트 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 설정
    model_path = os.path.join(project_root, "models/best_model.pth")
    test_csv_path = os.path.join(project_root, "data/sample_submission.csv")
    test_img_dir = os.path.join(project_root, "data/test")
    submission_path = os.path.join(project_root, "data/submission.csv")
    
    print(f"프로젝트 루트: {project_root}")
    print(f"모델 경로: {model_path}")
    print(f"테스트 CSV: {test_csv_path}")
    print(f"테스트 이미지: {test_img_dir}")
    
    # 추론 실행
    run_inference(model_path, test_csv_path, test_img_dir, submission_path) 