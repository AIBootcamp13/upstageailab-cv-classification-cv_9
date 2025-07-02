#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 분류 전체 학습 파이프라인 실행 스크립트
EDA 결과를 바탕으로 한 체계적인 학습 시스템
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path

# 현재 파일의 위치 (예: /root/CV/src/train/run_training.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 경로 (예: /root/CV)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 루트 경로를 PYTHONPATH에 추가
sys.path.append(project_root)

from src.data.EDA import ImageClassificationEDA
from src.data.data_preprocessing import preprocess_data
from src.model.model_architecture import get_model, get_loss_function
from src.train.training_pipeline import Trainer, ModelEvaluator, get_default_config

def run_eda():
    """EDA 실행"""
    print("=" * 50)
    print("1단계: EDA (탐색적 데이터 분석)")
    print("=" * 50)
    
    data_path = "data/train_aug"
    train_csv_path = "data/train_augmented.csv"
    meta_csv_path = "data/meta.csv"
    
    # EDA 실행
    eda = ImageClassificationEDA(data_path, train_csv_path, meta_csv_path)
    eda.generate_eda_report()
    
    return eda

def run_preprocessing():
    """데이터 전처리 실행"""
    print("\n" + "=" * 50)
    print("2단계: 데이터 전처리")
    print("=" * 50)
    
    try:
        train_loader, val_loader = preprocess_data()
        print("✅ 데이터 전처리 완료")
        return train_loader, val_loader
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        return None, None

def run_training(train_loader, val_loader, config):
    """모델 학습 실행"""
    print("\n" + "=" * 50)
    print("3단계: 모델 학습")
    print("=" * 50)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 생성
    model = get_model(config['model_name'], config['num_classes'], pretrained=True)
    
    # 학습기 생성 및 학습 실행
    trainer = Trainer(model, train_loader, val_loader, device, config)
    best_accuracy = trainer.train()
    
    return trainer, best_accuracy

def run_evaluation(trainer, val_loader, meta_df, device, model_path=None):
    """모델 평가 실행"""
    print("\n" + "=" * 50)
    print("4단계: 모델 평가")
    print("=" * 50)
    
    # 클래스 이름 가져오기
    class_names = list(meta_df['class_name'])
    
    # 모델 로드 (평가만 실행하는 경우)
    if model_path and os.path.exists(model_path):
        print(f"모델 로드 중: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 아키텍처 감지
        model_name = 'efficientnet_b0'  # 기본값
        
        if 'model_name' in checkpoint:
            model_name = checkpoint['model_name']
            print(f"체크포인트에서 감지된 모델 아키텍처: {model_name}")
        elif 'config' in checkpoint and 'model_name' in checkpoint['config']:
            model_name = checkpoint['config']['model_name']
            print(f"체크포인트 설정에서 감지된 모델 아키텍처: {model_name}")
        else:
            print(f"모델 아키텍처를 감지할 수 없어 기본값 사용: {model_name}")
            print("체크포인트에 model_name 정보가 없습니다.")
        
        # 모델 아키텍처 생성
        from ..model.model_architecture import get_model
        model = get_model(model_name, len(class_names), pretrained=False)
        
        # 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # 평가기 생성 (로드된 모델 사용)
        evaluator = ModelEvaluator(model, val_loader, device, class_names)
    else:
        # 기존 trainer의 모델 사용
        evaluator = ModelEvaluator(trainer.model, val_loader, device, class_names)
    
    # 평가 실행
    predictions, targets, probabilities = evaluator.evaluate()
    
    # 리포트 생성
    save_dir = os.path.join(project_root, "results/evaluation")
    report = evaluator.generate_report(predictions, targets, probabilities, save_dir)
    
    print("✅ 모델 평가 완료")
    print(f"평가 결과 저장 위치: {save_dir}")
    
    return report

def create_experiment_config(args):
    """실험 설정 생성"""
    config = get_default_config()
    
    # 명령행 인수로 설정 업데이트
    if args.model:
        config['model_name'] = args.model
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.optimizer:
        config['optimizer'] = args.optimizer
    if args.loss_function:
        config['loss_function'] = args.loss_function
    
    # 실험별 저장 디렉토리
    experiment_name = f"{config['model_name']}_{config['optimizer']}_{config['loss_function']}"
    config['save_dir'] = os.path.join(project_root, f"models/{experiment_name}")
    
    return config

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이미지 분류 학습 파이프라인')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       help='모델 이름 (efficientnet_b0, resnet50, vit_base_patch16_224, custom_cnn)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에포크 수')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='학습률')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       help='옵티마이저 (adam, adamw, sgd)')
    parser.add_argument('--loss-function', type=str, default='cross_entropy',
                       help='손실 함수 (cross_entropy, focal, label_smoothing)')
    parser.add_argument('--skip-eda', action='store_true',
                       help='EDA 단계 건너뛰기')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='전처리 단계 건너뛰기')
    parser.add_argument('--skip-training', action='store_true',
                       help='학습 단계 건너뛰기')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='평가 단계 건너뛰기')
    parser.add_argument('--evaluation-only', action='store_true',
                       help='평가 단계만 실행')
    parser.add_argument('--model-path', type=str, default='models/final_model.pth',
                       help='평가할 모델 파일 경로 (evaluation-only 모드에서 사용)')
    
    args = parser.parse_args()
    
    print("🚀 이미지 분류 학습 파이프라인 시작")
    print(f"실험 설정: {args}")
    
    # 결과 디렉토리 생성
    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    
    # 1단계: EDA
    # eda = None
    # if not args.skip_eda:
    #     eda = run_eda()
    
    # 2단계: 데이터 전처리
    train_loader, val_loader = None, None
    if not args.skip_preprocessing:
        train_loader, val_loader = run_preprocessing()
        if train_loader is None:
            print("❌ 전처리 실패로 인해 학습을 중단합니다.")
            return    
    
    # 3단계: 모델 학습
    trainer = None
    best_accuracy = 0.0
    if not args.skip_training:
        config = create_experiment_config(args)
        trainer, best_accuracy = run_training(train_loader, val_loader, config)
    
    # 4단계: 모델 평가
    if not args.skip_evaluation:
        meta_df = pd.read_csv(os.path.join(project_root, "data/meta.csv"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.evaluation_only:
            # 평가만 실행하는 경우
            if not os.path.exists(args.model_path):
                print(f"❌ 모델 파일이 없습니다: {args.model_path}")
                return
            
            # 데이터 로더 생성 (평가용)
            train_loader, val_loader = run_preprocessing()
            if val_loader is None:
                print("❌ 데이터 로더 생성 실패")
                return
            
            report = run_evaluation(None, val_loader, meta_df, device, args.model_path)
        elif trainer is not None:
            # 전체 파이프라인에서 평가
            report = run_evaluation(trainer, val_loader, meta_df, device)
        else:
            print("❌ 평가할 모델이 없습니다.")
            return
    
    print("\n" + "=" * 50)
    print("🎉 학습 파이프라인 완료!")
    print("=" * 50)
    print(f"최고 검증 정확도: {best_accuracy:.2f}%")
    print("결과 파일 위치:")
    print("- 모델: models/")
    print("- 평가 결과: results/evaluation/")
    print("- 학습 곡선: models/*/training_curves.png")

if __name__ == "__main__":
    main() 