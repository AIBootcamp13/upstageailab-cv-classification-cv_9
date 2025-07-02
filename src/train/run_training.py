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
import numpy as np
from pathlib import Path

# 커스텀 모듈 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
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
    
    data_path = "data/train"
    train_csv_path = "data/train.csv"
    meta_csv_path = "data/meta.csv"
    
    # EDA 실행
    eda = ImageClassificationEDA(data_path, train_csv_path, meta_csv_path)
    eda.generate_eda_report()
    
    return eda

def run_preprocessing(use_kfold=True, n_splits=5, fold_idx=None):
    """데이터 전처리 실행"""
    print("\n" + "=" * 50)
    print("2단계: 데이터 전처리")
    print("=" * 50)
    
    try:
        if use_kfold:
            print(f"Stratified K-Fold 검증 사용 (폴드 수: {n_splits})")
            if fold_idx is not None:
                print(f"특정 폴드 사용: Fold {fold_idx + 1}")
            else:
                print("기본 폴드 사용: Fold 1")
        
        train_loader, val_loader, validator, dataloaders = preprocess_data(
            use_kfold=use_kfold, 
            n_splits=n_splits, 
            fold_idx=fold_idx
        )
        print("✅ 데이터 전처리 완료")
        return train_loader, val_loader, validator, dataloaders
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

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
    save_dir = Path("results/evaluation")
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
    config['save_dir'] = f"models/{experiment_name}"
    
    # wandb 설정
    config['use_wandb'] = args.use_wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_run_name'] = args.wandb_run_name
    
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
    parser.add_argument('--use-wandb', action='store_true',
                       help='wandb 로깅 활성화')
    parser.add_argument('--wandb-project', type=str, default='image-classification',
                       help='wandb 프로젝트 이름')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='wandb 실행 이름 (None이면 자동 생성)')
    parser.add_argument('--use-kfold', action='store_true', default=True,
                       help='Stratified K-Fold 검증 사용')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='K-Fold 분할 수')
    parser.add_argument('--fold-idx', type=int, default=None,
                       help='사용할 폴드 인덱스 (None이면 첫 번째 폴드)')
    parser.add_argument('--cross-validate', action='store_true',
                       help='모든 폴드에 대해 교차 검증 실행')

    
    args = parser.parse_args()
    
    print("🚀 이미지 분류 학습 파이프라인 시작")
    print(f"실험 설정: {args}")
    
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1단계: EDA
    # eda = None
    # if not args.skip_eda:
    #     eda = run_eda()
    
    # 2단계: 데이터 전처리
    train_loader, val_loader = None, None
    validator, dataloaders = None, None
    
    if not args.skip_preprocessing:
        train_loader, val_loader, validator, dataloaders = run_preprocessing(
            use_kfold=args.use_kfold,
            n_splits=args.n_splits,
            fold_idx=args.fold_idx
        )
        if train_loader is None:
            print("❌ 전처리 실패로 인해 학습을 중단합니다.")
            return
    
    # 3단계: 모델 학습
    trainer = None
    best_accuracy = 0.0
    
    if not args.skip_training:
        config = create_experiment_config(args)
        
        if args.cross_validate and validator is not None:
            # 모든 폴드에 대해 교차 검증
            print("\n" + "=" * 50)
            print("교차 검증 시작")
            print("=" * 50)
            
            fold_accuracies = []
            fold_trainers = []
            
            for fold_idx in range(len(dataloaders)):
                print(f"\n--- Fold {fold_idx + 1} 학습 시작 ---")
                
                # 해당 폴드의 데이터 로더 가져오기
                fold_data = dataloaders[fold_idx]
                fold_train_loader = fold_data['train_loader']
                fold_val_loader = fold_data['val_loader']
                
                # 폴드별 설정 업데이트
                fold_config = config.copy()
                fold_config['save_dir'] = f"{config['save_dir']}/fold_{fold_idx + 1}"
                fold_config['wandb_run_name'] = f"{config['wandb_run_name']}_fold_{fold_idx + 1}" if config['wandb_run_name'] else f"fold_{fold_idx + 1}"
                
                # 학습 실행
                fold_trainer, fold_accuracy = run_training(fold_train_loader, fold_val_loader, fold_config)
                
                fold_accuracies.append(fold_accuracy)
                fold_trainers.append(fold_trainer)
                
                print(f"Fold {fold_idx + 1} 정확도: {fold_accuracy:.4f}")
            
            # 교차 검증 결과 요약
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            
            print(f"\n=== 교차 검증 결과 ===")
            print(f"평균 정확도: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"최고 정확도: {max(fold_accuracies):.4f}")
            print(f"최저 정확도: {min(fold_accuracies):.4f}")
            
            # 결과 저장
            results = {
                'fold_accuracies': fold_accuracies,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'best_accuracy': max(fold_accuracies),
                'worst_accuracy': min(fold_accuracies)
            }
            
            import json
            with open(f"{config['save_dir']}/cross_validation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"교차 검증 결과 저장: {config['save_dir']}/cross_validation_results.json")
            
            # 가장 좋은 성능의 모델을 메인 모델로 설정
            best_fold_idx = np.argmax(fold_accuracies)
            trainer = fold_trainers[best_fold_idx]
            best_accuracy = fold_accuracies[best_fold_idx]
            
            print(f"최고 성능 모델: Fold {best_fold_idx + 1} (정확도: {best_accuracy:.4f})")
            
        else:
            # 단일 폴드 학습
            trainer, best_accuracy = run_training(train_loader, val_loader, config)
    
    # 4단계: 모델 평가
    if not args.skip_evaluation:
        meta_df = pd.read_csv("data/meta.csv")
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