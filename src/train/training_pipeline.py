#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 분류 학습 파이프라인
EDA 결과를 바탕으로 한 체계적인 학습 시스템
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

# 커스텀 모듈 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..model.model_architecture import get_model, get_loss_function, count_parameters

class Trainer:
    """모델 학습 클래스"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            device: 학습 디바이스
            config: 학습 설정
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 옵티마이저 설정
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        # 학습률 스케줄러
        if config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['epochs']
            )
        elif config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config['step_size'], gamma=config['gamma']
            )
        elif config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=config['patience'], factor=0.5
            )
        else:
            self.scheduler = None
        
        # 손실 함수
        self.criterion = get_loss_function(config['loss_function'], config['num_classes'])
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        # 조기 종료
        self.patience = config.get('early_stopping_patience', 10)
        self.counter = 0
        
        # 모델 저장 경로
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # wandb 초기화
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """wandb 초기화"""
        project_name = self.config.get('wandb_project', 'image-classification')
        run_name = self.config.get('wandb_run_name', f"{self.config['model_name']}_{self.config['optimizer']}")
        
        wandb.init(
            project=project_name,
            name=run_name,
            config=self.config,
            tags=[self.config['model_name'], self.config['optimizer'], self.config['loss_function']]
        )
        
        # 모델 아키텍처 로깅
        wandb.watch(self.model, log="all", log_freq=100)
        
        print(f"wandb 초기화 완료: {project_name}/{run_name}")
    
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 그래디언트 클리핑
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 진행률 출력
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f'Batch [{batch_idx}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
                
                # wandb 배치 로깅
                if self.use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_accuracy': 100. * correct / total,
                        'batch': batch_idx
                    })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self):
        """전체 학습 과정"""
        print("=== 학습 시작 ===")
        print(f"모델: {self.config['model_name']}")
        print(f"파라미터 수: {count_parameters(self.model):,}")
        print(f"학습 데이터: {len(self.train_loader.dataset)}")
        print(f"검증 데이터: {len(self.val_loader.dataset)}")
        print(f"배치 크기: {self.config['batch_size']}")
        print(f"에포크: {self.config['epochs']}")
        print(f"학습률: {self.config['learning_rate']}")
        print(f"옵티마이저: {self.config['optimizer']}")
        print(f"손실 함수: {self.config['loss_function']}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc, predictions, targets = self.validate_epoch()
            
            # 학습률 조정
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 기록 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # wandb 에포크 로깅
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                })
            
            # 결과 출력
            epoch_time = time.time() - epoch_start
            print(f'Epoch [{epoch+1}/{self.config["epochs"]}] ({epoch_time:.1f}s)')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            print("-" * 30)
            
            # 최고 성능 모델 저장
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.save_model('best_model.pth')
                
                # wandb 최고 성능 로깅
                if self.use_wandb:
                    wandb.log({
                        'best_val_accuracy': val_acc,
                        'best_epoch': epoch + 1
                    })
                
                self.counter = 0
            else:
                self.counter += 1
            
            # 조기 종료 체크
            if self.counter >= self.patience:
                print(f"조기 종료: {self.patience} 에포크 동안 성능 향상 없음")
                break
        
        total_time = time.time() - start_time
        print(f"\n=== 학습 완료 ===")
        print(f"총 학습 시간: {total_time/3600:.2f}시간")
        print(f"최고 검증 정확도: {self.best_val_accuracy:.2f}% (에포크 {self.best_epoch+1})")
        
        # 최종 모델 저장
        self.save_model('final_model.pth')
        
        # 학습 곡선 저장
        self.plot_training_curves()
        
        # wandb 종료
        if self.use_wandb:
            wandb.finish()
        
        return self.best_val_accuracy
    
    def save_model(self, filename):
        """모델 저장"""
        save_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_accuracy': self.best_val_accuracy,
            'epoch': self.best_epoch
        }, save_path)
        print(f"모델 저장: {save_path}")
    
    def plot_training_curves(self):
        """학습 곡선 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 손실 곡선
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 정확도 곡선
        axes[0, 1].plot(self.train_accuracies, label='Train Acc')
        axes[0, 1].plot(self.val_accuracies, label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 학습률 곡선
        if self.scheduler:
            lrs = []
            for param_group in self.optimizer.param_groups:
                lrs.append(param_group['lr'])
            axes[1, 0].plot(lrs)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # 손실 vs 정확도
        axes[1, 1].scatter(self.train_losses, self.train_accuracies, 
                          alpha=0.6, label='Train', s=20)
        axes[1, 1].scatter(self.val_losses, self.val_accuracies, 
                          alpha=0.6, label='Val', s=20)
        axes[1, 1].set_title('Loss vs Accuracy')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, model, test_loader, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
    
    def evaluate(self):
        """모델 평가"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_targets, all_probabilities
    
    def generate_report(self, predictions, targets, probabilities, save_dir):
        """평가 리포트 생성"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 분류 리포트
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # 혼동 행렬
        cm = confusion_matrix(targets, predictions)
        
        # 시각화
        self.plot_confusion_matrix(cm, self.class_names, save_dir)
        self.plot_class_accuracy(report, save_dir)
        
        # 결과 저장
        with open(save_dir / 'evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def plot_confusion_matrix(self, cm, class_names, save_dir):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_accuracy(self, report, save_dir):
        """클래스별 정확도 시각화"""
        class_accuracies = []
        class_names = []
        
        for class_name in report.keys():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_accuracies.append(report[class_name]['precision'])
                class_names.append(class_name)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(class_names)), class_accuracies)
        plt.title('Class-wise Precision')
        plt.xlabel('Classes')
        plt.ylabel('Precision')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        
        # 값 표시
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()

def get_default_config():
    """기본 학습 설정"""
    return {
        'model_name': 'efficientnet_b0',
        'num_classes': 17,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'loss_function': 'cross_entropy',
        'weight_decay': 1e-4,
        'gradient_clip': 1.0,
        'early_stopping_patience': 10,
        'save_dir': 'models',
        'log_interval': 10,
        # wandb 설정
        'use_wandb': False
        'wandb_project': 'cv-team9',
        'wandb_run_name': None  # None이면 자동 생성
    }

def main():
    """메인 함수 - 학습 파이프라인 실행"""
    
    # 설정
    config = get_default_config()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 로드
    # data_path = "data/train"
    # train_csv_path = "data/train.csv"
    # meta_csv_path = "data/meta.csv"
    
    # train_df = pd.read_csv(train_csv_path)
    # meta_df = pd.read_csv(meta_csv_path)
    
    # 데이터 로더 생성 (train 폴더의 모든 이미지 사용)
    from ..data.data_preprocessing import preprocess_data
    train_loader, val_loader = preprocess_data()
    
    # 모델 생성
    model = get_model(config['model_name'], config['num_classes'], pretrained=True)
    
    # 학습기 생성
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # 학습 실행
    best_accuracy = trainer.train()
    
    print(f"최종 결과: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 