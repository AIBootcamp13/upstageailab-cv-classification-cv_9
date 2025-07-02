#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 분류 모델 아키텍처
EDA 결과를 바탕으로 한 효율적인 모델 설계
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional, List

class CustomCNN(nn.Module):
    """커스텀 CNN 모델 - 문서 이미지 분류에 최적화"""
    
    def __init__(self, num_classes=17, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        
        # 특징 추출기 (문서 이미지에 최적화)
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 두 번째 블록
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 세 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 네 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EfficientNetModel(nn.Module):
    """EfficientNet 기반 모델 - 효율적인 문서 분류"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=17, pretrained=True):
        super(EfficientNetModel, self).__init__()
        
        # EfficientNet 백본
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # 분류기 제거
        )
        
        # 특징 차원 가져오기
        feature_dim = self.backbone.num_features
        
        # 문서 분류를 위한 커스텀 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class ResNetModel(nn.Module):
    """ResNet 기반 모델 - 안정적인 문서 분류"""
    
    def __init__(self, model_name='resnet50', num_classes=17, pretrained=True):
        super(ResNetModel, self).__init__()
        
        # ResNet 백본
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        # 분류기 교체
        self.backbone.fc = nn.Identity()
        
        # 문서 분류를 위한 커스텀 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class VisionTransformerModel(nn.Module):
    """Vision Transformer 기반 모델 - 최신 문서 분류"""
    
    def __init__(self, model_name='vit_base_patch16_224', num_classes=17, pretrained=True):
        super(VisionTransformerModel, self).__init__()
        
        # ViT 백본
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # 분류기 제거
        )
        
        # 특징 차원 가져오기
        feature_dim = self.backbone.num_features
        
        # 문서 분류를 위한 커스텀 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class EnsembleModel(nn.Module):
    """앙상블 모델 - 여러 모델의 예측을 결합"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            # 균등 가중치
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "가중치 수와 모델 수가 일치하지 않습니다."
            self.weights = weights
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 가중 평균
        ensemble_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            ensemble_output += weight * output
        
        return ensemble_output

class FocalLoss(nn.Module):
    """Focal Loss - 클래스 불균형 문제 해결"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss - 과적합 방지"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_one_hot = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -(targets_one_hot * log_probs).sum(dim=1).mean()
        return loss

def get_model(model_name='efficientnet_b0', num_classes=17, pretrained=True):
    """모델 팩토리 함수"""
    
    if model_name.startswith('efficientnet'):
        return EfficientNetModel(model_name, num_classes, pretrained)
    elif model_name.startswith('resnet'):
        return ResNetModel(model_name, num_classes, pretrained)
    elif model_name.startswith('vit'):
        return VisionTransformerModel(model_name, num_classes, pretrained)
    elif model_name == 'custom_cnn':
        return CustomCNN(num_classes)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

def get_loss_function(loss_name='cross_entropy', num_classes=17):
    """손실 함수 팩토리 함수"""
    
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'label_smoothing':
        return LabelSmoothingLoss(num_classes)
    else:
        raise ValueError(f"지원하지 않는 손실 함수: {loss_name}")

def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """모델 아키텍처 테스트"""
    
    print("=== 모델 아키텍처 테스트 ===")
    
    # 테스트할 모델들
    model_configs = [
        ('efficientnet_b0', EfficientNetModel),
        ('resnet50', ResNetModel),
        ('vit_base_patch16_224', VisionTransformerModel),
        ('custom_cnn', CustomCNN)
    ]
    
    for model_name, model_class in model_configs:
        try:
            if model_name == 'custom_cnn':
                model = model_class(num_classes=17)
            else:
                model = model_class(model_name, num_classes=17, pretrained=False)
            
            # 더미 입력으로 테스트
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            
            param_count = count_parameters(model)
            
            print(f"✅ {model_name}:")
            print(f"   - 출력 형태: {output.shape}")
            print(f"   - 파라미터 수: {param_count:,}")
            print()
            
        except Exception as e:
            print(f"❌ {model_name}: {e}")
            print()

if __name__ == "__main__":
    main() 