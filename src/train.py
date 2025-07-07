import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import timm
from tqdm import tqdm
import os
import argparse

from config import CFG
from dataset import DocumentDataset
from augmentation import get_light_transforms, get_valid_transforms

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="학습 중"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_one_epoch(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="검증 중"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(valid_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, macro_f1

def main():
    parser = argparse.ArgumentParser(description="Train a model for document classification.")
    parser.add_argument(
        '--model', type=str, default=CFG.DEFAULT_MODEL,
        help=f"Model name from timm library. Default: {CFG.DEFAULT_MODEL}"
    )
    args = parser.parse_args()
    model_name = args.model
    print(f"======== Starting training for model: {model_name} ========")

    # 데이터 로드
    df_train = pd.read_csv(CFG.TRAIN_CSV_PATH)

    # --- Stratified K-Fold 분할 ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
        print(f"\n=========== Fold {fold+1} ===========")
        train_df = df_train.iloc[train_idx]
        valid_df = df_train.iloc[val_idx]
        # 이 예제에서는 첫 번째 fold에 대해서만 학습을 진행합니다.
        # break

    # --- 데이터셋 및 데이터로더 ---
    # 오프라인 증강을 사용했으므로, 온라인에서는 가벼운 증강만 적용하거나 적용하지 않습니다.
    train_dataset = DocumentDataset(train_df, CFG.TRAIN_IMG_PATH, transforms=get_light_transforms(CFG.IMG_SIZE))
    valid_dataset = DocumentDataset(valid_df, CFG.TRAIN_IMG_PATH, transforms=get_valid_transforms(CFG.IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 모델, 손실 함수, 옵티마이저 ---
    model = timm.create_model(model_name, pretrained=True, num_classes=CFG.NUM_CLASSES)
    model.to(CFG.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)

    # --- 학습률 스케줄러 추가 ---
    # CosineAnnealingLR: 에포크에 따라 코사인 곡선을 그리며 학습률을 점차 감소시킵니다.
    # T_max: 학습률이 최소치에 도달하기까지 걸리는 에포크 수 (총 에포크 수)
    # eta_min: 도달 가능한 최소 학습률 (0에 가까운 작은 값)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    # --- 학습 루프 ---
    best_f1 = 0.0
    # config.py에 정의된 모델 저장 경로를 사용합니다.
    model_save_dir = CFG.MODEL_SAVE_DIR
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"best_model_{model_name}.pth")

    for epoch in range(CFG.EPOCHS):
        print(f"--- 에포크 {epoch+1}/{CFG.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.DEVICE)
        val_loss, macro_f1 = validate_one_epoch(model, valid_loader, criterion, CFG.DEVICE)

        print(f"학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}, Macro F1: {macro_f1:.4f}, 현재 LR: {optimizer.param_groups[0]['lr']:.6f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            print(f"New best model found! Saving to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        
        # 에포크가 끝날 때마다 스케줄러를 업데이트하여 학습률을 조절합니다.
        scheduler.step()


if __name__ == "__main__":
    main()