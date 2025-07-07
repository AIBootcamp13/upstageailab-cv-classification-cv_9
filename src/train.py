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
    parser.add_argument(
        '--img-size', type=int, default=CFG.IMG_SIZE,
        help=f"Image size for training. Default: {CFG.IMG_SIZE}"
    )
    parser.add_argument(
        '--batch-size', type=int, default=CFG.BATCH_SIZE,
        help=f"Batch size for training. Default: {CFG.BATCH_SIZE}"
    )
    parser.add_argument(
        '--epochs', type=int, default=CFG.EPOCHS,
        help=f"Number of epochs to train. Default: {CFG.EPOCHS}"
    )
    parser.add_argument(
        '--patience', type=int, default=CFG.PATIENCE,
        help=f"Early stopping patience. Default: {CFG.PATIENCE}"
    )
    args = parser.parse_args()
    model_name = args.model
    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    print(f"======== Starting training ========")
    print(f"Model: {model_name}, Image Size: {img_size}, Batch Size: {batch_size}, Epochs: {epochs}, Patience: {patience}")

    # 데이터 로드
    df_train = pd.read_csv(CFG.TRAIN_CSV_PATH)

    # --- Stratified K-Fold 분할 ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # K-Fold의 모든 분할에 대해 학습을 실행하려면 아래 for 루프의 주석을 해제하세요.
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
        print(f"\n=========== Fold {fold+1} ===========")
        train_df = df_train.iloc[train_idx]
        valid_df = df_train.iloc[val_idx]
        # 이 예제에서는 첫 번째 fold에 대해서만 학습을 진행합니다.
        break

    # --- 데이터셋 및 데이터로더 ---
    # 오프라인 증강을 사용했으므로, 온라인에서는 가벼운 증강만 적용하거나 적용하지 않습니다.
    train_dataset = DocumentDataset(train_df, CFG.TRAIN_IMG_PATH, transforms=get_light_transforms(img_size))
    valid_dataset = DocumentDataset(valid_df, CFG.TRAIN_IMG_PATH, transforms=get_valid_transforms(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 모델, 손실 함수, 옵티마이저 ---
    model = timm.create_model(model_name, pretrained=True, num_classes=CFG.NUM_CLASSES)
    model.to(CFG.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)

    # --- 학습률 스케줄러 추가 ---
    # CosineAnnealingLR: 에포크에 따라 코사인 곡선을 그리며 학습률을 점차 감소시킵니다.
    # T_max: 학습률이 최소치에 도달하기까지 걸리는 에포크 수 (총 에포크 수)
    # eta_min: 도달 가능한 최소 학습률 (0에 가까운 작은 값)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- 학습 루프 ---
    best_f1 = 0.0
    patience_counter = 0
    # config.py에 정의된 모델 저장 경로를 사용합니다.
    model_save_dir = CFG.MODEL_SAVE_DIR
    os.makedirs(model_save_dir, exist_ok=True)
    # 모델 파일 이름에 모델명과 이미지 크기를 포함시켜 구별이 용이하게 합니다.
    model_save_path = os.path.join(model_save_dir, f"best_model_{model_name}_sz{img_size}.pth")

    for epoch in range(epochs):
        print(f"--- 에포크 {epoch+1}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.DEVICE)
        val_loss, macro_f1 = validate_one_epoch(model, valid_loader, criterion, CFG.DEVICE)

        print(f"학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}, Macro F1: {macro_f1:.4f}, 현재 LR: {optimizer.param_groups[0]['lr']:.6f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0  # F1 스코어가 개선되면 patience 카운터를 리셋합니다.
            print(f"New best model found! Saving to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            print(f"F1 스코어가 {patience_counter} 에포크 동안 개선되지 않았습니다.")
        
        # 에포크가 끝날 때마다 스케줄러를 업데이트하여 학습률을 조절합니다.
        scheduler.step()

        # Early Stopping 조건 확인
        if patience_counter >= patience:
            print(f"Early stopping: F1 스코어가 {patience} 에포크 동안 개선되지 않아 학습을 조기 종료합니다.")
            break


if __name__ == "__main__":
    main()