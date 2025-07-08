import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import argparse
import os

from config import CFG
from dataset import DocumentDataset
from augmentation import get_valid_transforms

def main():
    parser = argparse.ArgumentParser(description="Inference for document classification.")
    parser.add_argument(
        '--model', type=str, default=CFG.DEFAULT_MODEL,
        help=f"Model name from timm library. Default: {CFG.DEFAULT_MODEL}"
    )
    parser.add_argument(
        '--img-size', type=int, default=CFG.IMG_SIZE,
        help=f"Image size used during training. Default: {CFG.IMG_SIZE}"
    )
    parser.add_argument(
        '--batch-size', type=int, default=CFG.BATCH_SIZE,
        help=f"Batch size for inference. Default: {CFG.BATCH_SIZE}"
    )
    parser.add_argument(
        '--fold', type=int, required=True,
        help="Fold number to use for inference (0-4). This is a required argument."
    )
    args = parser.parse_args()
    model_name = args.model
    img_size = args.img_size
    batch_size = args.batch_size
    fold = args.fold
    print(f"======== Starting inference ========")
    print(f"Fold: {fold}, Model: {model_name}, Image Size: {img_size}, Batch Size: {batch_size}")

    # 테스트 데이터 로드
    df_test = pd.read_csv(CFG.SAMPLE_SUBMISSION_PATH)

    # --- Dataset and DataLoader ---
    test_dataset = DocumentDataset(df_test, CFG.TEST_IMG_PATH, transforms=get_valid_transforms(img_size), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Load Model ---
    # K-Fold 학습으로 생성된 fold별 모델을 불러옵니다.
    model_path = os.path.join(CFG.MODEL_SAVE_DIR, f'best_model_{model_name}_sz{img_size}_fold{fold}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    model = timm.create_model(model_name, pretrained=False, num_classes=CFG.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
    model.to(CFG.DEVICE)

    # --- 모델 평가 모드 설정 ---
    # Test-Time Augmentation을 사용하지 않으므로, 모델을 바로 평가 모드로 설정합니다.
    model.eval()

    # --- Inference ---
    # 소프트 보팅 앙상블을 위해 클래스별 확률을 저장합니다.
    all_probs = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(CFG.DEVICE)
            outputs = model(images)
            
            # 모델의 출력(logits)을 softmax를 통해 확률로 변환합니다.
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    # --- Create Submission File ---
    # 생성된 확률들을 데이터프레임으로 만듭니다.
    df_probs = pd.DataFrame(all_probs, columns=[f'prob_{i}' for i in range(CFG.NUM_CLASSES)])
    df_probs['ID'] = df_test['ID'] # 원본 ID와 매칭

    # 제출 파일 이름에 fold 번호를 포함시켜 덮어쓰기를 방지합니다.
    submission_filename = f'probs_submission_{model_name}_sz{img_size}_fold{fold}.csv'
    df_probs.to_csv(submission_filename, index=False)

    print(f"Inference complete and probability file '{submission_filename}' created!")

if __name__ == "__main__":
    main()
