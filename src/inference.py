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
    args = parser.parse_args()
    model_name = args.model
    print(f"======== Starting inference for model: {model_name} ========")

    # 테스트 데이터 로드
    df_test = pd.read_csv(CFG.SAMPLE_SUBMISSION_PATH)

    # --- Dataset and DataLoader ---
    test_dataset = DocumentDataset(df_test, CFG.TEST_IMG_PATH, transforms=get_valid_transforms(CFG.IMG_SIZE), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Load Model ---
    # config.py에 정의된 경로에서 모델을 불러옵니다.
    model_path = os.path.join(CFG.MODEL_SAVE_DIR, f'best_model_{model_name}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    model = timm.create_model(model_name, pretrained=False, num_classes=CFG.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
    model.to(CFG.DEVICE)
    model.eval()

    # --- Inference ---
    all_preds = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.to(CFG.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # --- Create Submission File ---
    df_test['target'] = all_preds
    submission_filename = f'submission_{model_name}.csv'
    df_test.to_csv(submission_filename, index=False)

    print(f"Inference complete and {submission_filename} created!")

if __name__ == "__main__":
    main()