import cv2
import torch
from torch.utils.data import Dataset
import os

class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['ID']
        # os.path.join을 사용하여 플랫폼에 독립적인 경로를 생성합니다.
        img_path = os.path.join(self.img_dir, img_id)
        
        # Read the image
        # cv2 reads images in BGR format, so we convert it to RGB
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']

        if self.is_test:
            return img
        else:
            label = self.df.iloc[idx]['target']
            return img, torch.tensor(label, dtype=torch.long)