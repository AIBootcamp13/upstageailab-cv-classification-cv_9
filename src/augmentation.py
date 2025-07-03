import albumentations as A
from albumentations.pytorch import ToTensorV2

# This is the same as the validation transform, for cases where no augmentation is needed.
def get_base_transforms(img_size):
    """기본적인 리사이즈와 정규화만 적용합니다."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 가벼운 수준의 증강
def get_light_transforms(img_size):
    """가벼운 수준의 증강을 적용합니다."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 중간 수준의 증강
def get_medium_transforms(img_size):
    """중간 수준의 증강을 적용합니다. 테스트 데이터의 변형을 커버하기에 적합합니다."""
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 강한 수준의 증강
def get_heavy_transforms(img_size):
    """강한 수준의 증강을 적용하여 모델의 강건함을 극대화합니다."""
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.5),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10,
                        min_holes=1, fill_value=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 검증 데이터셋용 변환 (증강 없음)
def get_valid_transforms(img_size):
    """검증 및 테스트 데이터에 적용할 기본 변환입니다."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])