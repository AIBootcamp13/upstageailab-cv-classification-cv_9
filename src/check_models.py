import timm

# timm에서 사전 학습된 가중치를 제공하는 모든 모델의 목록을 가져옵니다.
available_models = timm.list_models(pretrained=True)

print(f"timm 라이브러리에서 사용 가능한 사전 학습 모델의 총 개수: {len(available_models)}")

# 'convnext' 키워드가 포함된 모델만 필터링해서 보기
print("\n--- 'ConvNeXt' 계열 모델 예시 ---")
convnext_models = [name for name in available_models if 'convnext' in name]
for model_name in convnext_models[:5]: # 상위 5개만 출력
    print(model_name)

# 'swin' 키워드가 포함된 모델만 필터링해서 보기
print("\n--- 'Swin Transformer' 계열 모델 예시 ---")
swin_models = [name for name in available_models if 'swin' in name]
for model_name in swin_models[:5]: # 상위 5개만 출력
    print(model_name)

# 'vit' (Vision Transformer) 키워드가 포함된 모델만 필터링해서 보기
print("\n--- 'Vision Transformer' 계열 모델 예시 ---")
vit_models = [name for name in available_models if 'vit' in name]
for model_name in vit_models[:5]: # 상위 5개만 출력
    print(model_name)
