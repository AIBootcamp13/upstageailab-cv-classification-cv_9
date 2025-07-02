#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 분류 EDA (Exploratory Data Analysis) - CSV 기반
데이터셋 분석, 시각화, 전처리 등을 포함한 종합적인 EDA 코드
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import random
from pathlib import Path
import warnings
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정"""
    import platform
    
    system = platform.system()
    
    # 방법 1: 시스템별 기본 폰트
    if system == "Windows":
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    elif system == "Darwin":
        plt.rcParams['font.family'] = 'AppleGothic'   # Apple Gothic
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'   # Linux 기본 폰트
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    # 방법 2: 폰트 매니저를 통한 한글 폰트 찾기
    try:
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = []
        for font in font_list:
            if any(keyword in font.lower() for keyword in ['gothic', 'malgun', 'apple', 'nanum', 'batang']):
                korean_fonts.append(font)
        
        if korean_fonts:
            plt.rcParams['font.family'] = korean_fonts[0]
            print(f"한글 폰트 설정 완료: {korean_fonts[0]}")
        else:
            print(f"기본 폰트 설정: {plt.rcParams['font.family']}")
    except:
        print(f"기본 폰트 설정: {plt.rcParams['font.family']}")

def set_english_labels():
    """영어 라벨 설정 (한글 폰트 문제가 지속될 경우)"""
    return {
        'account_number': 'Account Number',
        'application_for_payment_of_pregnancy_medical_expenses': 'Pregnancy Medical Payment',
        'car_dashboard': 'Car Dashboard',
        'confirmation_of_admission_and_discharge': 'Admission/Discharge Confirmation',
        'diagnosis': 'Diagnosis',
        'driver_lisence': 'Driver License',
        'medical_bill_receipts': 'Medical Bill Receipts',
        'medical_outpatient_certificate': 'Outpatient Certificate',
        'national_id_card': 'National ID Card',
        'passport': 'Passport',
        'payment_confirmation': 'Payment Confirmation',
        'pharmaceutical_receipt': 'Pharmaceutical Receipt',
        'prescription': 'Prescription',
        'resume': 'Resume',
        'statement_of_opinion': 'Statement of Opinion',
        'vehicle_registration_certificate': 'Vehicle Registration Certificate',
        'vehicle_registration_plate': 'Vehicle Registration Plate'
    }

# 한글 폰트 설정 실행
set_korean_font()

class ImageClassificationEDA:
    def __init__(self, data_path, train_csv_path, meta_csv_path, use_english_labels=False):
        """
        이미지 분류 EDA 클래스 초기화
        
        Args:
            data_path (str): 이미지 데이터셋 경로
            train_csv_path (str): train.csv 파일 경로
            meta_csv_path (str): meta.csv 파일 경로
            use_english_labels (bool): 영어 라벨 사용 여부
        """
        self.data_path = Path(data_path)
        self.train_csv_path = train_csv_path
        self.meta_csv_path = meta_csv_path
        self.use_english_labels = use_english_labels
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.dataset_info = {}
        self.class_distribution = {}
        self.train_df = None
        self.meta_df = None
        
        # 영어 라벨 매핑
        if self.use_english_labels:
            self.english_labels = set_english_labels()
        else:
            self.english_labels = {}
        
    def load_csv_data(self):
        """CSV 데이터 로드"""
        print("=== CSV 데이터 로드 ===")
        
        try:
            self.train_df = pd.read_csv(self.train_csv_path)
            self.meta_df = pd.read_csv(self.meta_csv_path)
            print(f"Train CSV 로드 완료: {len(self.train_df)}개 행")
            print(f"Meta CSV 로드 완료: {len(self.meta_df)}개 클래스")
            print(f"클래스 목록: {list(self.meta_df['class_name'])}")
            return True
        except Exception as e:
            print(f"CSV 로드 실패: {e}")
            return False
    
    def analyze_class_distribution_from_csv(self):
        """CSV에서 클래스 분포 분석"""
        print("\n=== 클래스 분포 분석 (CSV 기반) ===")
        
        if self.train_df is None or self.meta_df is None:
            print("CSV 데이터가 로드되지 않았습니다.")
            return {}, []
        
        # 클래스별 이미지 수 계산
        class_counts = self.train_df['target'].value_counts().to_dict()
        
        # 클래스 이름 매핑
        class_name_mapping = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
        
        # 영어 라벨 사용 여부에 따라 클래스 이름 변환
        if self.use_english_labels:
            class_counts_named = {}
            for k, v in class_counts.items():
                original_name = class_name_mapping.get(k, f'class_{k}')
                english_name = self.english_labels.get(original_name, original_name)
                class_counts_named[english_name] = v
        else:
            class_counts_named = {class_name_mapping.get(k, f'class_{k}'): v for k, v in class_counts.items()}
        
        # 이미지 경로 생성
        image_paths = []
        for _, row in self.train_df.iterrows():
            image_path = self.data_path / row['ID']
            if image_path.exists():
                original_class_name = class_name_mapping.get(row['target'], f'class_{row["target"]}')
                if self.use_english_labels:
                    display_name = self.english_labels.get(original_class_name, original_class_name)
                else:
                    display_name = original_class_name
                image_paths.append((display_name, str(image_path)))
        
        self.class_distribution = class_counts_named
        self.dataset_info = {
            'total_images': len(self.train_df),
            'num_classes': len(class_counts),
            'image_paths': image_paths
        }
        
        print(f"총 이미지 수: {len(self.train_df):,}")
        print(f"클래스 수: {len(class_counts)}")
        print(f"평균 클래스당 이미지 수: {len(self.train_df)/len(class_counts):.1f}")
        
        return class_counts_named, image_paths
    
    def visualize_class_distribution(self, class_counts):
        """클래스 분포 시각화"""
        print("\n=== 클래스 분포 시각화 ===")
        
        if not class_counts:
            print("시각화할 클래스 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 막대 그래프
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        axes[0, 0].bar(range(len(classes)), counts)
        axes[0, 0].set_title('클래스별 이미지 수', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('클래스')
        axes[0, 0].set_ylabel('이미지 수')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 파이 차트
        axes[0, 1].pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('클래스별 비율', fontsize=14, fontweight='bold')
        
        # 3. 분포 통계
        axes[1, 0].hist(counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('클래스별 이미지 수 분포', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('이미지 수')
        axes[1, 0].set_ylabel('클래스 수')
        
        # 4. 박스플롯
        axes[1, 1].boxplot(counts)
        axes[1, 1].set_title('클래스별 이미지 수 박스플롯', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('이미지 수')
        
        plt.tight_layout()
        plt.show()
        
        # 통계 정보 출력
        if counts:
            print(f"최대 이미지 수: {max(counts)}")
            print(f"최소 이미지 수: {min(counts)}")
            print(f"중앙값: {np.median(counts):.1f}")
            print(f"표준편차: {np.std(counts):.1f}")
        else:
            print("통계를 계산할 데이터가 없습니다.")
    
    def analyze_image_characteristics(self, image_paths, sample_size=100):
        """이미지 특성 분석"""
        print(f"\n=== 이미지 특성 분석 (샘플 크기: {sample_size}) ===")
        
        if not image_paths:
            print("분석할 이미지가 없습니다.")
            return {'widths': [], 'heights': [], 'sizes': [], 'formats': []}
        
        # 랜덤 샘플링
        sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
        
        widths, heights, sizes, formats = [], [], [], []
        
        for class_name, img_path in sample_paths:
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    sizes.append(os.path.getsize(img_path))
                    formats.append(img.format)
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
        
        # 통계 계산
        stats = {
            'widths': widths,
            'heights': heights,
            'sizes': sizes,
            'formats': formats
        }
        
        print(f"분석된 이미지 수: {len(widths)}")
        if widths:
            print(f"평균 너비: {np.mean(widths):.1f}px")
            print(f"평균 높이: {np.mean(heights):.1f}px")
            print(f"평균 파일 크기: {np.mean(sizes)/1024:.1f}KB")
            print(f"이미지 포맷: {Counter(formats)}")
        else:
            print("분석할 이미지가 없습니다.")
        
        return stats
    
    def visualize_image_characteristics(self, stats):
        """이미지 특성 시각화"""
        print("\n=== 이미지 특성 시각화 ===")
        
        if not stats['widths']:
            print("시각화할 이미지 데이터가 없습니다.")
            return []
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 너비 분포
        axes[0, 0].hist(stats['widths'], bins=30, alpha=0.7, color='lightcoral')
        axes[0, 0].set_title('이미지 너비 분포', fontweight='bold')
        axes[0, 0].set_xlabel('너비 (px)')
        axes[0, 0].set_ylabel('빈도')
        
        # 2. 높이 분포
        axes[0, 1].hist(stats['heights'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('이미지 높이 분포', fontweight='bold')
        axes[0, 1].set_xlabel('높이 (px)')
        axes[0, 1].set_ylabel('빈도')
        
        # 3. 파일 크기 분포
        axes[0, 2].hist(np.array(stats['sizes'])/1024, bins=30, alpha=0.7, color='lightblue')
        axes[0, 2].set_title('파일 크기 분포', fontweight='bold')
        axes[0, 2].set_xlabel('크기 (KB)')
        axes[0, 2].set_ylabel('빈도')
        
        # 4. 너비 vs 높이 산점도
        axes[1, 0].scatter(stats['widths'], stats['heights'], alpha=0.6)
        axes[1, 0].set_title('너비 vs 높이', fontweight='bold')
        axes[1, 0].set_xlabel('너비 (px)')
        axes[1, 0].set_ylabel('높이 (px)')
        
        # 5. 종횡비 분포
        aspect_ratios = [w/h for w, h in zip(stats['widths'], stats['heights'])]
        axes[1, 1].hist(aspect_ratios, bins=30, alpha=0.7, color='gold')
        axes[1, 1].set_title('종횡비 분포', fontweight='bold')
        axes[1, 1].set_xlabel('종횡비 (너비/높이)')
        axes[1, 1].set_ylabel('빈도')
        
        # 6. 이미지 포맷 분포
        format_counts = Counter(stats['formats'])
        axes[1, 2].bar(format_counts.keys(), format_counts.values())
        axes[1, 2].set_title('이미지 포맷 분포', fontweight='bold')
        axes[1, 2].set_xlabel('포맷')
        axes[1, 2].set_ylabel('개수')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return aspect_ratios
    
    def display_sample_images(self, image_paths, samples_per_class=3):
        """샘플 이미지 표시"""
        print(f"\n=== 샘플 이미지 표시 (클래스당 {samples_per_class}개) ===")
        
        if not image_paths:
            print("표시할 이미지가 없습니다.")
            return
        
        # 클래스별로 샘플 선택
        class_samples = {}
        for class_name, img_path in image_paths:
            if class_name not in class_samples:
                class_samples[class_name] = []
            if len(class_samples[class_name]) < samples_per_class:
                class_samples[class_name].append(img_path)
        
        if not class_samples:
            print("샘플 이미지를 찾을 수 없습니다.")
            return
        
        # 이미지 표시
        num_classes = len(class_samples)
        fig, axes = plt.subplots(samples_per_class, num_classes, 
                                figsize=(3*num_classes, 3*samples_per_class))
        
        if num_classes == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (class_name, samples) in enumerate(class_samples.items()):
            for j, img_path in enumerate(samples):
                try:
                    img = Image.open(img_path)
                    if num_classes == 1:
                        axes[j].imshow(img)
                        axes[j].set_title(f'{class_name}\n{img.size}', fontsize=10)
                        axes[j].axis('off')
                    else:
                        axes[j, i].imshow(img)
                        axes[j, i].set_title(f'{class_name}\n{img.size}', fontsize=10)
                        axes[j, i].axis('off')
                except Exception as e:
                    print(f"이미지 로드 실패: {img_path} - {e}")
        
        plt.tight_layout()
        plt.show()
    
    def analyze_class_balance(self, class_counts):
        """클래스 불균형 분석"""
        print("\n=== 클래스 불균형 분석 ===")
        
        if not class_counts:
            print("분석할 클래스가 없습니다.")
            return 0, 0, "데이터 없음"
        
        counts = list(class_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # 불균형 지수 계산
        imbalance_ratio = max(counts) / min(counts)
        cv = std_count / mean_count  # 변동계수
        
        print(f"불균형 비율 (최대/최소): {imbalance_ratio:.2f}")
        print(f"변동계수: {cv:.2f}")
        
        # 불균형 정도 평가
        if imbalance_ratio < 2:
            balance_level = "균형"
        elif imbalance_ratio < 5:
            balance_level = "약간 불균형"
        elif imbalance_ratio < 10:
            balance_level = "불균형"
        else:
            balance_level = "심각한 불균형"
        
        print(f"클래스 불균형 정도: {balance_level}")
        
        return imbalance_ratio, cv, balance_level
    
    def analyze_target_distribution(self):
        """타겟 분포 상세 분석"""
        print("\n=== 타겟 분포 상세 분석 ===")
        
        if self.train_df is None or self.meta_df is None:
            print("CSV 데이터가 로드되지 않았습니다.")
            return
        
        # 타겟 분포
        target_counts = self.train_df['target'].value_counts().sort_index()
        
        # 클래스 이름과 매핑
        class_name_mapping = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
        
        print("클래스별 상세 분포:")
        print("-" * 60)
        print(f"{'Target':<5} {'Class Name':<40} {'Count':<10} {'Percentage':<10}")
        print("-" * 60)
        
        total = len(self.train_df)
        for target, count in target_counts.items():
            class_name = class_name_mapping.get(target, f'Unknown_{target}')
            percentage = (count / total) * 100
            print(f"{target:<5} {class_name:<40} {count:<10} {percentage:.2f}%")
        
        print("-" * 60)
        print(f"총 이미지 수: {total}")
        print(f"클래스 수: {len(target_counts)}")
        
        return target_counts
    
    def generate_eda_report(self):
        """전체 EDA 리포트 생성"""
        print("=" * 50)
        print("이미지 분류 EDA 리포트 (CSV 기반)")
        print("=" * 50)
        
        # 1. CSV 데이터 로드
        if not self.load_csv_data():
            return
        
        # 2. 타겟 분포 상세 분석
        target_counts = self.analyze_target_distribution()
        
        # 3. 클래스 분포 분석
        class_counts, image_paths = self.analyze_class_distribution_from_csv()
        
        if not class_counts:
            print("분석할 데이터가 없습니다. 데이터셋 경로를 확인해주세요.")
            return
        
        # 4. 클래스 분포 시각화
        self.visualize_class_distribution(class_counts)
        
        # 5. 이미지 특성 분석
        stats = self.analyze_image_characteristics(image_paths)
        
        # 6. 이미지 특성 시각화
        aspect_ratios = self.visualize_image_characteristics(stats)
        
        # 7. 샘플 이미지 표시
        self.display_sample_images(image_paths)
        
        # 8. 클래스 불균형 분석
        imbalance_ratio, cv, balance_level = self.analyze_class_balance(class_counts)
        
        # 9. 요약 리포트
        print("\n" + "=" * 50)
        print("EDA 요약 리포트")
        print("=" * 50)
        print(f"📊 데이터셋 크기: {self.dataset_info['total_images']:,}개 이미지")
        print(f"🏷️  클래스 수: {self.dataset_info['num_classes']}개")
        
        if stats['widths']:
            print(f"📏 평균 이미지 크기: {np.mean(stats['widths']):.0f}x{np.mean(stats['heights']):.0f}px")
            print(f"💾 평균 파일 크기: {np.mean(stats['sizes'])/1024:.1f}KB")
        else:
            print("📏 이미지 크기 정보: 없음")
            print("💾 파일 크기 정보: 없음")
            
        print(f"⚖️  클래스 불균형: {balance_level} (비율: {imbalance_ratio:.2f})")
        print(f"📈 변동계수: {cv:.2f}")
        
        # 권장사항
        print("\n💡 권장사항:")
        if imbalance_ratio > 5:
            print("- 클래스 불균형이 심각합니다. 데이터 증강이나 샘플링 전략을 고려하세요.")
        if stats['widths'] and (np.mean(stats['widths']) > 1000 or np.mean(stats['heights']) > 1000):
            print("- 이미지 크기가 큽니다. 리사이징을 고려하세요.")
        if stats['formats'] and len(set(stats['formats'])) > 3:
            print("- 다양한 이미지 포맷이 있습니다. 통일된 포맷으로 변환을 고려하세요.")

def main():
    """메인 함수 - 예제 실행"""
    
    # 데이터셋 경로 설정
    data_path = "data/train"  # 이미지 폴더
    train_csv_path = "data/train.csv"  # train.csv 파일
    meta_csv_path = "data/meta.csv"    # meta.csv 파일
    
    if not os.path.exists(data_path):
        print(f"데이터셋 경로가 존재하지 않습니다: {data_path}")
        print("실제 데이터셋 경로를 지정해주세요.")
        return
    
    if not os.path.exists(train_csv_path):
        print(f"train.csv 파일이 존재하지 않습니다: {train_csv_path}")
        return
        
    if not os.path.exists(meta_csv_path):
        print(f"meta.csv 파일이 존재하지 않습니다: {meta_csv_path}")
        return
    
    # 한글 폰트 문제 해결 시도
    print("한글 폰트 설정 확인 중...")
    try:
        # 간단한 한글 테스트
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '한글 테스트', ha='center', va='center', fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.close(fig)
        print("한글 폰트 설정 성공!")
        use_english_labels = False
    except:
        print("한글 폰트 설정 실패. 영어 라벨을 사용합니다.")
        use_english_labels = True
    
    # EDA 실행
    eda = ImageClassificationEDA(data_path, train_csv_path, meta_csv_path, use_english_labels=use_english_labels)
    eda.generate_eda_report()

if __name__ == "__main__":
    main()
