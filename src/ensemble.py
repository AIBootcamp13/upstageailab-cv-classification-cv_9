import pandas as pd
import numpy as np
import argparse
import os
from glob import glob
from config import CFG

def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple models using soft voting.")
    parser.add_argument(
        '--probs_dir', type=str, default='.',
        help="Directory containing the probability CSV files. Defaults to the project root."
    )
    parser.add_argument(
        '--file_pattern', type=str, default='probs_submission_*.csv',
        help="Pattern to find probability CSV files."
    )
    parser.add_argument(
        '--output_file', type=str, default='submission_ensemble.csv',
        help="Name of the final submission file."
    )
    args = parser.parse_args()

    # 지정된 패턴으로 확률 예측 파일들을 찾습니다.
    prob_files = glob(os.path.join(args.probs_dir, args.file_pattern))

    if len(prob_files) < 2:
        print(f"오류: 앙상블을 위해 최소 2개 이상의 확률 예측 파일이 필요합니다. 현재 {len(prob_files)}개 발견됨.")
        return

    print(f"{len(prob_files)}개의 파일을 앙상블합니다:")
    for f in prob_files:
        print(f" - {os.path.basename(f)}")

    # 첫 번째 파일을 기준으로 초기화합니다.
    ensemble_probs = pd.read_csv(prob_files[0]).sort_values('ID').reset_index(drop=True)
    prob_columns = [col for col in ensemble_probs.columns if col.startswith('prob_')]

    # 나머지 파일들을 순회하며 확률을 더합니다.
    for i in range(1, len(prob_files)):
        df_prob = pd.read_csv(prob_files[i]).sort_values('ID').reset_index(drop=True)
        ensemble_probs[prob_columns] += df_prob[prob_columns]

    # 합산된 확률에서 가장 높은 값의 인덱스(클래스)를 최종 예측으로 선택합니다.
    final_preds = np.argmax(ensemble_probs[prob_columns].values, axis=1)

    # 최종 제출 파일 생성
    submission_df = pd.read_csv(CFG.SAMPLE_SUBMISSION_PATH)
    submission_df['target'] = final_preds
    submission_df.to_csv(args.output_file, index=False)
    print(f"\n앙상블 완료! 최종 제출 파일 '{args.output_file}'이 생성되었습니다.")

if __name__ == "__main__":
    main()