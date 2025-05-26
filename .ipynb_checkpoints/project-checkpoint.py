import pandas as pd
import numpy as np
import random
df=pd.read_csv('power_marketing_dataset.csv')

def append_outliers_and_missing_by_row(df, n_missing=10, n_outliers=30, seed=42):
    """
    기존 행을 복사한 뒤, 해당 행의 수치형 컬럼 중 하나에 결측치 또는 이상치를 삽입해 추가
    
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
        n_missing (int): 컬럼당 결측치 포함 행 개수
        n_outliers (int): 컬럼당 이상치 포함 행 개수
        seed (int): 랜덤 시드
    
    Returns:
        pd.DataFrame: 결측치/이상치가 삽입된 행이 추가된 데이터프레임
    """
    random.seed(seed)
    np.random.seed(seed)
    
    df_result = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    new_rows = []

    # 결측치 삽입 행 생성
    for col in numeric_cols:
        for _ in range(n_missing):
            row = df.sample(1).iloc[0].copy()
            row[col] = np.nan
            new_rows.append(row)

    # 이상치 삽입 행 생성
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        for _ in range(n_outliers):
            row = df.sample(1).iloc[0].copy()
            row[col] = random.choice([col_mean + 5 * col_std, col_mean - 5 * col_std])
            new_rows.append(row)

    df_extra = pd.DataFrame(new_rows, columns=df.columns)
    df_result = pd.concat([df_result, df_extra], ignore_index=True)

    return df_result

print(df.describe)