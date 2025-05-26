import random
import numpy as np
import pandas as pd
# 결측치 및 이상치를 삽입하는 함수 
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
    ###결측치와 이상치를 삽입하는 과정에서 sample() 함수를 사용해 무작위로 행을 선택하고, 이상치로는 평균에서 크게 벗어난 값을 무작위로 삽입한다.
    ###이때 시드를 고정하지 않으면 실행할 때마다 삽입되는 위치와 값이 달라져 결과 비교가 어려워집니다.
    ###따라서 random.seed(seed)와 np.random.seed(seed)를 함께 설정하여, 샘플링과 이상치 값 선택이 항상 동일하게 반복되도록 설정
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
            row[col] = random.choice([col_mean + 4 * col_std, col_mean - 4 * col_std])
            new_rows.append(row)

    df_extra = pd.DataFrame(new_rows, columns=df.columns)
    df_result = pd.concat([df_result, df_extra], ignore_index=True)

    return df_result