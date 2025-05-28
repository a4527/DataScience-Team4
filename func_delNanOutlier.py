# 데이터 프레임의 결측치와 이상치 제거 함수
def remove_missing_and_outliers_iqr(df):
    """
    IQR 방식을 사용하여 데이터 프레임의 결측치와 이상치를 제거"""
    # 1. 결측치 제거
    df_cleaned = df.dropna()

    # 2. 이상치 제거 (IQR 방식)
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치가 아닌 행만 남기기
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    return df_cleaned