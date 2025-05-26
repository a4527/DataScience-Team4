import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from func_insertNanOutlier import append_outliers_and_missing_by_row
from func_delNanOutlier import remove_missing_and_outliers_iqr
from func_getPreprocessedDataframe import get_preprocessed_dataframe
df=pd.read_csv('power_marketing_dataset.csv')

# 결측치와 이상치가 삽입된 데이터프레임 생성
df=append_outliers_and_missing_by_row(df)

# 결측치와 이상치가 삽입된 데이터 프레임을 csv 파일로 저장
df.to_csv('power_marketing_dataset_with_outliers_and_missing.csv',index=False)

# 데이터 프레임의 결측치 개수 확인
print("결측치 개수: ",df.isnull().sum().sum())

# 데이터 프레임의 결측치 시각화
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
missing_counts.plot(kind='barh', figsize=(8, 4), color='salmon')
plt.title("Number of Missing Values per Column")
plt.xlabel("Missing Count")
plt.ylabel("Column")
plt.show()

#데이터 프레임의 이상치 시각화
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].plot(kind='box', subplots=True, layout=(3, 3), figsize=(15, 10), sharex=False)
plt.tight_layout()
plt.show()

# 결측치와 이상치를 제거한 데이터 프레임 생성
df_cleaned=remove_missing_and_outliers_iqr(df)
df_cleaned.to_csv('power_marketing_dataset_cleaned.csv',index=False)

# 결측치와 이상치를 제거한 데이터 프레임의 결측치 개수 확인 
print("결측치 개수 (이상치 제거 후):",df_cleaned.isnull().sum().sum())

"""
이미 극단적인 이상치는 제거함 → RobustScaler 필요는 낮음
모델은 회귀/분류/클러스터링에 모두 사용 예정
값의 분포가 평균 중심으로 있고 음수도 존재함
-> standardScaler 사용하는 것이 적절한 것으로 판단됨 
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 불필요한 컬럼 제거
df_cleaned=df_cleaned.drop(columns=["User ID"])
# 타깃 변수 분리 
x=df_cleaned.iloc[:,:-1] # 마지막 열을 제외한 모든 열
y=df_cleaned.iloc[:,-1] # 마지막 열 (타깃 변수)
# 범주형 / 수치형 컬럼 분리
categorical_cols = x.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()

#전처리 파이프라인 구성
"""여러 개의 서로 다른 전처리 기법을 컬럼에 따라 자동으로 적용할 수 있도록 ColumnTransformer를 사용
수치형 변수와 범주형 변수에 서로 다른 변환을 적용"""
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

#전체 파이프라인 구성 (모델 제외 - 여기선 전처리까지만)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)])

# 학습/테스트 분할 , test_Size는 일반적인 0.2로 설정 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# 전처리 실행
x_train_processed = pipeline.fit_transform(x_train)
x_test_processed = pipeline.transform(x_test)

# 전처리가 잘 됐는지 확인하기 위한 용도 / 필요없으면 지워도 됨
x_train_df = get_preprocessed_dataframe(pipeline, x_train_processed, numeric_cols, categorical_cols)
print(x_train_df.head())







