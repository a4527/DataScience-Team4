# 스케일링이 잘 되었는지 확인하기 위한 함수

import pandas as pd

def get_preprocessed_dataframe(pipeline, X_processed, numeric_cols, categorical_cols):
    """
    전처리된 X 데이터를 보기 쉬운 DataFrame으로 변환
    - pipeline: ColumnTransformer가 포함된 sklearn Pipeline
    - X_processed: fit_transform 또는 transform된 결과 (sparse or ndarray)
    - numeric_cols: 수치형 컬럼 이름 리스트
    - categorical_cols: 범주형 컬럼 이름 리스트
    """
    # 1. OneHot 인코딩된 컬럼 이름 얻기
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = encoder.get_feature_names_out(categorical_cols)

    # 2. 전체 컬럼명 구성
    all_column_names = numeric_cols + list(encoded_cat_names)

    # 3. 희소행렬이면 toarray()
    if hasattr(X_processed, "toarray"):
        X_array = X_processed.toarray()
    else:
        X_array = X_processed

    # 4. DataFrame 생성
    return pd.DataFrame(X_array, columns=all_column_names)