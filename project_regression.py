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

##### 파생 변수 생성 #######
# 1인당 소비량 (소비량 / 가구 수)
#df['Consumption_per_Person'] = df['Monthly Consumption (kWh)'] / (df['Household Size'] + 1)

# 최대 소비 / 평균 소비 비율
#df['Peak_to_Avg_Consumption'] = df['Peak Consumption (kWh)'] / (df['Avg Consumption (kWh)'] + 1)

# 저녁 - 아침 소비 차이
df['Evening_Morning_Diff'] = df['Consumption by Time of Day (Evening)'] - df['Consumption by Time of Day (Morning)']

# 이진 변수: 높은 참여율 여부
df['High_Engagement'] = (df['Engagement Rate'] > 0.5).astype(int)

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
df[numeric_cols].plot(kind='box', subplots=True, layout=(5, 3), figsize=(15, 10), sharex=False)
plt.tight_layout()
plt.show()

# 결측치와 이상치를 제거한 데이터 프레임 생성
df_cleaned=remove_missing_and_outliers_iqr(df)
df_cleaned.to_csv('power_marketing_dataset_cleaned.csv',index=False)

# 변수별 분포 시각화 
df_cleaned.select_dtypes(include=['int64','float64']).hist(bins=20,figsize=(16,10))
plt.tight_layout()
plt.show()

# 결측치와 이상치를 제거한 데이터 프레임의 결측치 개수 확인 
print("결측치 개수 (이상치 제거 후):",df_cleaned.isnull().sum().sum())

"""
이미 극단적인 이상치는 제거함 → RobustScaler 필요는 낮음
모델은 회귀/분류/클러스터링에 모두 사용 예정
값의 분포가 평균 중심으로 있고 음수도 존재함
-> standardScaler 사용하는 것이 적절한 것으로 판단됨 
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 불필요한 컬럼 제거
df_cleaned=df_cleaned.drop(columns=["User ID"])
# 너무 과적합되어서 의심되는 누수 피처 제거 
df_cleaned=df_cleaned.drop(columns=["Monthly Consumption (kWh)"]) 
# 타깃 변수 분리 
target_col="Energy Usage Reduction (%)"
target_col2="Avg Consumption (kWh)"
x=df_cleaned.drop(columns=[target_col2]) # 타깃변수를 제외한 모든 열
y=df_cleaned[target_col2] # (타깃 변수)


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

################################################################################################################
################################################################################################################

# regression 모델 학습 및 평가 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#####################################
#1. 기초 모델 Linear Regression 학습 및 예측
LR_model=LinearRegression()
LR_model.fit(x_train_processed,y_train)
y_pred=LR_model.predict(x_test_processed)

# 평가 지표 계산
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(LR_model, x_train_processed, y_train, cv=5, scoring="r2").mean()

# 평가 결과 출력
print("Linear Regression Results:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Cross-Validated R2:", cv_r2)
""" 결과 
Linear Regression Results:
MAE: 4.506577625388453   예측값이 실제값과 평균적으로 약 4.51만큼 차이남 
MSE: 29.8461662657935    오차를 제곱한 평균으로 큰 오차에 더 민감, MAE보다 크다는 것을 일부 큰 오차가 존재할 수 있다는뜻뜻
RMSE: 5.463164491921646  MSE에 루트를 씌워 실제 단위로 환산, 한 예측당 평균적으로 약 5.46만큼 오차 발생, 
R2 Score: 0.4504609051386237  전체 데이터 변동성 중 약 45%를 설명할 수 있다는 뜼, 성능이 낮다.
Cross-Validated R2: 0.37555152436414463 교차검증 결과 평균 설명력이 약 37.6%로 설명력이 부족
"""
###############################################
# 선형 회귀 모델의 성능을 높이기 위한 정규화 기법 적용 
#2. 정규화 회귀 모델 학습 및 예측 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# 평가 함수
def evaluate_model(name, model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid.fit(x_train_processed, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test_processed)

    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "CV R2": cross_val_score(best_model, x_train_processed, y_train, cv=5, scoring="r2").mean(),
        "Best Params": grid.best_params_
    }

# 정규화 모델 목록과 파라미터 세트
models = {
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1.0, 10.0]}),   # Ridge 회귀 모델은 L2 정규화를 적용하여 과적합을 방지
    "Lasso": (Lasso(), {"alpha": [0.01, 0.1, 1.0, 10.0]}),   # Lasso 회귀 모델은 L1 정규화를 적용하여 변수 선택과 과적합을 방지
    "ElasticNet": (ElasticNet(), {    # ElasticNet 회귀모델은 L1과 L2 정규화를 모두 적용하여 유연한 모델링 가능 
        "alpha": [0.01, 0.1, 1.0],    # ppt에 L1 L2 정규화 정리 
        "l1_ratio": [0.2, 0.5, 0.8]
    }),
}

results = [evaluate_model(name, model, params) for name, (model, params) in models.items()]

# 결과 출력
results_df = pd.DataFrame(results)
print(results_df)
""" 결과
        Model       MAE        MSE      RMSE        R2     CV R2                      Best Params
0       Ridge  4.506261  29.845047  5.463062  0.450482  0.378123                   {'alpha': 1.0}
1       Lasso  4.463250  29.209554  5.404586  0.462182  0.386787                   {'alpha': 0.1}
2  ElasticNet  4.494046  29.615579  5.442020  0.454707  0.382880  {'alpha': 0.1, 'l1_ratio': 0.8}
정규화 기법을 적용한 모델들이 큰 개선을 보이지 않는다는건 오버피팅이 발생하지 않았고 변수 간
중요도 차이도 크지 않다는 것을 의미한다. """


###############################################################################
# 비선형 회귀모델을 쓰면 선형 모델보다 더 유연하고 복잡한 패턴을 학습할 수 있어 성능이 좋은경우가있기에 사용
#3. 비선형 회귀 모델 학습 및 예측 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
def evaluate_non_linear_with_grid(name, model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(x_train_processed, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test_processed)

    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "CV R2": cross_val_score(best_model, x_train_processed, y_train, cv=5, scoring="r2").mean(),
        "Best Params": grid.best_params_
    }

# 모델별 파라미터 설정
non_linear_models_with_params = {
    "DecisionTree": (
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10]} 
        # max_depth는 트리의 최대 깊이로 클수록 복잡한 모델이 되며 과적합 가능성증가
        # min_samples_split는 노드를 분할하기 위한 최소 샘플 수로 작을수록 트리가 더 깊게 분할가능
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [50, 100, 200],  # 앙상블에 사용되는 트리 개수, 많은 수록 안정적이지만 오래걸림
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5]
        }
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],   # 각 단계에서 보정하는 비율, 작을수록 더 정밀하지만 느림 
            "max_depth": [3, 5, 10]
        }
    )
}

# 결과 수집
non_linear_grid_results = [
    evaluate_non_linear_with_grid(name, model, params)
    for name, (model, params) in non_linear_models_with_params.items()
]

# 결과 출력
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)
non_linear_grid_df = pd.DataFrame(non_linear_grid_results)
print(non_linear_grid_df)
""" 결과
              Model       MAE        MSE      RMSE        R2     CV R2                                                    Best Params
0      DecisionTree  4.521363  30.407134  5.514266  0.440132  0.362167                       {'max_depth': 3, 'min_samples_split': 2}
1      RandomForest  4.463511  29.175322  5.401419  0.462813  0.381013  {'max_depth': 3, 'min_samples_split': 5, 'n_estimators': 100}
2  GradientBoosting  4.475334  29.813993  5.460219  0.451053  0.369777    {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 50}"""


#####################################################################
#4. XGBoost 모델 학습 및 예측 
from xgboost import XGBRegressor

# 평가 함수
def evaluate_boosting_model(name, model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(x_train_processed, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test_processed)

    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "CV R2": cross_val_score(best_model, x_train_processed, y_train, cv=5, scoring="r2").mean(),
        "Best Params": grid.best_params_
    }

# 하이퍼파라미터 범위 설정
boosting_models = {
    "XGBoost": (
        XGBRegressor(objective="reg:squarederror", random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }
    )
}

# 결과 저장
boosting_results = [
    evaluate_boosting_model(name, model, param_grid)
    for name, (model, param_grid) in boosting_models.items()
]

# 결과 출력
boosting_df = pd.DataFrame(boosting_results)

# 출력 옵션 설정
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

print(boosting_df)




#################################################################################
# 모델별 예측 시각화 
def plot_all_models(y_test, predictions_dict):
    plt.figure(figsize=(16, 12))

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items(), 1):
        plt.subplot(3, 3, idx)
        plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(model_name)
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    
predictions = {}
# 모델별 결과로 나온온 최적의 파라미터를 사용하여 예측 수행 
# 1. 선형 회귀 모델
predictions["Linear Regression"] = LR_model.predict(x_test_processed)
predictions["Ridge"] = Ridge(alpha=1.0).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["Lasso"] = Lasso(alpha=0.1).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.8).fit(x_train_processed, y_train).predict(x_test_processed)

# 2. 비선형 회귀 모델
predictions["Decision Tree"] = DecisionTreeRegressor(max_depth=3, min_samples_split=2, random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["Random Forest"] = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_split=5, random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=3, random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)

# 3. XGBoost
from xgboost import XGBRegressor
predictions["XGBoost"] = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, objective="reg:squarederror", random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)


plot_all_models(y_test, predictions)

""" 결과를 보면 전혀 예측을 못하고 있음 
타깃 분포 확인 """

y.hist(bins=20)
plt.title("Target Variable Distribution")
plt.show()
x_train_df.corrwith(y_train).sort_values(ascending=False)

import seaborn as sns
""" 상관 계수 확인 해보면 전반적으로 약한 설명력을 가진다. """
# 1. 전체가 NaN인 열 제거
x_train_df = x_train_df.dropna(axis=1, how='all')

# 2. 표준편차 0인 열(상수 열) 제거
x_train_df = x_train_df.loc[:, x_train_df.std() != 0]
correlations = x_train_df.corrwith(y_train).sort_values(ascending=False)
print(correlations)


""" 개선 방향 -> 파생 변수 생성 -> 별차이 없음
-> 타깃변수 변경 ->  거의 완벽한 """


###########################
# 과적합 여부 확인
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_residuals(y_test, predictions["XGBoost"], "XGBoost")


###################################
# 결과적으로 gradient boosting 모델이 가장 좋은 성능을 보였다. feature importance 확인해봦 
# 모델 학습
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gb_model.fit(x_train_processed, y_train)

# 피처 이름 가져오기
numeric_features = pipeline.named_steps["preprocessor"].transformers_[0][2]
categorical_encoder = pipeline.named_steps["preprocessor"].transformers_[1][1]
categorical_features = categorical_encoder.get_feature_names_out()
feature_names = list(numeric_features) + list(categorical_features)

# 중요도 추출 및 정렬
importances = gb_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:20][::-1], importance_df["Importance"][:20][::-1])
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Top 20 Feature Importances")
plt.tight_layout()
plt.show()


