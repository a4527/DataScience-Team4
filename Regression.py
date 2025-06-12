import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from func_insertNanOutlier import append_outliers_and_missing_by_row
from func_delNanOutlier import remove_missing_and_outliers_iqr
from func_getPreprocessedDataframe import get_preprocessed_dataframe

df=pd.read_csv('power_marketing_dataset.csv')

# Create DataFrames with missing values and outliers 
df=append_outliers_and_missing_by_row(df)


##################################################################################################
##### Create Derivative variable #######
# 1. Consumption per person 
df['Consumption_per_Person'] = df['Monthly Consumption (kWh)'] / (df['Household Size'] + 1)

# 2. Peak to Average Consumption Ratio
df['Peak_to_Avg_Consumption'] = df['Peak Consumption (kWh)'] / (df['Avg Consumption (kWh)'] + 1)

# 3. Consumption by TIme of Day Ratio
df['Evening_Morning_Diff'] = df['Consumption by Time of Day (Evening)'] - df['Consumption by Time of Day (Morning)']

# 4. Binary variaable for high engagement
df['High_Engagement'] = (df['Engagement Rate'] > 0.5).astype(int)
##################################################################################################


# Save missing and outliers inserted dataframe as csv file 
df.to_csv('power_marketing_dataset_with_outliers_and_missing.csv',index=False)


# Check the number of missing values in the dataframe 
print("결측치 개수: ",df.isnull().sum().sum())


# Visualization of missing values in data frames 
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
missing_counts.plot(kind='barh', figsize=(8, 4), color='salmon')
plt.title("Number of Missing Values per Column")
plt.xlabel("Missing Count")
plt.ylabel("Column")
plt.show()


# Visualization of outliers in dataframes
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].plot(kind='box', subplots=True, layout=(5, 3), figsize=(15, 10), sharex=False)
plt.tight_layout()
plt.show()


# Create a data frame with missing and outliers removed
df_cleaned=remove_missing_and_outliers_iqr(df)
df_cleaned.to_csv('power_marketing_dataset_cleaned.csv',index=False)


# Visualize distribution by variable
df_cleaned.select_dtypes(include=['int64','float64']).hist(bins=20,figsize=(16,10))
plt.tight_layout()
plt.show()


# Check the number of missing and outliers removed data frames
print("결측치 개수 (이상치 제거 후):",df_cleaned.isnull().sum().sum())


#################################################################################################
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Remove unnecessary columns
df_cleaned=df_cleaned.drop(columns=["User ID"])


# Overfitting to eliminate suspected leak features
df_cleaned=df_cleaned.drop(columns=["Monthly Consumption (kWh)"]) 


# seperate target variable and features
target_col="Energy Usage Reduction (%)"
target_col2="Avg Consumption (kWh)"
x=df_cleaned.drop(columns=[target_col2]) 
y=df_cleaned[target_col2] 


# seperate categorical and numeric columns
categorical_cols = x.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()


# Create a preprocessor pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)


# Create a pipelne that includes the preprocessor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)])


# split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


# execute the preprocessor pipeline on the training and test sets
x_train_processed = pipeline.fit_transform(x_train)
x_test_processed = pipeline.transform(x_test)


# Use to check if preprocessing is done well
x_train_df = get_preprocessed_dataframe(pipeline, x_train_processed, numeric_cols, categorical_cols)
print(x_train_df.head())


################################################################################################################

# Regression model learning and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


#1. Basic Model Linear Regression Learning and Prediction
LR_model=LinearRegression()
LR_model.fit(x_train_processed,y_train)
y_pred=LR_model.predict(x_test_processed)


# 1-1  Calculate Evaluation Indicators
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(LR_model, x_train_processed, y_train, cv=5, scoring="r2").mean()


# 1-2 print evaluation results
print("Linear Regression Results:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Cross-Validated R2:", cv_r2)


#2. Reaularized Regression Models Learning and prediction
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV


# 2-1 evaluation function for regularized regression models
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

models = { # Setting Normalization Models and Parameters
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1.0, 10.0]}),  
    "Lasso": (Lasso(), {"alpha": [0.01, 0.1, 1.0, 10.0]}),  
    "ElasticNet": (ElasticNet(), {    
        "alpha": [0.01, 0.1, 1.0],    
        "l1_ratio": [0.2, 0.5, 0.8]
    }),
}

results = [evaluate_model(name, model, params) for name, (model, params) in models.items()]

# 2-2 print results
results_df = pd.DataFrame(results)
print(results_df)



#3. Nonlinear Regression Models Learning and Prediction
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


#3-1 Evaluation function for non-linear regression models
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

non_linear_models_with_params = { # setting non-linear regression models and parameters
    "DecisionTree": (
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10]} 
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [50, 100, 200],  
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5]
        }
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],   
            "max_depth": [3, 5, 10]
        }
    )
}

non_linear_grid_results = [
    evaluate_non_linear_with_grid(name, model, params)
    for name, (model, params) in non_linear_models_with_params.items()
]

# 3-2 print result
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)
non_linear_grid_df = pd.DataFrame(non_linear_grid_results)
print(non_linear_grid_df)


#4. Learning and predicting XGBoost models
from xgboost import XGBRegressor

# 4-1 Evaluation functionfor XGBoost Models
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

boosting_models = { # Setting Range of parameter
    "XGBoost": (
        XGBRegressor(objective="reg:squarederror", random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }
    )
}

boosting_results = [
    evaluate_boosting_model(name, model, param_grid)
    for name, (model, param_grid) in boosting_models.items()
]

# 4-2 print result
boosting_df = pd.DataFrame(boosting_results)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

print(boosting_df)




##########################################################################################
# Visualize predictions by each model

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

# Make predictions using the best parameters resulting from each model

# 1. Linear Regression Model
predictions["Linear Regression"] = LR_model.predict(x_test_processed)
predictions["Ridge"] = Ridge(alpha=1.0).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["Lasso"] = Lasso(alpha=0.1).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.8).fit(x_train_processed, y_train).predict(x_test_processed)


# 2. NonLinear Regression Model
predictions["Decision Tree"] = DecisionTreeRegressor(max_depth=3, min_samples_split=2, random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["Random Forest"] = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_split=5, random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)
predictions["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=3, random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)


# 3. XGBoost Model
from xgboost import XGBRegressor
predictions["XGBoost"] = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, objective="reg:squarederror", random_state=42).fit(x_train_processed, y_train).predict(x_test_processed)


plot_all_models(y_test, predictions)



# check target variable Distribution
y.hist(bins=20)
plt.title("Target Variable Distribution")
plt.show()
x_train_df.corrwith(y_train).sort_values(ascending=False)

# Remove columns with zero informaion (all Nan or constants) to reduce the amount of computation when learning the model
x_train_df = x_train_df.dropna(axis=1, how='all')
x_train_df = x_train_df.loc[:, x_train_df.std() != 0]
correlations = x_train_df.corrwith(y_train).sort_values(ascending=False)
print(correlations)





##############################################################################
# Check Overfitting 
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


########################################################################################
# As a result, the gradient boosting model showed the best performance. Let's check the feature importance

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
gb_model.fit(x_train_processed, y_train)

# Get feature name 
numeric_features = pipeline.named_steps["preprocessor"].transformers_[0][2]
categorical_encoder = pipeline.named_steps["preprocessor"].transformers_[1][1]
categorical_features = categorical_encoder.get_feature_names_out()
feature_names = list(numeric_features) + list(categorical_features)

# Extract and sort importance
importances = gb_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:20][::-1], importance_df["Importance"][:20][::-1])
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Top 20 Feature Importances")
plt.tight_layout()
plt.show()


