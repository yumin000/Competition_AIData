# Bayesian Optimizer 적용

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------- 데이터 불러오기 및 전처리 ----------
FILE_NAME = 'total_rename_data/trainData.csv'
DATE_COLUMN = 'ymd'
TARGET_COLUMN = 'elev'

df = pd.read_csv(FILE_NAME, encoding='cp949')

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df.set_index(DATE_COLUMN, inplace=True)
df.sort_index(inplace=True)

print(f"청소 전 원본 데이터 크기 : {df.shape}")

# 가짜 0으로 의심되는 값을 진짜 결측치(np.nan)로 변경
df[TARGET_COLUMN].replace(0, np.nan, inplace=True)

# 결측치가 포함된 행 전체를 제거
df = df.dropna()
print(f"청소 후 데이터 크기: {df.shape}")

print(df.head())

# ---------- 피쳐 엔지니어링 ----------
def feature_engineering(df, target_col):
    df_copy = df.copy()

    # 시간 피쳐
    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['dayofyear'] = df_copy.index.dayofyear

    # 지연 피쳐
    lag_days = [7, 14, 21, 30]
    for lag in lag_days:
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)

    # 이동 피쳐
    rolling_windows = [7, 14, 30]
    for window in rolling_windows:
        df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
        df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window).std()

    # 생성된 결측치 제거
    df_copy = df_copy.dropna()
    
    return df_copy

# 깨끗해진 데이터로 피쳐 엔지니어링 실행
final_df = feature_engineering(df, TARGET_COLUMN)

# ---------- 데이터 분할 ----------
features_to_drop = ['wtemp', 'ec']

X = final_df.drop(columns=[TARGET_COLUMN] + features_to_drop, errors='ignore')
y = final_df[TARGET_COLUMN]

split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"\n훈련 데이터 크기: {X_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}")
print("\n최종 사용된 피쳐 목록: \n", X_train.columns.tolist())


# ---------- 베이지안 최적화로 하이퍼파라미터 튜닝 ----------
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    rmses = []

    # 시계열 교차 검증으로 모델 성능 평가
    for train_index, val_index in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmses.append(rmse)

    return np.mean(rmses)

print("\n\n=== Starting Bayesian Optimization... ===")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("\n=== Optimal Hyperparameters ===")
print(study.best_params)


# ---------- 최종 모델 학습 및 예측 및 평가 ----------
best_params = study.best_params
final_model = lgb.LGBMRegressor(**best_params, n_estimators=2000, random_state=42)

print("\n===Start Training ... ===")
final_model.fit(X_train, y_train)
print("===Completed===")

# 테스트 데이터로 예측
predictions = final_model.predict(X_test)

# 평가지표 계산 함수
def get_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def get_kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = np.std(y_pred) / np.std(y_true)
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

# 성능 평가
nse_score = get_nse(y_test, predictions)
kge_score = get_kge(y_test, predictions)

print("\n--- 🚀 최종 모델 성능 평가 🚀 ---")
print(f"NSE Score: {nse_score:.4f}")
print(f"KGE Score: {kge_score:.4f}")

# ---------- 결과 시각화 ----------"
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual (실제값)', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted (예측값)', line=dict(color='red', dash='dot', width=1.5), opacity=0.8))
fig.update_layout(title='📈 지하수위 실제값 vs. 모델 예측값 비교', xaxis_title='날짜', yaxis_title='지하수위')
fig.show()

# 피쳐 중요도
lgb.plot_importance(final_model, figsize=(10, 12), max_num_features=20, title='피처 중요도 (Feature Importance)')
plt.show()