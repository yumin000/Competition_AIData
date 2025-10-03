# 학습은 한번에, 평가는 관측소 별로

# ==============================================================================
# 1. 라이브러리 불러오기
# ==============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os

# ==============================================================================
# 2. 초기 변수 설정 및 데이터 로딩/클리닝
# ==============================================================================
FILE_NAME = 'total_rename_data/trainData.csv'
DATE_COLUMN = 'ymd'
TARGET_COLUMN = 'elev'
# -----------------------------------------

# 데이터 로딩
df = pd.read_csv(FILE_NAME, encoding='cp949')

# 날짜 컬럼 처리 및 인덱스 설정
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df.set_index(DATE_COLUMN, inplace=True)
df.sort_index(inplace=True)

# 데이터 클리닝 ('가짜 0' 값 제거)
df[TARGET_COLUMN].replace(0, np.nan, inplace=True)
df.dropna(inplace=True)

print("--- 데이터 로딩 및 클리닝 완료 ---")
print(f"처리 후 데이터 크기: {df.shape}\n")

# ==============================================================================
# 3. 피처 엔지니어링 함수 정의
# ==============================================================================
def feature_engineering(df, target_col):
    """
    데이터프레임에서 시계열 피처를 생성하는 함수
    (규칙: 예측 시점 기준 최소 8일 전 데이터만 사용)
    """
    df_copy = df.copy()
    
    # --- 기준이 되는 데이터를 8일 전으로 이동 ---
    df_shifted = df_copy.shift(8)
    
    # 시간 관련 피처
    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['dayofyear'] = df_copy.index.dayofyear
    df_copy['time_index'] = (df_copy.index - df_copy.index.min()).days
    
    # 지연(Lag) 피처
    df_copy[f'{target_col}_lag_8'] = df_shifted[target_col]
    df_copy[f'{target_col}_lag_15'] = df_shifted[target_col].shift(7)
    df_copy[f'{target_col}_lag_30'] = df_shifted[target_col].shift(22)
    
    # 이동(Rolling) 및 변화량(Diff) 피처
    df_copy[f'{target_col}_rolling_mean_7_shifted'] = df_shifted[target_col].rolling(window=7).mean()
    df_copy[f'{target_col}_rolling_std_7_shifted'] = df_shifted[target_col].rolling(window=7).std()
    df_copy[f'{target_col}_rolling_mean_14_shifted'] = df_shifted[target_col].rolling(window=14).mean()
    df_copy[f'{target_col}_diff_7d_shifted'] = df_shifted[target_col].diff(periods=7)

    # 장기 이동 평균
    df_copy[f'{target_col}_rolling_mean_30_shifted'] = df_shifted[target_col].rolling(window=30).mean()
    df_copy[f'{target_col}_rolling_mean_60_shifted'] = df_shifted[target_col].rolling(window=60).mean()

    # 최종 결측치 제거
    df_copy = df_copy.dropna()
    
    return df_copy

# ==============================================================================
# 4. 평가지표 함수 정의
# ==============================================================================
def get_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def get_kge(y_true, y_pred):
    s_true, s_pred = pd.Series(y_true), pd.Series(y_pred)
    r = s_true.corr(s_pred)
    beta = s_pred.mean() / s_true.mean()
    gamma = s_pred.std() / s_true.std()
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

# ==============================================================================
# 5. 메인 실행 로직: code_new 별 모델 학습 및 평가
# ==============================================================================
# (1) 전체 데이터에 대해 피처 엔지니어링 우선 실행
final_df = feature_engineering(df, TARGET_COLUMN)

# 데이터 분할
features_to_drop = ['wtemp', 'ec', 'gtemp']
X = final_df.drop(columns=[TARGET_COLUMN] + features_to_drop, errors='ignore')
y = final_df[TARGET_COLUMN]

split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# (4) Optuna Objective 함수 정의
def objective(trial):
    params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'verbosity': -1,
            'n_estimators': 2000, 'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60), # 과대적합 방지를 위해 범위를 좁힘
            'max_depth': trial.suggest_int('max_depth', 5, 12), # 과대적합 방지를 위해 범위를 좁힘
            'reg_alpha' : trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda' : trial.suggest_float('reg_lambda', 0.0, 1.0), # L1 정규화
            'subsample': trial.suggest_float('subsample', 0.7, 1.0), # L2 정규화 
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }

    # 시계열 교차검증으로 안정적인 성능 평가
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    for train_idx, val_idx in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmses.append(rmse)
       
    return np.mean(rmses)

# 전체 모델 최적화 
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# 최종 모델 학습
best_params = {'random_state' : 42}
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)

# ==============================================================================
# 6. "평가는 따로": 학습된 모델로 관측소별 성능 평가
# ==============================================================================
results_df = pd.DataFrame(columns=['code_new', 'NSE', 'KGE'])
all_predictions = final_model.predict(X_test)
X_test_with_preds = X_test.copy()
X_test_with_preds['predictions'] = all_predictions
X_test_with_preds['actual'] = y_test

# 테스트 데이터에 있는 code_new에 대해서만 루프 실행
for code in X_test['code_new'].unique():
    # 해당 관측소의 테스트 데이터만 필터링
    station_test_data = X_test_with_preds[X_test_with_preds['code_new'] == code]
    
    station_y_test = station_test_data['actual']
    station_predictions = station_test_data['predictions']
    
    # KGE가 NaN이 나오지 않도록 데이터가 충분한지 확인
    if len(station_y_test) < 2 or station_y_test.std() == 0:
        nse_score, kge_score = -9999, -9999
    else:
        nse_score = get_nse(station_y_test, station_predictions)
        kge_score = get_kge(station_y_test, station_predictions)
    
    print(f"평가 완료: code_new = {code} | NSE: {nse_score:.4f}, KGE: {kge_score:.4f}")
    
    new_row = pd.DataFrame([{'code_new': code, 'NSE': nse_score, 'KGE': kge_score}])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# 7. 최종 성능 요약
print("\n최종 성능 요약")
print(results_df)   