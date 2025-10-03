# Optuna 말고 수동으로 파라미터 조정

# 하나의 모델로 학습
# trial을 레전드로 크게 

# ==============================================================================
# 1. 라이브러리 불러오기
# ==============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# ==============================================================================
# 2. 초기 변수 설정 및 데이터 로딩/클리닝
# ==============================================================================
FILE_NAME = 'total_rename_data/trainData.csv'
DATE_COLUMN = 'ymd'
TARGET_COLUMN = 'elev'
RAIN_COLUMN = 'rainfall' # 실제 강수량 컬럼명 확인

# 데이터 로딩
df = pd.read_csv(FILE_NAME, encoding='cp949')
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df.set_index(DATE_COLUMN, inplace=True)
df.sort_index(inplace=True)

# 데이터 클리닝
df[TARGET_COLUMN].replace(0, np.nan, inplace=True)
df.dropna(inplace=True)
print("--- 데이터 로딩 및 클리닝 완료 ---")

# ==============================================================================
# 3. 피처 엔지니어링 함수 정의
# ==============================================================================
def feature_engineering(df, target_col, rain_col):
    df_copy = df.copy()
    df_shifted = df_copy.shift(8)

    
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy.index.month/12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy.index.month/12)
    df_copy['dayofyear'] = df_copy.index.dayofyear
    df_copy['time_index'] = (df_copy.index - df_copy.index.min()).days
    

    df_copy[f'{target_col}_lag_8'] = df_shifted[target_col]
    df_copy[f'{target_col}_rolling_mean_14_shifted'] = df_shifted[target_col].rolling(window=14).mean()
    df_copy[f'{target_col}_rolling_std_14_shifted'] = df_shifted[target_col].rolling(window=14).std()
    df_copy[f'{target_col}_diff_7d_shifted'] = df_shifted[target_col].diff(periods=7)
    df_copy[f'{target_col}_rolling_mean_30_shifted'] = df_shifted[target_col].rolling(window=30).mean()
    

    df_copy[f'{rain_col}_rolling_sum_7_shifted'] = df_shifted[rain_col].rolling(window=7).sum()
    df_copy[f'{rain_col}_rolling_sum_30_shifted'] = df_shifted[rain_col].rolling(window=30).sum()
    df_copy[f'{rain_col}_lag_8'] = df_shifted[rain_col]
    df_copy = df_copy.dropna()
    
    
    return df_copy

# ==============================================================================
# 4. 평가지표 함수 정의 (NaN 방지 기능 포함)
# ==============================================================================
def get_nse(y_true, y_pred):
    if np.std(y_true) == 0: return -9999.0
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def get_kge(y_true, y_pred):
    s_true, s_pred = pd.Series(y_true), pd.Series(y_pred)
    if s_true.std() == 0 or len(s_true) < 2: return -9999.0
    r = s_true.corr(s_pred)
    if pd.isna(r): return -9999.0
    beta = s_pred.mean() / s_true.mean()
    gamma = s_pred.std() / s_true.std()
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

# ==============================================================================
# 5. "학습은 함께": num_leaves 수동 테스트
# ==============================================================================
print("\n--- 🚀 글로벌 모델 학습 시작 ---")

# (1) 전체 데이터에 대해 피처 엔지니어링 실행
final_df = feature_engineering(df, TARGET_COLUMN, RAIN_COLUMN)

# (2) 데이터 분할 (전체 데이터를 사용)
features_to_drop = ['wtemp', 'ec', 'gtemp', RAIN_COLUMN]
X = final_df.drop(columns=[TARGET_COLUMN] + features_to_drop, errors='ignore')
y = final_df[TARGET_COLUMN]

split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# (3) --- Optuna 최적화 부분 주석 처리 ---
# print("--- 글로벌 모델 최적화 (Optuna)는 건너뜁니다... ---")
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=30)
# best_params = study.best_params
# print(f"--- 최적 파라미터 발견: {best_params} ---")

# (4) 수동으로 파라미터 설정 
manual_params = {
    'objective': 'regression_l1',
    'random_state': 42,
    'learning_rate': 0.01,
    'n_estimators': 2000,
    
    'num_leaves': 81,
}
print(f"--- 수동 파라미터로 학습 시작 (num_leaves = {manual_params['num_leaves']}) ---")

# (5) 설정된 파라미터로 단 하나의 최종 모델 학습
final_model = lgb.LGBMRegressor(**manual_params)
# early_stopping을 위해 학습 시에만 X_test, y_test를 eval_set으로 사용
final_model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(100, verbose=False)])
print("\n--- 단일 글로벌 모델 학습 완료 ---\n")


# ==============================================================================
# 6. "평가는 따로": 학습된 모델로 관측소별 성능 평가
# ==============================================================================
results_df = pd.DataFrame(columns=['code_new', 'NSE', 'KGE'])
all_predictions = final_model.predict(X_test)
X_test_with_results = X_test.copy()
X_test_with_results['predictions'] = all_predictions
X_test_with_results['actual'] = y_test

for code in sorted(X_test['code_new'].unique()):
    station_test_data = X_test_with_results[X_test_with_results['code_new'] == code]
    nse_score = get_nse(station_test_data['actual'], station_test_data['predictions'])
    kge_score = get_kge(station_test_data['actual'], station_test_data['predictions'])
    
    print(f"평가 완료: code_new = {code} | NSE: {nse_score:.4f}, KGE: {kge_score:.4f}")
    new_row = pd.DataFrame([{'code_new': code, 'NSE': nse_score, 'KGE': kge_score}])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# ==============================================================================
# 7. 최종 결과 출력
# ==============================================================================
print("\n\n--- 🚀 최종 모델 성능 ---")
print(results_df)