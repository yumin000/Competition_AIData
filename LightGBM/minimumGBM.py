import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# 데이터 로딩 및 기본 처리 확인
# --------------------------------------------------------------------------
FILE_NAME = 'total_rename_data/trainData.csv'
DATE_COLUMN = 'ymd'
TARGET_COLUMN = 'elev'

df_original = pd.read_csv(FILE_NAME, encoding='cp949')
df_original[DATE_COLUMN] = pd.to_datetime(df_original[DATE_COLUMN])
df_original.set_index(DATE_COLUMN, inplace=True)
df_original.sort_index(inplace=True)

print("초기 데이터 shape:", df_original.shape)
print("초기 데이터 컬럼:", df_original.columns.tolist())


# ---------- 데이터 클리닝 ----------
df_cleaned = df_original.copy()
df_cleaned[TARGET_COLUMN].replace(0, np.nan, inplace=True)
df_cleaned.dropna(inplace=True)

print("클리닝 후 데이터 shape:", df_cleaned.shape)
print(f"클리닝 후 '{TARGET_COLUMN}' 컬럼에 0이 있는지 확인:", (df_cleaned[TARGET_COLUMN] == 0).sum())

# ---------- 최소한의 피처 엔지니어링 ----------

# ⭐⭐ 1. 이 함수 전체를 새 버전으로 교체 ⭐⭐
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
    
    # 지연(Lag) 피처
    df_copy[f'{target_col}_lag_8'] = df_shifted[target_col]
    df_copy[f'{target_col}_lag_15'] = df_shifted[target_col].shift(7)
    df_copy[f'{target_col}_lag_30'] = df_shifted[target_col].shift(22)
    
    # 이동(Rolling) 및 변화량(Diff) 피처
    df_copy[f'{target_col}_rolling_mean_7_shifted'] = df_shifted[target_col].rolling(window=7).mean()
    df_copy[f'{target_col}_rolling_std_7_shifted'] = df_shifted[target_col].rolling(window=7).std()
    df_copy[f'{target_col}_rolling_mean_14_shifted'] = df_shifted[target_col].rolling(window=14).mean()
    df_copy[f'{target_col}_diff_7d_shifted'] = df_shifted[target_col].diff(periods=7)

    # 최종 결측치 제거
    df_copy = df_copy.dropna()
    
    return df_copy
final_df = feature_engineering(df_cleaned, TARGET_COLUMN)


# ---------- 데이터 분할 ----------
# 유출 가능성이 있는 모든 피처를 일단 제거
features_to_drop = ['wtemp', 'ec', 'gtemp']

X = final_df.drop(columns=[TARGET_COLUMN] + features_to_drop, errors='ignore')
y = final_df[TARGET_COLUMN]

split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_train 컬럼에 타겟 변수가 없는지 최종 확인:")
print(X_train.columns.tolist())



# ---------- 모델 학습 및 평가 ----------
print("Training ... ")
# Optuna 없이 가장 기본적인 모델로만 테스트
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

def get_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def get_kge(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    beta = np.mean(y_pred) / np.mean(y_true)
    gamma = np.std(y_pred) / np.std(y_true)
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return kge

nse_score = get_nse(y_test, predictions)
kge_score = get_kge(y_test, predictions)

print(f"최소 기능 모델의 NSE Score: {nse_score:.4f}")
print(f"KGE Score: {kge_score:.4f}")


# 1. 실제값 vs 예측값 비교 그래프 생성 및 저장
fig = go.Figure()

# 실제값(Actual) 라인 추가
fig.add_trace(go.Scatter(
    x=y_test.index, 
    y=y_test, 
    mode='lines', 
    name='Actual (실제값)', 
    line=dict(color='blue', width=2)
))

# 예측값(Predicted) 라인 추가
fig.add_trace(go.Scatter(
    x=y_test.index, 
    y=predictions, 
    mode='lines', 
    name='Predicted (예측값)', 
    line=dict(color='red', dash='dot', width=1.5), 
    opacity=0.8
))

# 그래프 레이아웃 설정
fig.update_layout(
    title='📈 지하수위 실제값 vs. 모델 예측값 비교', 
    xaxis_title='날짜', 
    yaxis_title='지하수위'
)

# 그래프를 화면에 바로 표시
fig.show()

# 시각화할 데이터 샘플링 (예: 마지막 365일)
SAMPLE_DAYS = 365
y_test_sample = y_test.tail(SAMPLE_DAYS)
predictions_sample = predictions[-SAMPLE_DAYS:] # predictions는 numpy 배열이므로 인덱싱 방식이 다름

# 1. 실제값 vs 예측값 비교 그래프 생성
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test_sample.index, 
    y=y_test_sample, 
    mode='lines', 
    name='Actual (실제값)', 
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=y_test_sample.index, 
    y=predictions_sample, 
    mode='lines', 
    name='Predicted (예측값)', 
    line=dict(color='red', dash='dot', width=1.5), 
    opacity=0.8
))

fig.update_layout(
    title=f'📈 지하수위 실제값 vs. 모델 예측값 비교 (최근 {SAMPLE_DAYS}일)', 
    xaxis_title='날짜', 
    yaxis_title='지하수위'
)

fig.show()

# 2. 피처 중요도 그래프 생성 및 저장
# 그래프 사이즈 설정
fig_importance, ax_importance = plt.subplots(figsize=(10, 12))

# 피처 중요도 시각화
lgb.plot_importance(
    model, 
    ax=ax_importance, 
    max_num_features=20, # 상위 20개 피처만 표시
    title='피처 중요도 (Feature Importance)'
)

# 그래프를 화면에 바로 표시
plt.show()



# ======= 다른 성능 지표 =======

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RMSE 계산 (MSE에 제곱근을 취함)
# 0에 가까울수록 좋음
rmse_score = np.sqrt(mean_squared_error(y_test, predictions))

# MAE 계산
# 0에 가까울수록 좋음
mae_score = mean_absolute_error(y_test, predictions)

# R² 계산
# 1에 가까울수록 좋음
r2_score_value = r2_score(y_test, predictions)

print(f"RMSE Score: {rmse_score:.4f}")
print(f"MAE Score: {mae_score:.4f}")
print(f"R² Score: {r2_score_value:.4f}")