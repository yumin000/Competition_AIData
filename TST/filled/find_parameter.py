import pandas as pd
import numpy as np
import warnings
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMAX 최적화 중 발생하는 경고 무시
warnings.filterwarnings("ignore")

# --- 0. Objective Function 정의 (NaN 값 처리 기능 강화) ---
def objective(trial, series, s):
    # SARIMA 파라미터의 탐색 공간 정의
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)

    order = (p, d, q)
    seasonal_order = (P, D, Q, s)

    try:
        # 결측치(NaN)가 포함된 원본 series를 그대로 사용합니다.
        # SARIMAX는 내부적으로 NaN을 처리하며, 이 경우 AIC 계산은 NaN이 없는 부분만 사용합니다.
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # 최대 반복 횟수(maxiter)를 늘려 수렴 실패를 줄이고,
        # 실패 시에도 오류 대신 None을 반환하도록 합니다.
        results = model.fit(disp=False, maxiter=200, low_memory=True)
        
        # 목표 지표: AIC 최소화
        return results.aic
        
    except Exception as e:
        # 모델 적합 실패 (수렴 실패, 특이점, Singular Matrix 등) 시
        # Optuna가 해당 Trial을 버리고 다음 Trial을 진행하도록 매우 큰 값을 반환
        print(e)
        return float('inf')

# --- 1. 데이터 및 설정 준비 ---
# 가정: 데이터프레임 `df`가 있고, 인덱스는 DatetimeIndex이며, 
# 결측치는 이미 np.nan 형태로 존재합니다.

# Optuna 튜닝에 사용할 데이터프레임 (원본 df를 그대로 사용)
df=pd.read_csv('TST_test5_.csv')
df_for_tuning = df.copy()

# 결과 저장을 위한 딕셔너리
best_params_per_column = {}
# 계절 주기(Seasonality) 설정 (데이터의 주기에 맞게 설정)
SEASONALITY = 12 
N_TRIALS = 5 # Optuna 시도 횟수 (적절히 조절)

# --- 2. 컬럼별 최적 파라미터 탐색 루프 ---
for col in df_for_tuning.columns:
    print(f"\n==================================================")
    print(f"컬럼 '{col}': Optuna 최적 SARIMAX 파라미터 탐색 시작 (NaN 포함)")
    print(f"==================================================")
    
    # 해당 컬럼의 데이터 추출 (NaN 포함)
    ts_data = df_for_tuning[col]
    
    # NaN이 아닌 유효 데이터 포인트가 최소한 계절 주기 2배 이상은 있어야 튜닝 가능
    if ts_data.dropna().shape[0] < 2 * SEASONALITY:
        print(f"컬럼 '{col}': 유효 데이터 포인트가 너무 적어 SARIMAX 튜닝을 건너뜜.")
        continue

    # Optuna Study 생성 및 최적화 실행
    study = optuna.create_study(direction='minimize')
    
    # objective 함수에 결측치가 포함된 ts_data를 인자로 전달
    study.optimize(lambda trial: objective(trial, ts_data, SEASONALITY), 
                   n_trials=N_TRIALS, 
                   show_progress_bar=True)

    # 최적 파라미터 저장
    if study.best_value != float('inf'):
        best_params = study.best_params
        best_order = (best_params['p'], best_params['d'], best_params['q'])
        best_seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], SEASONALITY)
        
        best_params_per_column[col] = {
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'best_aic': study.best_value
        }
        
        print(f"컬럼 '{col}' 최종 결과:")
        print(f"  Best AIC: {study.best_value:.2f}")
        print(f"  Order: {best_order}")
        print(f"  Seasonal Order: {best_seasonal_order}")
    else:
        print(f"컬럼 '{col}': 모든 시도에서 SARIMAX 모델 적합에 실패했습니다.")


# --- 3. 최종 결과 출력 ---
print("\n\n##################################################")
print("모든 컬럼의 최적 SARIMAX 파라미터")
print("##################################################")
for col, params in best_params_per_column.items():
    print(f"컬럼 '{col}':")
    print(f"  AIC: {params['best_aic']:.2f}, Order: {params['order']}, Seasonal: {params['seasonal_order']}")

# 이 결과를 사용하여 이후에 결측치 보간을 수행하면 됩니다.