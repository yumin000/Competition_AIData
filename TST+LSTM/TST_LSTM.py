import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import random
import os

# ----------------------------------------------------------------------------------
# 0. 설정 및 유틸리티 함수
# ----------------------------------------------------------------------------------

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 

def nse(obs, sim):
    """Nash-Sutcliffe Efficiency"""
    denominator = np.sum((obs - np.mean(obs))**2)
    if denominator == 0: return -np.inf
    return 1 - np.sum((sim - obs)**2) / denominator

def kge(obs, sim):
    """Kling-Gupta Efficiency"""
    # obs와 sim의 길이가 다를 경우 np.corrcoef에서 ValueError 발생할 수 있으므로, 
    # nse/kge 호출 전에 길이를 맞춰야 합니다. (아래 평가 코드에서 처리됨)
    if np.std(obs) == 0 or np.std(sim) == 0: return -np.inf
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)


# ----------------------------------------------------------------------------------
# 1. 위치 인코딩 클래스
# ----------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (S, 1, E)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# ----------------------------------------------------------------------------------
# 2. 시계열 데이터셋 클래스
# ----------------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1):
        self.x_data = x_data
        self.y_data = y_data
        self.input_len = input_len
        self.pred_len = pred_len
        self.dates = [str(d) for d in dates]

    def __len__(self):
        return len(self.x_data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 인코더 입력: 과거 시퀀스
        encoder_input = self.x_data[idx : idx + self.input_len]
        
        # 디코더 입력 (여기서는 사용되지 않지만 구조 유지를 위해 포함)
        decoder_input = self.y_data[idx + self.input_len - 1].reshape(1, -1) 
        
        # 타겟 출력: 예측 시점의 정답
        target_output = self.y_data[idx + self.input_len : idx + self.input_len + self.pred_len]
        
        # 예측 시점의 날짜
        date = self.dates[idx + self.input_len]
        
        return (
            torch.tensor(encoder_input, dtype=torch.float32), 
            torch.tensor(decoder_input, dtype=torch.float32),
            torch.tensor(target_output, dtype=torch.float32), 
            date
        )


# ----------------------------------------------------------------------------------
# 3. 트랜스포머 + LSTM 하이브리드 모델 클래스
# ----------------------------------------------------------------------------------

class TimeSeriesHybridModel(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.2, pred_len=1):
        super(TimeSeriesHybridModel, self).__init__()
        
        # 1. 인코더 임베딩 및 위치 인코딩
        self.encoder_embedding = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. LSTM 레이어 (Transformer의 출력을 시퀀스 입력으로 받음)
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=1, 
            batch_first=False 
        )
        
        # 4. 최종 출력 레이어
        self.fc_out = nn.Linear(d_model, pred_len) 
        self.d_model = d_model

    def forward(self, x):
        # x: (B, S, F) -> B: Batch, S: Sequence, F: Features
        
        # 임베딩
        x = self.encoder_embedding(x) * math.sqrt(self.d_model)
        x = x.permute(1, 0, 2) # (S, B, E)
        x = self.pos_encoder(x) 
        
        # 트랜스포머 인코더
        memory = self.transformer_encoder(x) # (S, B, E)
        
        # LSTM 통과
        # hn: 최종 은닉 상태 (num_layers, B, H)
        lstm_output, (hn, cn) = self.lstm(memory) 
        
        # 최종 예측: LSTM의 마지막 은닉 상태 사용 (hn[0] = (B, E))
        out = hn.squeeze(0) 
        
        # 최종 선형 변환
        out = self.fc_out(out) # (B, pred_len)
        return out


# ----------------------------------------------------------------------------------
# 4. 데이터 로드 및 학습 실행
# ----------------------------------------------------------------------------------

# 모델과 스케일러 저장을 위한 딕셔너리
trained_models = {}
feature_scalers = {}
target_scalers = {}

try:
    # 훈련 데이터 및 테스트 입력 데이터 로드
    ddf = pd.read_csv("total_rename_data/trainData.csv")
    test_df = pd.read_csv("total_rename_data/test_inputs.csv") 
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
    raise # 파일 없으면 실행 중지

# 데이터 전처리 공통 부분
ddf["ymd"] = pd.to_datetime(ddf["ymd"])
ddf = ddf.ffill().bfill() # 결측치 처리
test_df['ymd'] = pd.to_datetime(test_df['ymd'])
test_df = test_df.ffill().bfill()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10

# code_new 별로 학습 및 평가 루프
for code, df in ddf.groupby('code_new'):
    
    df = df.sort_values(by="ymd").reset_index(drop=True)
    
    # --------------------
    # 데이터셋 및 스케일링
    # --------------------
    features = df.drop(columns=["ymd", "code_new", "elev"]).values
    target = df[["elev"]].values
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # 스케일러 저장 (나중에 test_inputs 예측에 사용)
    feature_scalers[code] = feature_scaler
    target_scalers[code] = target_scaler
    
    dataset = TimeSeriesDataset(
        x_data=scaled_features,
        y_data=scaled_target,
        dates=df["ymd"].values,
        input_len=24, pred_len=1
    )
    
    train_size = int(len(dataset) * 0.9)
    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    
    # --------------------
    # 모델 정의 및 학습
    # --------------------
    model = TimeSeriesHybridModel(feature_size=scaled_features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    print(f"\n--- Training for code_new: {code} (Transformer + LSTM Hybrid Model) ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for src, _, y, _ in train_loader: 
            src, y = src.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(src) 
            loss = criterion(preds, y.view(-1, 1))
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

    # 모델 저장
    trained_models[code] = model
    
    # --------------------
    # 평가 및 시각화 (Train Data의 테스트 셋)
    # --------------------
    model.eval()
    preds_list, actuals_list, dates_list = [], [], []

    with torch.no_grad():
        for src, _, y, d_batch in test_loader:
            src, y = src.to(device), y.to(device)
            pred = model(src) 
            preds_list.extend(pred.view(-1, 1).cpu().numpy())
            actuals_list.extend(y.view(-1, 1).cpu().numpy())
            dates_list.extend(d_batch)

    preds_arr = target_scaler.inverse_transform(np.array(preds_list))
    actuals_arr = target_scaler.inverse_transform(np.array(actuals_list))

    # **NSE/KGE 오류 해결: 길이 불일치 수정**
    min_len = min(len(preds_arr), len(actuals_arr))
    if len(preds_arr) != len(actuals_arr):
        print(f"[WARNING] Test set 길이 불일치 ({len(preds_arr)} vs {len(actuals_arr)}). 길이를 맞춥니다.")
        preds_arr = preds_arr[:min_len]
        actuals_arr = actuals_arr[:min_len]
    
    dates_arr = pd.to_datetime(dates_list[:len(preds_arr)])

    print("\n--- Evaluation Metrics (Internal Test Set) ---")
    print(f"Station: {code}")
    print(f"NSE: {nse(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
    print(f"KGE: {kge(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
    
    # 시각화 (생략)
    if not os.path.exists("results"):
        os.makedirs("results")
    # plt.figure(figsize=(12, 6))
    # plt.plot(dates_arr, actuals_arr.flatten(), label='Actual', alpha=0.7)
    # plt.plot(dates_arr, preds_arr.flatten(), label='Predicted', alpha=0.7)
    # plt.title(f'Test Prediction - Station: {code}')
    # plt.legend()
    # plt.savefig(f"results/prediction_station_{code}.png")
    # plt.close()


# ----------------------------------------------------------------------------------
# 5. 최종 테스트 데이터 예측 및 CSV 생성
# ----------------------------------------------------------------------------------
# 기존 전체 코드의 5번과 6번 섹션을 이 코드로 대체하세요.

# ----------------------------------------------------------------------------------
# 5. 최종 테스트 데이터 예측 (Long Format으로 통합)
# ----------------------------------------------------------------------------------
print("\n" + "="*50)
print("--- Generating Final Test Predictions on test_inputs.csv (Long Format) ---")
print("="*50)

final_predictions_long_df = pd.DataFrame()
test_df_processed = test_df.copy()

# 각 스테이션별로 저장된 모델과 스케일러를 사용하여 예측 수행
for code, df_test in test_df_processed.groupby('code_new'):
    
    # 훈련되지 않은 스테이션 스킵
    if code not in trained_models:
        print(f"[SKIP] Station {code} not trained. Skipping final prediction.")
        continue
        
    model = trained_models[code]
    feature_scaler = feature_scalers[code]
    target_scaler = target_scalers[code]
    
    df_test = df_test.sort_values(by="ymd").reset_index(drop=True)
    
    # 'ymd', 'code_new' 칼럼 제외하고 피처 추출
    test_features = df_test.drop(columns=["ymd", "code_new"]).values 
    
    # 스케일링
    scaled_test_features = feature_scaler.transform(test_features)
    
    # 더미 타겟 배열
    dummy_target = np.zeros((len(scaled_test_features), 1)) 
    
    # TimeSeriesDataset 생성
    test_data_for_pred = TimeSeriesDataset(
        x_data=scaled_test_features,
        y_data=dummy_target, 
        dates=df_test["ymd"].values,
        input_len=24, pred_len=1
    )
    
    if len(test_data_for_pred) == 0:
        print(f"[SKIP] Station {code}: Not enough data (min length {24+1}) for prediction sequence.")
        continue
        
    test_pred_loader = DataLoader(test_data_for_pred, batch_size=512, shuffle=False)

    # 예측 수행
    model.eval()
    test_preds_list, test_dates_list = [], []
    
    with torch.no_grad():
        for src, _, _, d_batch in test_pred_loader:
            src = src.to(device)
            pred = model(src)
            
            test_preds_list.extend(pred.view(-1, 1).cpu().numpy())
            test_dates_list.extend(d_batch)
    
    # 스케일 역변환 및 DataFrame 생성
    test_preds_arr = target_scaler.inverse_transform(np.array(test_preds_list))
    test_dates_arr = pd.to_datetime(test_dates_list)
    
    pred_df = pd.DataFrame({
        "ymd": test_dates_arr,
        "code_new": code,
        "elev": test_preds_arr.flatten()
    })
    
    final_predictions_long_df = pd.concat([final_predictions_long_df, pred_df], ignore_index=True)


# ----------------------------------------------------------------------------------
# 6. 최종 CSV 파일 형식 맞추기 (Wide Format) 및 저장
# ----------------------------------------------------------------------------------
print("\n--- Converting to Wide Format (ymd, 1, 2, 3, ...) ---")

# 1. Wide Format으로 변환 (Pivot Table 사용)
# index: ymd, columns: code_new, values: elev (예측값)
final_output_wide_df = final_predictions_long_df.pivot_table(
    index='ymd', 
    columns='code_new', 
    values='elev', 
    aggfunc='first' # 중복된 ymd/code_new 쌍이 없으므로 'first' 사용
).reset_index()

# 2. 칼럼 이름 정리 및 정렬 (요구 형식: ymd, 1, 2, ..., 12)
# 'code_new'가 문자열일 경우를 대비하여 정수형으로 변환 후 정렬
try:
    # 'code_new' 칼럼(숫자)을 정수형으로 변환하고, ymd 다음으로 정렬
    code_cols = [col for col in final_output_wide_df.columns if col != 'ymd']
    code_cols_int = [int(col) for col in code_cols]
    
    # 정렬된 칼럼 이름 목록 생성
    sorted_code_cols = sorted(code_cols_int)
    
    # 최종 칼럼 순서
    final_cols = ['ymd'] + [str(col) for col in sorted_code_cols]
    
    # DataFrame의 칼럼 이름과 순서를 맞춤
    final_output_wide_df.columns = ['ymd'] + [str(col) for col in final_output_wide_df.columns[1:]]
    final_output_wide_df = final_output_wide_df[final_cols]
    
except Exception as e:
    print(f"[WARNING] 스테이션 칼럼 이름 정렬 중 오류 발생: {e}. 기존 칼럼 순서로 저장합니다.")


# 3. CSV 파일 저장 (요구 형식: 값이 없는 곳은 공백)
# to_csv의 na_rep='' 옵션을 사용하면 NaN 값을 공백으로 채워줍니다.
final_output_wide_df.to_csv("final_test_predictions_wide.csv", index=False, na_rep='')

print(f"\n✅ Final predictions saved to 'final_test_predictions_wide.csv' in Wide Format.")
print(f"   Generated dates: {len(final_output_wide_df)}")