import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import math
import random
import matplotlib.pyplot as plt # Matplotlib import 추가


# --------------------
# 1. 클래스 및 함수 정의 (Encoder-Only 모델 사용)
# --------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x.size(0) is the sequence length
        return x + self.pe[:x.size(0), :]

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1, is_inference=False):
        self.x_data = x_data
        self.y_data = y_data
        self.dates = [str(d) for d in dates]
        self.input_len = input_len
        self.pred_len = pred_len
        self.is_inference = is_inference # 추론 모드 플래그 추가

    def __len__(self):
        # 학습: 마지막 시점까지 데이터를 만들 수 있는 인덱스 + 1
        # 추론: 전체 시계열에서 인코더 입력 길이만큼 빼서 첫 예측 시작점을 찾음
        if self.is_inference:
            return len(self.x_data) - self.input_len + 1
        else:
            return len(self.x_data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.x_data[idx:idx+self.input_len]
        date = self.dates[idx+self.input_len-1] # 인코더의 마지막 시점 날짜

        if not self.is_inference:
            # 학습/검증: 타겟 Y값 존재
            y = self.y_data[idx+self.input_len:idx+self.input_len+self.pred_len]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), date
        else:
            # 추론: 타겟 Y값은 사용하지 않음
            # 타겟 날짜는 인코더 마지막 시점의 다음 시점입니다.
            prediction_date = self.dates[idx + self.input_len] if idx + self.input_len < len(self.dates) else None
            return torch.tensor(x, dtype=torch.float32), torch.empty(self.pred_len, 1, dtype=torch.float32), prediction_date


class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.1, pred_len=1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(feature_size, d_model) 
        self.pos_encoder = PositionalEncoding(d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
        self.fc_out = nn.Linear(d_model, pred_len) 
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model) 
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x) 
        out = self.transformer_encoder(x) 
        out = out[-1, :, :] # 마지막 시점의 출력값
        out = self.fc_out(out)
        return out

def nse(obs, sim):
    denominator = np.sum((obs - np.mean(obs))**2)
    if denominator == 0: return -np.inf
    return 1 - np.sum((sim - obs)**2) / denominator

def kge(obs, sim):
    # NaN이나 무한대 값 처리
    sim = sim[~np.isnan(sim)]
    obs = obs[~np.isnan(obs)]
    if len(obs) == 0 or np.std(obs) == 0 or np.std(sim) == 0: return -np.inf
    
    r = np.corrcoef(sim, obs)[0, 1]
    # np.corrcoef가 NaN을 반환하는 경우 대비
    if np.isnan(r): return -np.inf 
    
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)


# --------------------
# 2. 데이터 로드 및 전처리
# --------------------

# ⭐ 학습 데이터: 2014_2020_시계열_지하수_기상_train.csv (파일 경로는 환경에 맞게 조정하세요)
ddf_train = pd.read_csv("total_rename_data_trainData.csv", encoding='utf-8')
# ⭐ 추론 데이터: TST_test1.csv
test_df_raw = pd.read_csv('trainData.csv') 

# 날짜 변환 및 결측치 처리
ddf_train["ymd"] = pd.to_datetime(ddf_train["ymd"])
test_df_raw["ymd"] = pd.to_datetime(test_df_raw["ymd"])
ddf_train = ddf_train.ffill()
test_df_raw = test_df_raw.ffill()

ddf_train['code_new'] = ddf_train['code_new'].astype(str)
test_df_raw['code_new'] = test_df_raw['code_new'].astype(str)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_preds_df = None 

# 특성 열 목록 (ddf_train 기반)
feature_cols = [col for col in ddf_train.columns if col not in ["ymd", "code_new", "elev"]]
INPUT_LEN = 24
PRE_STEPS = INPUT_LEN # 이전 시계열 데이터 연결 길이

# ⭐ 최종 제출에 사용될 코드와 열 이름을 매핑 (학습 데이터의 순서를 기준으로 1부터 12까지 부여)
all_train_codes = sorted(ddf_train['code_new'].unique().tolist())
code_mapping = {code_name: str(i + 1) for i, code_name in enumerate(all_train_codes)}


# --------------------
# 4. 관측소(code_new)별 학습 및 추론 루프
# --------------------

for code, df_train_full in ddf_train.groupby('code_new'):
    # 해당 코드가 최종 제출 12개 코드에 포함되지 않으면 건너뜁니다.
    if code not in code_mapping:
        continue

    print(f"\n=======================================================")
    print(f"--- Processing code_new: {code} (-> Column {code_mapping[code]}) ---")
    
    # 1. 스케일링 설정 및 학습 데이터셋 준비
    df_train_full = df_train_full.sort_values(by="ymd").reset_index(drop=True)
    train_features = df_train_full[feature_cols].values
    train_target = df_train_full[["elev"]].values
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_train_features = feature_scaler.fit_transform(train_features)
    scaled_train_target = target_scaler.fit_transform(train_target) # target_scaler는 fit_transform 사용

    
    # Encoder-only 모델의 학습 데이터셋
    dataset = TimeSeriesDataset(
        x_data=scaled_train_features,
        y_data=scaled_train_target,
        dates=df_train_full["ymd"].values,
        input_len=INPUT_LEN, pred_len=1, is_inference=False
    )

    # 학습 데이터 로더 (전체 데이터셋 사용)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # 2. 추론 데이터 구성 (오버랩 데이터 병합)
    # 추론을 위해 마지막 24개 시점 데이터 (elev 포함)를 가져와 테스트 데이터와 합칩니다.
    df_train_overlap = df_train_full.tail(PRE_STEPS).reset_index(drop=True)
    
    # test_df_raw 사용 (누락 처리 로직 제거 반영)
    df_test_raw_code = test_df_raw[test_df_raw['code_new'] == code].copy().reset_index(drop=True)
    
    # 학습 데이터의 마지막 부분 + 테스트 데이터를 합쳐서 추론용 전체 시계열 구성
    df_inference_merged = pd.concat([
        df_train_overlap[['ymd'] + feature_cols + ['elev']], 
        df_test_raw_code[['ymd'] + feature_cols]
    ], ignore_index=True)
    
    # 추론 데이터셋의 target(elev)을 채워 넣음 (단순히 이전 값으로 ffill)
    df_inference_merged['elev'] = df_inference_merged['elev'].ffill().fillna(0) 

    inference_features = df_inference_merged[feature_cols].values
    inference_target = df_inference_merged[["elev"]].values
    
    # feature_scaler와 target_scaler는 train 데이터로 fit된 것을 사용해야 함
    scaled_inference_features = feature_scaler.transform(inference_features)
    scaled_inference_target = target_scaler.transform(inference_target)

    inference_dataset = TimeSeriesDataset(
        x_data=scaled_inference_features, 
        y_data=scaled_inference_target, # 타겟값은 사용되지 않지만, 데이터셋 구조를 맞춤
        dates=df_inference_merged["ymd"].values,
        input_len=INPUT_LEN, pred_len=1, is_inference=True
    )
    inference_loader = DataLoader(inference_dataset, batch_size=128, shuffle=False)
    
    # 3. 모델 학습 (제공된 코드와 동일)
    model = TimeSeriesTransformer(feature_size=scaled_train_features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y.view(-1, 1))
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}") # 학습 중 출력은 주석 처리
    
    # 4. 검증 데이터 평가 및 점수 확인 (NSE, KGE) - 제거됨
    
    # 5. 최종 추론 (Inference)
    print(f"--- Generating Predictions for {code} ---")
    preds_list, dates_list = [], []
    
    model.eval() # 추론 전에 모델을 평가 모드로 전환
    with torch.no_grad():
        for x, _, pred_date in inference_loader:
            x = x.to(device)
            pred = model(x)
            
            preds_list.extend(pred.view(-1, 1).cpu().numpy())
            # 추론 데이터셋에서 예측 시점의 날짜를 가져옵니다.
            dates_list.extend(pred_date)

    # 6. 결과 정리 및 병합
    preds_arr = target_scaler.inverse_transform(np.array(preds_list))
    
    # 예측 날짜는 inference_loader에서 가져온 날짜 (다음 시점)
    final_dates = pd.to_datetime([d for d in dates_list if d is not None])
    final_preds = preds_arr[:len(final_dates)] # 날짜 개수에 맞게 예측 값 자르기

    # ⭐ 열 이름을 제출 형식의 숫자 문자열로 사용 ('1' ~ '12')
    column_name_for_submission = code_mapping[code]
    
    current_preds_df = pd.DataFrame({'ymd': final_dates, column_name_for_submission: final_preds.flatten()})
    current_preds_df = current_preds_df.set_index('ymd')
    
    if all_preds_df is None:
        all_preds_df = current_preds_df
    else:
        all_preds_df = all_preds_df.merge(current_preds_df, left_index=True, right_index=True, how='outer')


print("\n=======================================================")
print("=== Inference Complete and Formatting Final Output ===")

# 7. ⭐ 최종 제출 형식 정리 (열 순서 맞추기)
all_preds_df = all_preds_df.reset_index()

# 제출에 필요한 최종 열 목록 ('ymd', '1', '2', ..., '12')
submission_cols = ['ymd'] + [str(i) for i in range(1, 13)]

# 최종 DataFrame 생성: 필요한 열만 선택하고 순서를 맞춤 (누락된 코드는 NaN이 됨)
final_output = all_preds_df[all_preds_df.columns.intersection(submission_cols)]
final_output = final_output.reindex(columns=submission_cols)
final_output = final_output.sort_values(by='ymd')


print("\n--- Final Output (First 5 Rows - Submission Format) ---")
print(final_output.head())

# 최종 결과 파일을 저장하려면 아래 주석을 해제하세요.
final_output.to_csv("TST_test2.csv", index=False, encoding='utf-8')
