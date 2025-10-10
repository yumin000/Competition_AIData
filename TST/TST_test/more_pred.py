import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import math
import random

# --- 시드 설정 ---
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------
# 위치 인코딩 클래스 (기존 코드와 동일)
# --------------------
decoder_input=[105.51818432210463, 201.97753013045494, 54.40671110375093, 77.50952757471778, 80.85326710636139, 140.61231703894475, 125.13273404237557, 224.21731086375632, 214.23033454052356, 27.585659739402935, 2.9331123190800343, 113.38119163648338]
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
        return x + self.pe[:x.size(0), :]

# --------------------
# ⭐ TimeSeriesDataset 클래스 (학습/추론 분리)
# --------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1, ar_lag=8, is_inference=False):
        self.x_data = x_data
        self.y_data = y_data
        self.input_len = input_len
        self.pred_len = pred_len
        self.dates = [str(d) for d in dates]
        self.ar_lag = ar_lag
        self.is_inference = is_inference # 추론 모드 플래그

    def __len__(self):
        # 학습/추론 모두 동일하게 인코더 입력 + 예측 구간 필요
        return len(self.x_data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        encoder_input = self.x_data[idx : idx + self.input_len]
        
        # SOS 역할을 하는 8일 전 elev 값 (Target)
        sos_index = idx + self.input_len - self.ar_lag 
        
        # SOS 인덱스가 0 미만이면 (데이터 시작점)
        if sos_index < 0:
             # 데이터가 충분치 않을 경우, 0 값을 SOS 토큰으로 사용
             # y_data가 스케일링된 상태이므로 0은 평균값을 의미
             decoder_input = np.zeros_like(self.y_data[0].reshape(1, -1))
        else:
             # SOS 토큰으로 사용할 이전 시점의 정답(y_data) 사용
             decoder_input = self.y_data[sos_index].reshape(1, -1)
        
        date = self.dates[idx + self.input_len]
        
        # ⭐ 추론 모드가 아닐 때만 정답(Target Output)을 반환
        if not self.is_inference:
            target_output = self.y_data[idx + self.input_len : idx + self.input_len + self.pred_len]
            y_tensor = torch.tensor(target_output, dtype=torch.float32)
        else:
            # 추론 모드일 경우, 정답은 없으므로 더미 텐서 반환 (학습 시 y 위치에 들어갈 값)
            y_tensor = torch.empty(self.pred_len, 1, dtype=torch.float32)

        return (
            torch.tensor(encoder_input, dtype=torch.float32), 
            torch.tensor(decoder_input, dtype=torch.float32), 
            y_tensor, # 정답 또는 더미
            date
        )

# --------------------
# TimeSeriesTransformerEncoderDecoder, nse, kge 함수는 기존 코드와 동일
# --------------------

class TimeSeriesTransformerEncoderDecoder(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.1, pred_len=1):
        super(TimeSeriesTransformerEncoderDecoder, self).__init__()
        self.encoder_embedding = nn.Linear(feature_size, d_model)
        self.decoder_embedding = nn.Linear(1, d_model) 
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=False 
        )
        self.fc_out = nn.Linear(d_model, pred_len) 
        self.d_model = d_model

    def forward(self, src, tgt, tgt_mask=None):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        src = self.pos_encoder(src) 
        tgt = self.pos_encoder(tgt)

        if tgt_mask is None:
            tgt_seq_len = tgt.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
            
        out = self.transformer(src, tgt, tgt_mask=tgt_mask) 
        out = out[-1, :, :] 
        return self.fc_out(out)

def nse(obs, sim):
    denominator = np.sum((obs - np.mean(obs))**2)
    if denominator == 0: return -np.inf
    return 1 - np.sum((sim - obs)**2) / denominator

def kge(obs, sim):
    if np.std(obs) == 0 or np.std(sim) == 0: return -np.inf
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)


# --------------------
# ⭐ 데이터 로드 및 다중 코드 처리 (추론 전용)
# --------------------

# ⭐ 실제 파일 경로 적용
ddf = pd.read_csv("total_rename_data/trainData.csv")
test_df_raw = pd.read_csv("total_rename_data/test_inputs.csv")

# 날짜 변환 및 결측치 처리
ddf["ymd"] = pd.to_datetime(ddf["ymd"])
test_df_raw["ymd"] = pd.to_datetime(test_df_raw["ymd"])
ddf = ddf.ffill()
test_df_raw = test_df_raw.ffill()

# --------------------
# ⭐ 관측소(code_new)별 학습 및 추론 루프
# --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_preds_df = None # 최종 예측 결과를 저장할 DataFrame

for code, df_train in ddf.groupby('code_new'):
    print(f"\n=======================================================")
    print(f"--- Processing code_new: {code} ---")
    
    # 해당 code_new에 대한 테스트 데이터 분리
    df_test = test_df_raw[test_df_raw['code_new'] == code].copy().reset_index(drop=True)

    if df_test.empty:
        print(f"경고: code_new {code}에 대한 테스트 데이터가 없습니다. 건너뜁니다.")
        continue

    # 1. 스케일링 설정 (학습 데이터 기반)
    train_features = df_train.drop(columns=["ymd", "code_new", "elev"]).values
    train_target = df_train[["elev"]].values
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_train_features = feature_scaler.fit_transform(train_features)
    scaled_train_target = target_scaler.fit_transform(train_target)
    
    # 2. 테스트 데이터 스케일링 (FIT 없이 TRANSFORM만 적용)
    test_features = df_test.drop(columns=["ymd", "code_new"]).values # feature만 추출
    
    # ⭐ 테스트 데이터에는 elev가 없으므로, 더미 elev 열을 추가하여 스케일링 파이프라인 유지
    # 이는 오직 target_scaler.transform 호출을 위한 형태 맞추기입니다.
    test_target_dummy = np.full((len(df_test), 1), decoder_input[code-1])
    
    scaled_test_features = feature_scaler.transform(test_features)
    scaled_test_target_dummy = target_scaler.transform(test_target_dummy) # 이 값은 SOS 토큰으로 사용됨

    # 3. Dataset 및 DataLoader 생성
    # 학습
    train_dataset = TimeSeriesDataset(
        x_data=scaled_train_features, y_data=scaled_train_target, dates=df_train["ymd"].values,
        input_len=24, pred_len=1, ar_lag=8*24, is_inference=False
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    # ⭐ 추론 (is_inference=True)
    inference_dataset = TimeSeriesDataset(
        x_data=scaled_test_features, y_data=scaled_test_target_dummy, dates=df_test["ymd"].values,
        input_len=24, pred_len=1, is_inference=True
    )
    inference_loader = DataLoader(inference_dataset, batch_size=128, shuffle=False)

    # 4. 모델 학습 (이전과 동일)
    model = TimeSeriesTransformerEncoderDecoder(feature_size=scaled_train_features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epochs = 5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    print(f"--- Training Model for {code} ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for src, tgt, y, _ in train_loader:
            src, tgt, y = src.to(device), tgt.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(src, tgt) 
            loss = criterion(preds, y.view(-1, 1))
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

    # 5. ⭐ 추론 (Inference)
    model.eval()
    preds_list, dates_list = [], []

    print(f"--- Generating Predictions for {code} ---")
    with torch.no_grad():
        for src, tgt_init, _, d_batch in inference_loader: # y (정답)은 사용하지 않음
            src = src.to(device)
            
            # 인코더 및 디코더 추론 로직 (이전과 동일)
            memory = model.transformer.encoder(model.pos_encoder(model.encoder_embedding(src).permute(1, 0, 2)))
            decoder_input_sequence = tgt_init.to(device)
            tgt_emb = model.decoder_embedding(decoder_input_sequence) * math.sqrt(model.d_model)
            tgt_emb = model.pos_encoder(tgt_emb.permute(1, 0, 2))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(device)
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask) 
            pred = model.fc_out(out[-1, :, :]) 
            
            preds_list.extend(pred.view(-1, 1).cpu().numpy())
            dates_list.extend(d_batch)

    # 6. ⭐ 스케일 역변환 및 결과 DataFrame 생성
    preds_arr = target_scaler.inverse_transform(np.array(preds_list))
    
    # 예측 결과 DataFrame
    current_preds_df = pd.DataFrame({
        'ymd': pd.to_datetime(dates_list[:len(preds_arr)]),
        str(code): preds_arr.flatten()
    })
    current_preds_df = current_preds_df.set_index('ymd')
    
    # 최종 결과 DataFrame에 병합
    if all_preds_df is None:
        all_preds_df = current_preds_df
    else:
        # ymd를 기준으로 병합 (관측소별 열 추가)
        all_preds_df = all_preds_df.merge(current_preds_df, left_index=True, right_index=True, how='outer')


print("\n=======================================================")
print("=== Inference Complete ===")

# 7. ⭐ 최종 제출 형식에 맞추기
# 인덱스를 열로 변환
all_preds_df = all_preds_df.reset_index()

# 제출 형식이 ymd 다음에 1, 2, ..., 12 열로 정렬되어야 한다고 가정하고
# 'code_new' 열의 이름을 숫자로 변환합니다.
# 현재 모델은 관측소별로 학습되므로, 실제 code_new 이름을 그대로 사용했습니다.
# 만약 제출 파일의 열 이름이 code_new 값이 아닌 '1', '2', ... 라면 추가적인 매핑이 필요합니다.
# 여기서는 예시로 '1', '2', ..., '12' 열을 가진 제출 형식의 틀을 만듭니다.

# (⭐ 여기서는 모든 code_new가 최종 12개 열에 포함된다고 가정)
submission_cols = ['ymd'] + [str(c) for c in ddf['code_new'].unique()] # 예측된 code_new를 열 이름으로 사용
submission_cols.extend([f"Missing_Col_{i}" for i in range(len(submission_cols), 13)]) # 예시로 12개 열 맞추기

# 최종 출력
final_output = all_preds_df
print("\n--- Final Output (First 5 Rows) ---")
print(final_output.head())

# 최종 결과 파일을 저장
final_output.to_csv("TST_test1.csv", index=False)