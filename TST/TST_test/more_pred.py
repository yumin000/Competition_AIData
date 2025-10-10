import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import random

# --- (기존의 PositionalEncoding 및 Utility 함수는 그대로 사용) ---

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
# 시계열 데이터셋 클래스 (⭐ 디코더 입력: 8일 전 elev 값 사용)
# --------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1, ar_lag=8*24): # ⭐ ar_lag 추가
        self.x_data = x_data
        self.y_data = y_data
        self.input_len = input_len
        self.pred_len = pred_len
        self.dates = [str(d) for d in dates]
        self.ar_lag = ar_lag # 8일 전 (AR 초기값 딜레이)

    def __len__(self):
        # 인코더 입력(input_len) + 예측 길이(pred_len)를 고려
        # 디코더 초기값 (SOS 역할)이 ar_lag 만큼 더 뒤에서 시작할 필요는 없으므로,
        # 기존 계산 방식을 유지합니다.
        # 즉, 0번 인덱스부터 시작해서 input_len + pred_len - 1 까지 사용합니다.
        return len(self.x_data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 인코더 입력 (Source Sequence) - 과거 특징 데이터
        encoder_input = self.x_data[idx : idx + self.input_len]
        
        # ⭐ 디코더 입력 (tgt_input) - SOS 역할을 하는 값
        # 예측해야 할 시점 (idx + input_len)의 8일 전 값 (elev)
        # 예측 시점 T = idx + input_len
        # 디코더 입력 (SOS) 시점 = T - ar_lag 
        
        sos_index = idx + self.input_len - self.ar_lag 
        
        # sos_index가 0보다 작으면 데이터가 충분하지 않음 -> 이 경우 제외하거나 다른 값 사용해야 함
        # 현재 __len__ 설정상 0 <= idx 이므로, sos_index >= input_len - ar_lag 가 됩니다.
        # 인코더 입력 길이 (input_len=24)가 AR_LAG (8)보다 충분히 크므로 일반적으로 문제 없음
        if sos_index < 0:
             # 만약 데이터가 충분치 않아 인덱스 에러가 발생하면, SOS 토큰 역할을 하는 0 값 사용
             # 하지만 현재 설정에서는 idx=0일 때도 sos_index=16이므로 문제 없음
             decoder_input = np.zeros_like(self.y_data[0].reshape(1, -1))
        else:
             decoder_input = self.y_data[sos_index].reshape(1, -1)
        
        # 정답 (Target Output)
        target_output = self.y_data[idx + self.input_len : idx + self.input_len + self.pred_len]
        
        # 예측해야 할 시점의 날짜
        date = self.dates[idx + self.input_len]
        
        return (
            torch.tensor(encoder_input, dtype=torch.float32), 
            torch.tensor(decoder_input, dtype=torch.float32), # SOS 역할의 8일 전 elev
            torch.tensor(target_output, dtype=torch.float32), 
            date
        )

# --------------------
# 트랜스포머 인코더-디코더 클래스 (기존 코드와 동일)
# --------------------
# (클래스 정의는 기존 코드와 동일합니다. AR 방식으로 동작하는 것은 학습/추론 루프에 의해 결정됩니다.)
class TimeSeriesTransformerEncoderDecoder(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.1, pred_len=1):
        super(TimeSeriesTransformerEncoderDecoder, self).__init__()
        self.encoder_embedding = nn.Linear(feature_size, d_model)
        self.decoder_embedding = nn.Linear(1, d_model) 
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False # (S, B, E)
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
        out = self.fc_out(out) 
        return out

# --- (기존의 nse, kge 함수는 그대로 사용) ---

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
# 데이터 로드 및 학습 실행
# --------------------

# (임시로 파일을 가정하고 실행)
ddf = pd.read_csv("TST/TST_test/code_4.csv")

ddf["ymd"] = pd.to_datetime(ddf["ymd"])
ddf = ddf.ffill()

for code,df in ddf.groupby('code_new'):
    df = df.sort_values(by="ymd").reset_index(drop=True)
    
    # 특징(X)과 타겟(Y) 정의
    features = df.drop(columns=["ymd", "code_new", "elev"]).values
    target = df[["elev"]].values
    
    # 스케일러 정의 및 적용
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # Dataset 생성 (⭐ ar_lag=8 설정)
    dataset = TimeSeriesDataset(
        x_data=scaled_features,
        y_data=scaled_target,
        dates=df["ymd"].values,
        input_len=24, 
        pred_len=1,
    )
    
    # 데이터셋 분리 및 DataLoader 설정
    train_size = int(len(dataset) * 0.8)
    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 정의
    model = TimeSeriesTransformerEncoderDecoder(feature_size=scaled_features.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    
    print(f"\n--- Training for code_new: {code} (AR-like Transfomer with 8-day lag SOS) ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for src, tgt, y, _ in train_loader:
            # tgt: 8일 전 elev 값 (SOS 역할)
            src, tgt, y = src.to(device), tgt.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 예측 (tgt_mask는 pred_len=1 이므로 [0]만 포함하여 큰 영향 없음)
            preds = model(src, tgt) 
            
            # 손실 계산
            loss = criterion(preds, y.view(-1, 1))
            
            if torch.isnan(loss):
                print(f"Epoch {epoch+1}: Loss is NaN. Skipping update.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

    # --------------------
    # 평가 및 예측 (추론 모드는 기존과 동일, 단일 스텝 예측)
    # --------------------
    model.eval()
    preds_list, actuals_list, dates_list = [], [], []

    with torch.no_grad():
        for src, tgt_init, y, d_batch in test_loader:
            src, y = src.to(device), y.to(device)
            
            # 추론 시에도 tgt_init (8일 전 elev)를 디코더의 초기 입력으로 사용
            
            # 1. 인코더 처리
            memory = model.transformer.encoder(model.pos_encoder(model.encoder_embedding(src).permute(1, 0, 2)))
            
            # 2. 디코더 초기 입력 설정: tgt_init은 8일 전 elev 값
            decoder_input_sequence = tgt_init.to(device) # (B, 1, 1)
            
            # 3. 임베딩 및 위치 인코딩
            tgt_emb = model.decoder_embedding(decoder_input_sequence) * math.sqrt(model.d_model)
            tgt_emb = tgt_emb.permute(1, 0, 2) # (S=1, B, E)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            # 4. 디코더 처리
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(device)
            out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask) 
            
            # 5. 최종 예측
            pred = model.fc_out(out[-1, :, :]) 
            
            # 리스트에 저장
            preds_list.extend(pred.view(-1, 1).cpu().numpy())
            actuals_list.extend(y.view(-1, 1).cpu().numpy())
            dates_list.extend(d_batch)

    # 스케일 역변환 및 평가
    preds_arr = target_scaler.inverse_transform(np.array(preds_list))
    actuals_arr = target_scaler.inverse_transform(np.array(actuals_list))

    dates_arr = pd.to_datetime(dates_list[:len(preds_arr)])

    # 결과 출력
    print("\n--- Evaluation Metrics ---")
    print(f"{code} NSE: {nse(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
    print(f"{code} KGE: {kge(actuals_arr.flatten(), preds_arr.flatten()):.4f}")