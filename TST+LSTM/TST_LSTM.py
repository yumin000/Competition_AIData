import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import random

# --- (기존의 PositionalEncoding, TimeSeriesDataset, Utility 함수는 그대로 사용) ---

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
        pe = pe.unsqueeze(0).transpose(0, 1) # (S, 1, E)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# --------------------
# 시계열 데이터셋 클래스 (기존 코드와 동일)
# --------------------

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
        encoder_input = self.x_data[idx : idx + self.input_len]
        decoder_input = self.y_data[idx + self.input_len - 1].reshape(1, -1)
        target_output = self.y_data[idx + self.input_len : idx + self.input_len + self.pred_len]
        date = self.dates[idx + self.input_len]
        
        return (
            torch.tensor(encoder_input, dtype=torch.float32), 
            torch.tensor(decoder_input, dtype=torch.float32), 
            torch.tensor(target_output, dtype=torch.float32), 
            date
        )


# --------------------
# 🌟 트랜스포머 + LSTM 하이브리드 모델 클래스
# --------------------

class TimeSeriesHybridModel(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.1, pred_len=1):
        super(TimeSeriesHybridModel, self).__init__()
        
        # 1. 트랜스포머 인코더 파트
        self.encoder_embedding = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. LSTM 파트 (Transformer의 최종 출력을 받음)
        # input_size는 d_model, hidden_size도 d_model로 설정하여 가볍게 유지
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=1, # 가장 가벼운 1개 레이어
            batch_first=False 
        )
        
        # 3. 최종 출력 레이어 (LSTM의 hidden_size -> pred_len)
        self.fc_out = nn.Linear(d_model, pred_len) 
        self.d_model = d_model

    def forward(self, x):
        # x: 인코더 입력 (B, S, F) -> (S, B, F)로 변환됨
        
        # 1. 인코더 임베딩 및 위치 인코딩
        x = self.encoder_embedding(x) * math.sqrt(self.d_model)
        x = x.permute(1, 0, 2) # (S, B, E)
        x = self.pos_encoder(x) 
        
        # 2. 트랜스포머 인코더 통과
        # memory: (S, B, E)
        memory = self.transformer_encoder(x) 
        
        # 3. LSTM 통과
        # LSTM은 장기 의존성을 포착하기 위해 시퀀스 전체를 입력으로 받습니다.
        # output: (S, B, H=E), (hn, cn): 최종 은닉 상태
        lstm_output, (hn, cn) = self.lstm(memory) 
        
        # 4. 최종 예측: LSTM의 마지막 시점의 출력 (혹은 최종 은닉 상태 hn[0]) 사용
        # hn[0]: (B, H) - hn의 첫 번째 레이어 출력
        # 여기서는 hn[0]을 최종 특징으로 사용합니다.
        out = hn.squeeze(0) # (1, B, E) -> (B, E)
        
        # 5. 선형 변환을 통한 최종 예측
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
# 데이터 로드 및 학습 실행 (모델 교체)
# --------------------


ddf = pd.read_csv("TST/TST_test/code_4.csv")

    
ddf["ymd"] = pd.to_datetime(ddf["ymd"])
ddf = ddf.ffill()

for code,df in ddf.groupby('code_new'):
    df = df.sort_values(by="ymd").reset_index(drop=True)
    
    features = df.drop(columns=["ymd", "code_new", "elev"]).values
    target = df[["elev"]].values
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # TimeSeriesDataset을 그대로 사용하지만, 디코더 입력(tgt)은 사용하지 않습니다.
    dataset = TimeSeriesDataset(
        x_data=scaled_features,
        y_data=scaled_target,
        dates=df["ymd"].values,
        input_len=24, pred_len=1
    )
    
    train_size = int(len(dataset) * 0.8)
    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))
    
    # DataLoader는 여전히 (src, tgt, y, date) 4개의 값을 반환하지만, src와 y만 사용합니다.
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🌟 모델 교체: TimeSeriesHybridModel 사용
    model = TimeSeriesHybridModel(feature_size=scaled_features.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    epochs = 5
    
    print(f"\n--- Training for code_new: {code} (Transformer + LSTM Hybrid Model) ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        # src: 인코더 입력, y: 정답. tgt는 사용하지 않으므로 무시.
        for src, _, y, _ in train_loader: 
            src, y = src.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 예측
            preds = model(src) 
            
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

    # --------------------
    # 평가 및 예측
    # --------------------
    model.eval()
    preds_list, actuals_list, dates_list = [], [], []

    with torch.no_grad():
        for src, _, y, d_batch in test_loader:
            src, y = src.to(device), y.to(device)
            
            # 예측
            pred = model(src) 
            
            # 리스트에 저장
            preds_list.extend(pred.view(-1, 1).cpu().numpy())
            actuals_list.extend(y.view(-1, 1).cpu().numpy())
            dates_list.extend(d_batch)

    # 스케일 역변환
    preds_arr = target_scaler.inverse_transform(np.array(preds_list))
    actuals_arr = target_scaler.inverse_transform(np.array(actuals_list))

    dates_arr = pd.to_datetime(dates_list[:len(preds_arr)])

    # 결과 출력
    print("\n--- Evaluation Metrics ---")
    print(f"NSE: {nse(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
    print(f"KGE: {kge(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
    
    # 시각화 (선택 사항)
    plt.figure(figsize=(12, 6))
    plt.plot(dates_arr, actuals_arr.flatten(), label='Actual (elev)', color='blue')
    plt.plot(dates_arr, preds_arr.flatten(), label='Prediction (elev)', color='red', linestyle='--')
    plt.title(f'Transformer + LSTM Hybrid Prediction vs Actual (Code: {code})')
    plt.xlabel('Date')
    plt.ylabel('elev (Inverse Scaled)')
    plt.legend()
    plt.show()