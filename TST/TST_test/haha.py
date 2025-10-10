import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
# 위치 인코딩 클래스
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
# TimeSeriesDataset 클래스
# --------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1, ar_lag=8):
        self.x_data = x_data
        self.y_data = y_data
        self.input_len = input_len
        self.pred_len = pred_len
        self.dates = [str(d) for d in dates]
        self.ar_lag = ar_lag

    def __len__(self):
        required_len = max(self.input_len, self.ar_lag)
        if len(self.x_data) < required_len + self.pred_len:
            return 0
        return len(self.x_data) - required_len - self.pred_len + 1
        
    def __getitem__(self, idx):
        start_offset = max(0, self.ar_lag - self.input_len)
        actual_idx = idx + start_offset

        encoder_input = self.x_data[actual_idx : actual_idx + self.input_len]
        sos_index = actual_idx + self.input_len - self.ar_lag 
        decoder_input = self.y_data[sos_index].reshape(1, -1)
        date = self.dates[actual_idx + self.input_len]
        target_output = self.y_data[actual_idx + self.input_len : actual_idx + self.input_len + self.pred_len]
        y_tensor = torch.tensor(target_output, dtype=torch.float32)

        return (
            torch.tensor(encoder_input, dtype=torch.float32), 
            torch.tensor(decoder_input, dtype=torch.float32), 
            y_tensor,
            date
        )

# --------------------
# Transformer 모델 클래스
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

# --- 평가 함수 ---
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

# --- 데이터 로드 및 전역 설정 ---
ddf = pd.read_csv("total_rename_data/trainData.csv")
ddf["ymd"] = pd.to_datetime(ddf["ymd"])
ddf = ddf.ffill().bfill()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 관측소별 학습 및 평가 루프 ---
for code, df_train in ddf.groupby('code_new'):
    print(f"\n=======================================================")
    print(f"--- Processing code_new: {code} ---")
    
    if len(df_train) < 50:
        print(f"Warning: Not enough data for code_new {code} ({len(df_train)} rows). Skipping.")
        continue

    # --- 스케일링 설정 ---
    features_df = df_train.drop(columns=["ymd", "code_new", "elev"])
    train_features = features_df.values
    train_target = df_train[["elev"]].values
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_train_features = feature_scaler.fit_transform(train_features)
    scaled_train_target = target_scaler.fit_transform(train_target)
    
    # --- Dataset 및 DataLoader 생성 ---
    full_train_dataset = TimeSeriesDataset(
        x_data=scaled_train_features, y_data=scaled_train_target, dates=df_train["ymd"].values,
        input_len=24, pred_len=1, ar_lag=8
    )
    
    train_size = int(len(full_train_dataset) * 0.85)
    val_size = len(full_train_dataset) - train_size
    
    if val_size == 0:
        print(f"Warning: Cannot create a validation set for code_new {code}. Skipping.")
        continue

    train_subset, val_subset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    # --- 모델 학습 ---
    model = TimeSeriesTransformerEncoderDecoder(feature_size=scaled_train_features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    epochs = 2
    
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
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

    # ==========================================================
    #               ⭐ 1. 변수 중요도(PFI) 계산 및 시각화 ⭐
    # ==========================================================
    print("\n--- Calculating Permutation Feature Importance ---")
    model.eval()

    # 기준 성능(손실) 측정
    original_loss = 0
    with torch.no_grad():
        for src, tgt_init, y, _ in val_loader:
            src, y, tgt_init = src.to(device), y.to(device), tgt_init.to(device)
            preds = model(src, tgt_init) 
            loss = criterion(preds, y.view(-1, 1))
            original_loss += loss.item()
    original_loss /= len(val_loader)
    print(f"Original Validation Loss: {original_loss:.6f}")

    # 각 변수별 중요도 계산
    importances = {}
    feature_names = features_df.columns

    original_val_src_list = [batch[0] for batch in val_loader]
    original_val_src = torch.cat(original_val_src_list, dim=0).numpy()

    for i, name in enumerate(feature_names):
        permuted_loss = 0
        permuted_src = original_val_src.copy()
        np.random.shuffle(permuted_src[:, :, i])
        permuted_src_tensor = torch.tensor(permuted_src, dtype=torch.float32)
        permuted_dataset = torch.utils.data.TensorDataset(permuted_src_tensor)
        permuted_loader = DataLoader(permuted_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for (perm_src_batch,), (_, tgt_init_batch, y_batch, _) in zip(permuted_loader, val_loader):
                perm_src_batch, tgt_init_batch, y_batch = perm_src_batch.to(device), tgt_init_batch.to(device), y_batch.to(device)
                preds = model(perm_src_batch, tgt_init_batch)
                loss = criterion(preds, y_batch.view(-1, 1))
                permuted_loss += loss.item()
        
        permuted_loss /= len(val_loader)
        importance = permuted_loss - original_loss
        importances[name] = importance
        print(f"  Importance of '{name}': {importance:.6f}")

    # 중요도 시각화
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_importances]
    scores = [item[1] for item in sorted_importances]

    plt.figure(figsize=(12, 8))
    plt.barh(names, scores)
    plt.xlabel("Importance (Validation Loss Increase)")
    plt.title(f"Permutation Feature Importance for code_new: {code}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # ==========================================================
    #               ⭐ 2. 최종 성능 평가 (NSE/KGE) ⭐
    # ==========================================================
    print(f"\n--- Evaluating Model for {code} on Validation Set ---")
    model.eval()
    preds_list, actuals_list = [], []

    with torch.no_grad():
        for src, tgt_init, y, d_batch in val_loader:
            src, tgt_init, y = src.to(device), tgt_init.to(device), y.to(device)
            pred = model(src, tgt_init) 
            preds_list.extend(pred.view(-1, 1).cpu().numpy())
            actuals_list.extend(y.view(-1, 1).cpu().numpy())

    # 스케일 역변환
    preds_arr = target_scaler.inverse_transform(np.array(preds_list))
    actuals_arr = target_scaler.inverse_transform(np.array(actuals_list))

    # NSE, KGE 점수 계산 및 출력
    nse_score = nse(actuals_arr.flatten(), preds_arr.flatten())
    kge_score = kge(actuals_arr.flatten(), preds_arr.flatten())

    print(f"Validation NSE for code {code}: {nse_score:.4f}")
    print(f"Validation KGE for code {code}: {kge_score:.4f}")

print("\n=======================================================")
print("=== All Evaluations Complete ===")