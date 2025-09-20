import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

# (PositionalEncoding, TimeSeriesTransformer, nse, kge 함수는 이전과 동일)
# ...
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
# Dataset 정의 (✨ 수정됨)
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
        x = self.x_data[idx:idx+self.input_len]
        y = self.y_data[idx+self.input_len:idx+self.input_len+self.pred_len]
        
        # ✨ 해결 1: 날짜를 리스트가 아닌 문자열로 바로 반환 (pred_len=1 이므로)
        date = self.dates[idx+self.input_len]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), date

# --------------------
# Transformer 모델 정의
# --------------------
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
        out = out[-1, :, :]
        out = self.fc_out(out)
        return out

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


# (데이터 불러오기 및 분할 부분은 이전과 동일)
# ...
df = pd.read_csv("TST/code_4.csv")
df["ymd"] = pd.to_datetime(df["ymd"])
df = df.sort_values(by="ymd").reset_index(drop=True)
df = df.ffill().bfill() 
features = df.drop(columns=["ymd", "code_new", "elev"])
target = df[["elev"]].values
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target)
dataset = TimeSeriesDataset(
    x_data=scaled_features,
    y_data=scaled_target,
    dates=df["ymd"].values,
    input_len=24, pred_len=1
)
train_size = int(len(dataset) * 0.8)
train_dataset = Subset(dataset, range(train_size))
test_dataset = Subset(dataset, range(train_size, len(dataset)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# (학습 준비 및 학습 루프 부분은 이전과 동일)
# ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(feature_size=scaled_features.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
epochs = 2
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
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
# 평가 및 예측 (✨ 수정됨)
# --------------------
model.eval()
preds_list, actuals_list, dates_list = [], [], []

with torch.no_grad():
    for x, y, d_batch in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        
        preds_list.extend(pred.view(-1, 1).cpu().numpy())
        actuals_list.extend(y.view(-1, 1).cpu().numpy())
        
        # ✨ 해결 2: d_batch는 이제 날짜 문자열의 튜플이므로 바로 extend
        dates_list.extend(d_batch)

preds_arr = target_scaler.inverse_transform(np.array(preds_list))
actuals_arr = target_scaler.inverse_transform(np.array(actuals_list))

# 이제 dates_list는 문자열 리스트이므로 오류 없이 변환됩니다.
dates_arr = pd.to_datetime(dates_list[:len(preds_arr)])

# (이후 시각화 및 저장 코드는 이전과 동일)
# ...
print("NSE:", nse(actuals_arr.flatten(), preds_arr.flatten()))
print("KGE:", kge(actuals_arr.flatten(), preds_arr.flatten()))

plt.figure(figsize=(15, 7))
plt.plot(dates_arr, actuals_arr, '.-', label="Actual (elev)")
plt.plot(dates_arr, preds_arr, '.--', label="Predicted (elev)")
plt.xlabel("Date")
plt.ylabel("Elev")
plt.title("Elev Prediction (Test Set: Last 20%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

result_df = pd.DataFrame({
    "date": dates_arr,
    "actual_elev": actuals_arr.flatten(),
    "predicted_elev": preds_arr.flatten()
})
result_df.to_csv("elev_predictions_last_20_percent.csv", index=False)


