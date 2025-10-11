import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import random



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #shape=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #오버플로우 예방 exp+ln
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #shape=(max_len,d_model)
        pe = pe.unsqueeze(0).transpose(0, 1) #shape=(max_len,1,d_model)
        self.register_buffer('pe', pe) #위치 인코더 값 고정 -> 버퍼저장
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1):
        self.x_data = x_data
        self.y_data = y_data
        self.input_len = input_len
        self.pred_len = pred_len
        self.dates = [str(d) for d in dates] #날짜 문자열로 변경

    def __len__(self):
        return len(self.x_data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.x_data[idx:idx+self.input_len]
        y = self.y_data[idx+self.input_len:idx+self.input_len+self.pred_len]
        date = self.dates[idx+self.input_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), date
#이때, Q,K,V 벡터의 차원(논문에서는 64 = d model(512) / num_heads(8) ) < d model의 차원 (논문에서는 512)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.1, pred_len=1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(feature_size, d_model) #특징 차원을 d_model 과 동일하게 바꿔줌
        self.pos_encoder = PositionalEncoding(d_model) #위치 인코딩
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False) # 단일 레이어#batch_first = false <- 입력 데이터 차원 순서 (시퀀스 길이,배치크기,특징 차원)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) #지정 수만큼 인코더 레이어 복제
        self.fc_out = nn.Linear(d_model, pred_len) #입력 차원-> 출력 차원 길이로 변환
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model) #임베딩 값(d_model 차원으로 변환) -> 스케일링 : attention score
        x = x.permute(1, 0, 2)#입력텐서 차원 순서 변경
        x = self.pos_encoder(x) #임베딩 정보 더하기
        out = self.transformer_encoder(x) #전체 인코더레이어 통과
        out = out[-1, :, :] #마지막 시점의 출력값 (24일 간의 데이터를 모두 합쳐놓은 값)
        out = self.fc_out(out) #차원 변경
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


ddf = pd.read_csv("TST/TST_test/code_4.csv")
test_df=pd.read_csv("total_rename_data/test_inputs.csv")
ddf["ymd"] = pd.to_datetime(ddf["ymd"])

ddf = ddf.ffill()

for code,df in ddf.groupby('code_new'):
    df = df.sort_values(by="ymd").reset_index(drop=True)
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(feature_size=scaled_features.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3) #손실함수 안줄어 들면 rate 줄이기
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
                print(f"Epoch {epoch+1}: Loss is NaN. Skipping update.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        #scheduler.step(avg_loss)

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
# final_output.to_csv("submission_final_predictions.csv", index=False, encoding='utf-8')