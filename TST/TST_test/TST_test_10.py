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
# 시계열 데이터셋 클래스 (디코더 입력 처리를 위해 x_data에 target_data도 포함시킴)
# --------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, dates, input_len=24, pred_len=1):
        self.x_data = x_data
        self.y_data = y_data
        self.input_len = input_len
        self.pred_len = pred_len
        self.dates = [str(d) for d in dates]

    def __len__(self):
        # 인코더 입력 (input_len) + 디코더 입력 (pred_len)을 고려해야 함
        # 여기서는 인코더 입력 (x_data)이 시퀀스 길이를 결정
        return len(self.x_data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 인코더 입력 (Source Sequence) - 과거 특징 데이터
        encoder_input = self.x_data[idx : idx + self.input_len]
        
        # 디코더 입력 (Target Sequence) - 예측해야 할 구간의 이전 시점 값 (Teacher Forcing용)
        # 시계열 예측에서는 보통 'input_len'의 마지막 시점 값 + 예측 구간의 실제 값(정답)을 사용
        # 여기서는 단순화를 위해 input_len의 마지막 값(전 시점)을 디코더의 첫 입력으로 간주합니다.
        # 실제 예측 값은 y_data에서 가져옵니다.
        
        # Decoder Input (tgt_input): 인코더 마지막 시점 + 예측해야 할 시점들
        # 학습 시에는 정답 데이터 (y_data)를 인코더의 마지막 시점과 이어 붙여 사용합니다.
        # [End_of_Input_State] + [Target_1, Target_2, ..., Target_{pred_len-1}]
        # 여기서는 input_len의 마지막 값(scalar)을 디코더의 첫 입력으로 간주하고,
        # 이후 예측할 값들을 y_data에서 가져옵니다.

        # 디코더 입력 (tgt_input)
        # 인코더 시퀀스의 마지막 타겟 값(y_data)을 가져와 디코더의 첫 입력(SOS 역할)으로 사용
        # y_data[idx + input_len - 1] (이전 시점의 정답) + y_data[idx + input_len : idx + input_len + pred_len - 1]
        
        # 단순화를 위해, 인코더의 마지막 입력 시점의 target 값(y_data)을 가져와 SOS 역할로 사용
        # 이후 예측은 1개이므로 target_sequence는 SOS 역할의 1개 값만 가집니다.
        
        # SOS (Start of Sequence) 역할을 하는 디코더 입력: 예측 시작 시점의 Target 값
        decoder_input = self.y_data[idx + self.input_len - 1].reshape(1, -1)
        
        # 정답 (Target Output)
        target_output = self.y_data[idx + self.input_len : idx + self.input_len + self.pred_len]
        
        date = self.dates[idx + self.input_len]
        
        return (
            torch.tensor(encoder_input, dtype=torch.float32), 
            torch.tensor(decoder_input, dtype=torch.float32), 
            torch.tensor(target_output, dtype=torch.float32), 
            date
        )

# --------------------
# 트랜스포머 인코더-디코더 클래스
# --------------------

class TimeSeriesTransformerEncoderDecoder(nn.Module):
    def __init__(self, feature_size, d_model=32, nhead=4, num_layers=1, dropout=0.1, pred_len=1):
        super(TimeSeriesTransformerEncoderDecoder, self).__init__()
        
        # 인코더 입력 (Feature Size) 임베딩
        self.encoder_embedding = nn.Linear(feature_size, d_model)
        
        # 디코더 입력 (Target Value, 여기서는 1차원) 임베딩
        self.decoder_embedding = nn.Linear(1, d_model) 
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # PyTorch 기본 Transformer 모델 사용
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False # (S, B, E)
        )
        
        # 최종 출력 레이어 (d_model -> pred_len)
        self.fc_out = nn.Linear(d_model, pred_len) 
        self.d_model = d_model

    def forward(self, src, tgt, tgt_mask=None):
        # 1. 임베딩 및 스케일링
        # src: 인코더 입력 (Features)
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        # tgt: 디코더 입력 (Target Value)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)

        # 2. Permute: (B, S, E) -> (S, B, E)
        # PyTorch Transformer는 기본적으로 (S, B, E) 순서를 요구합니다.
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        
        # 3. 위치 인코딩
        src = self.pos_encoder(src) 
        tgt = self.pos_encoder(tgt)

        # 4. Transformer 실행
        # tgt_mask: 디코더에서 미래 정보를 보지 못하게 하는 Lookahead Mask (필수)
        # 여기서는 pred_len=1 이므로 마스크는 [0]만 포함하는 단일 값으로 구성됩니다.
        if tgt_mask is None:
            # tgt의 시퀀스 길이만큼 마스크 생성
            tgt_seq_len = tgt.size(0)
            # 마스크는 (S, S) 크기의 상삼각 행렬로, 미래 정보를 차단합니다.
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
            
        # out: 디코더의 최종 출력 (S_tgt, B, E)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask) 
        
        # 5. 최종 출력: 디코더의 마지막 시점 (tgt 시퀀스의 마지막 요소)만 사용
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
# 데이터 로드 및 학습 실행 (모델 교체)
# --------------------

# (임시로 파일을 가정하고 실행)
# 주의: 이 코드를 실제 환경에서 실행하려면 'TST/TST_test/code_4.csv' 파일이 필요합니다.

ddf = pd.read_csv("total_rename_data/trainData.csv")

    
ddf["ymd"] = pd.to_datetime(ddf["ymd"])
ddf = ddf.ffill()

for code,df in ddf.groupby('code_new'):
    if(code==10):
        df = df.sort_values(by="ymd").reset_index(drop=True)
        
        # 특징(X)과 타겟(Y) 정의
        features = df.drop(columns=["ymd", "code_new", "elev"]).values
        target = df[["elev"]].values
        
        # 스케일러 정의 및 적용
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)
        
        # Dataset 생성
        # input_len=24, pred_len=1 (기존과 동일)
        dataset = TimeSeriesDataset(
            x_data=scaled_features,
            y_data=scaled_target, # Target 데이터(y)도 스케일링된 값 사용
            dates=df["ymd"].values,
            input_len=24, pred_len=1
        )
        
        # 데이터셋 분리 및 DataLoader 설정
        train_size = int(len(dataset) * 0.8)
        train_dataset = Subset(dataset, range(train_size))
        test_dataset = Subset(dataset, range(train_size, len(dataset)))
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🌟 모델 교체: TimeSeriesTransformerEncoderDecoder 사용
        model = TimeSeriesTransformerEncoderDecoder(feature_size=scaled_features.shape[1]).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
        epochs = 10
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1) #손실함수 안줄어 들면 rate 줄이기
        
        print(f"\n--- Training for code_new: {code} (Encoder-Decoder Transformer) ---")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for src, tgt, y, _ in train_loader:
                # src: 인코더 입력 (features), tgt: 디코더 입력 (SOS 역할), y: 정답 (Target)
                src, tgt, y = src.to(device), tgt.to(device), y.to(device)
                optimizer.zero_grad()
                
                # 예측 (tgt_mask는 내부에서 생성)
                preds = model(src, tgt) 
                
                # 손실 계산: 예측값과 정답 타겟(y)을 비교
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
            if(epoch>4):
                # --------------------
                # 평가 및 예측 (Encoder-Decoder 추론 모드)
                # --------------------
                model.eval()
                preds_list, actuals_list, dates_list = [], [], []

                with torch.no_grad():
                    for src, tgt_init, y, d_batch in test_loader:
                        src, y = src.to(device), y.to(device)
                        batch_size = src.size(0)
                        
                        # 추론 시에는 디코더의 첫 입력(SOS)만 주고, 이후 예측값을 다시 입력으로 사용
                        # (여기서는 pred_len=1 이므로 한 번의 예측만 수행)
                        
                        # 1. 인코더 처리: 인코더 입력(src)만 전달하여 메모리 생성 (인코더의 역할)
                        memory = model.transformer.encoder(model.pos_encoder(model.encoder_embedding(src).permute(1, 0, 2)))
                        
                        # 2. 디코더 초기 입력 설정: tgt_init은 SOS 역할을 하는 이전 시점의 타겟 값
                        decoder_input_sequence = tgt_init.to(device) # (B, 1, 1)
                        
                        # 3. 임베딩 및 위치 인코딩
                        tgt_emb = model.decoder_embedding(decoder_input_sequence) * math.sqrt(model.d_model)
                        tgt_emb = tgt_emb.permute(1, 0, 2) # (S=1, B, E)
                        tgt_emb = model.pos_encoder(tgt_emb)
                        
                        # 4. 디코더 처리 (크로스 어텐션에 memory 사용)
                        # tgt_mask 생성 (S=1이므로 단순 0)
                        tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(device)
                        
                        # out: (S=1, B, E)
                        out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask) 
                        
                        # 5. 최종 예측
                        pred = model.fc_out(out[-1, :, :]) 
                        
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
                print(f"{code}_{epoch+1}NSE: {nse(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
                print(f"{code}_{epoch+1}KGE: {kge(actuals_arr.flatten(), preds_arr.flatten()):.4f}")
            
        # --------------------
        # 시각화 (선택 사항)
        # --------------------
        plt.figure(figsize=(12, 6))
        plt.plot(dates_arr, actuals_arr.flatten(), label='Actual (elev)', color='blue')
        plt.plot(dates_arr, preds_arr.flatten(), label='Prediction (elev)', color='red', linestyle='--')
        plt.title(f'Encoder-Decoder Transformer Prediction vs Actual (Code: {code})')
        plt.xlabel('Date')
        plt.ylabel('elev (Inverse Scaled)')
        plt.legend()
        plt.show()
        #plt.savefig(f'TST/TST_encoder_decoder {code} graph.png')