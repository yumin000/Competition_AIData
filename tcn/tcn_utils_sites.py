# tcn_utils.py
import os, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------- 기본 설정 ----------------------
# 랜덤 시드
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_COL, SITE_COL, TARGET_COL = "ymd", "code_new", "elev"

# site별 데이터 기준 피처 정의
TIME_VARYING_FEATURES = ["wtemp","ec","temp","rainfall","wind","humidity","pressure","gtemp","welect"]
STATIC_NUMERIC_FEATURES = ["lat","lon","level"]

# ---------------------- 데이터 전처리 ----------------------
# CSV 로드 및 정렬
def load_data(path): 
    df=pd.read_csv(path); df[TIME_COL]=pd.to_datetime(df[TIME_COL],errors="coerce")
    return df.sort_values([SITE_COL,TIME_COL]).reset_index(drop=True)   # code와 시간 순으로 정렬

# 하천 이름 인코딩
def encode_river(df):
    if "river" in df.columns: df["river"]=LabelEncoder().fit_transform(df["river"].astype(str))
    return df

# 정제
def coerce_numeric(df, cols):
    """문자열/결측치 → float 변환"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# 주기형(캘린더) 변수 추가
''' 주기성 정보(일, 주, 계절 패턴)를 인코딩한 피쳐
    시간(hour), 요일(dow), 월(month)을 sin/cos으로 변환
    TIME_VARYING_FEATURES에 추가해 TCN에 전달하기 위함
    TCN은 시간적 패턴 + 계절성 주기를 함께 학습할 수 있음 '''
def add_calendar_feats(df):
    """주기형 캘린더 피처 (sin/cos) 추가"""
    df["hour"] = df[TIME_COL].dt.hour
    df["dow"] = df[TIME_COL].dt.dayofweek
    df["month"] = df[TIME_COL].dt.month
    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2*np.pi*df[col]/period)
        df[f"{col}_cos"] = np.cos(2*np.pi*df[col]/period)
    return df

# 피쳐 메트릭스 구성
def build_feature_matrix(df, use_calendar=True):
    """ 단일 site 기준 feature matrix 생성 """
    X_time = df[TIME_VARYING_FEATURES].values.astype(np.float32)
    S_num  = df[STATIC_NUMERIC_FEATURES].values.astype(np.float32)
    river_idx = df["river"].astype(int).values
    site_idx  = df["site_idx"].astype(int).values  # site별 학습이라 거의 0 고정

    if use_calendar:
        cal_cols = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
        X_time = np.hstack([X_time, df[cal_cols].values.astype(np.float32)])

    y = df[TARGET_COL].values.astype(np.float32)

    return X_time, S_num, site_idx, river_idx, y, df[TIME_COL].values, df[SITE_COL].values.astype(int), {
        "num_features_time": X_time.shape[1],
        "num_features_static": S_num.shape[1],
        "n_sites": df["site_idx"].nunique(),
        "n_rivers": df["river"].nunique()
    }

def make_indices_per_site(df, L, H):
    """하나의 site 데이터에서 슬라이딩 윈도우 시작 인덱스 생성"""
    n = len(df)
    max_start = n - (L + H)
    if max_start <= 0:
        return np.array([], dtype=int)
    return np.arange(0, max_start)

# ---------------------- Dataset ----------------------
# 시계열 샘플 생성
class SeqDataset(Dataset):
    """site 단위 시계열 샘플 (lookback L → horizon H)"""
    def __init__(self,X_time,S_num,site_idx,river_idx,y,times,sites_original,idx,L,H):
        self.X_time,self.S_num,self.site_idx,self.river_idx,self.y=X_time,S_num,site_idx,river_idx,y
        self.times,self.sites_original,self.indices,self.L,self.H=times,sites_original,idx,L,H
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        s=self.indices[i]; e=s+self.L; t=e+self.H
        return (torch.from_numpy(self.X_time[s:e]),
                torch.from_numpy(self.S_num[e-1].astype(np.float32)),
                torch.tensor(self.site_idx[e-1],dtype=torch.long),
                torch.tensor(self.river_idx[e-1],dtype=torch.long),
                torch.from_numpy(self.y[e:t]),
                str(self.times[e-1]),
                int(self.sites_original[e-1]))

# ---------------------- Scalers ----------------------
# 표준화 (평균 0, 분산 1), 역변환
class Scalers:
    def __init__(self):
        self.time_scaler=StandardScaler()
        self.static_scaler=StandardScaler()
        self.y_scaler=StandardScaler()
    def fit(self,Xt,Xs,y):
        self.time_scaler.fit(Xt)
        self.static_scaler.fit(Xs)
        self.y_scaler.fit(y.reshape(-1,1))
    def transform_time(self,X): return self.time_scaler.transform(X).astype(np.float32)
    def transform_static(self,S): return self.static_scaler.transform(S).astype(np.float32)
    def transform_y(self,y): return self.y_scaler.transform(y.reshape(-1,1)).astype(np.float32).squeeze(-1)
    def inverse_y(self,y): return self.y_scaler.inverse_transform(y.reshape(-1,1)).squeeze(-1)

# ---------------------- Model ----------------------
# causal padding 제거
''' Conv1d의 양방향 패딩으로 미래 정보가 들어오는 것을 차단해 causal 컨볼루션 보장) '''
class Chomp1d(nn.Module):
    def __init__(self,c): super().__init__(); self.c=c
    def forward(self,x): return x[:,:,:-self.c].contiguous() if self.c>0 else x

# TCN의 기본 단위
''' dilated convolution 2층 + ReLU + Dropout + residual 연결 '''
class TemporalBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,dilation,dropout=0.2):
        super().__init__()
        pad=(kernel_size-1)*dilation
        self.conv1=nn.Conv1d(in_ch,out_ch,kernel_size,padding=pad,dilation=dilation)
        self.chomp1=Chomp1d(pad); self.relu1=nn.ReLU(); self.drop1=nn.Dropout(dropout)
        self.conv2=nn.Conv1d(out_ch,out_ch,kernel_size,padding=pad,dilation=dilation)
        self.chomp2=Chomp1d(pad); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(dropout)
        self.down=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else None; self.relu=nn.ReLU()
    def forward(self,x):
        out=self.conv1(x); out=self.chomp1(out); out=self.relu1(out); out=self.drop1(out)
        out=self.conv2(out); out=self.chomp2(out); out=self.relu2(out); out=self.drop2(out)
        res=x if self.down is None else self.down(x)
        return self.relu(out+res)

# 전체 TCN 모델 (TCN -> MLP -> Head)
class TCN(nn.Module):
    """site별 TCN 모델"""
    def __init__(self,num_features_time,num_features_static,n_sites,n_rivers,
                 channels=(64,64,128),kernel_size=3,dropout=0.2,horizon=24):
        super().__init__()
        emb_in=0
        self.river_emb=nn.Embedding(max(n_rivers,1),4) if n_rivers>0 else None
        if self.river_emb is not None: emb_in+=4
        ch_in=num_features_time+emb_in
        layers=[]
        for i,ch_out in enumerate(channels):
            layers.append(TemporalBlock(ch_in,ch_out,kernel_size,2**i,dropout))
            ch_in=ch_out
        self.tcn=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.static_mlp=nn.Sequential(nn.Linear(num_features_static,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(ch_in+32,ch_in),nn.ReLU(),nn.Linear(ch_in,horizon))
    def forward(self,x_time,s_static,site_id,river_id):
        x=x_time.transpose(1,2)     # (Batch, L, F_time)을 Conv1D 입력 형태 (Batch, F_time, L)로
        B,_,L=x.shape
        if self.river_emb is not None:
            emb_riv=self.river_emb(river_id).unsqueeze(-1).repeat(1,1,L)
            # site, river 임베딩을 피쳐 채널로 추가
            ''' 벡터들을 시계열에 복제하여 concat 
                시계열이 어떤 site/river의 데이터인지 인식하면서 시간 패턴을 학습하도록 '''
            x=torch.cat([x,emb_riv],dim=1)
        z=self.tcn(x)                       # 시계열 특징
        z=self.pool(z).squeeze(-1)          # (Batch, Channel)
        z_static=self.static_mlp(s_static)  # (Batch, 32) 정적 MLP와 합쳐
        return self.head(torch.cat([z,z_static],dim=1)) # head에서 horizon 길이만큼의 예측 출력

# ---------------------- 학습 / 평가 ----------------------
# 1 epoch 학습
''' MSELoss로 학습 '''
def train_one_epoch(model,loader,optim,loss_fn):
    model.train(); total=0
    for x_time,s_static,sid,ridx,y,_,_ in loader:
        x_time,s_static,sid,ridx,y=x_time.to(DEVICE),s_static.to(DEVICE),sid.to(DEVICE),ridx.to(DEVICE),y.to(DEVICE)
        pred=model(x_time,s_static,sid,ridx)
        loss=loss_fn(pred,y)
        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step(); total+=loss.item()*x_time.size(0)
    return total/len(loader.dataset)

# 검증
@torch.no_grad()
def evaluate(model,loader,loss_fn):     # validation MSE 반환
    model.eval(); total=0
    for x_time,s_static,sid,ridx,y,_,_ in loader:
        pred=model(x_time.to(DEVICE),s_static.to(DEVICE),sid.to(DEVICE),ridx.to(DEVICE))
        total+=loss_fn(pred,y.to(DEVICE)).item()*x_time.size(0)
    return total/len(loader.dataset)

# NSE/KGE 계산
@torch.no_grad()
def evaluate_full(model,loader,scalers,horizon):
    """예측 결과 (preds, trues, meta_df) 반환"""
    model.eval(); preds,trues,times,sites,horizons=[],[],[],[],[]
    for x_time,s_static,sid,ridx,y,base_time,site_o in loader:
        pred=model(x_time.to(DEVICE),s_static.to(DEVICE),sid.to(DEVICE),ridx.to(DEVICE)).cpu().numpy()
        y_s=y.numpy()
        for i in range(pred.shape[0]):
            preds.append(scalers.inverse_y(pred[i])); trues.append(scalers.inverse_y(y_s[i]))
            base=pd.to_datetime(base_time[i])
            for h in range(horizon):
                times.append(base+pd.Timedelta(hours=h+1)); sites.append(int(site_o[i])); horizons.append(h+1)
    return np.array(preds),np.array(trues),pd.DataFrame({"time":times,"site":sites,"horizon":horizons})

def nse(y_true,y_pred):
    denom=np.sum((y_true-np.mean(y_true))**2)
    return np.nan if denom==0 else 1-np.sum((y_true-y_pred)**2)/denom

def kge(y_true,y_pred):
    if len(y_true)<2: return np.nan
    r=np.corrcoef(y_true,y_pred)[0,1]
    alpha=np.std(y_pred,ddof=1)/(np.std(y_true,ddof=1)+1e-12)
    beta=(np.mean(y_pred)+1e-12)/(np.mean(y_true)+1e-12)
    return 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)
