# tcn_utils.py
import os, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------- 설정 ----------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_COL, SITE_COL, TARGET_COL = "ymd", "code_new", "elev"

# 피처 정의
TIME_VARYING_FEATURES = ["wtemp","ec","temp","rainfall","wind","humidity","pressure","gtemp"]
STATIC_NUMERIC_FEATURES = ["lat","lon","level"]
STATIC_CATEGORICAL_FEATURES = ["river"]

# ---------------------- 데이터 전처리 ----------------------
def load_data(path): 
    df=pd.read_csv(path); df[TIME_COL]=pd.to_datetime(df[TIME_COL],errors="coerce")
    return df.sort_values([SITE_COL,TIME_COL]).reset_index(drop=True)

def encode_river(df):
    if "river" in df.columns: df["river"]=LabelEncoder().fit_transform(df["river"].astype(str))
    return df

def coerce_numeric(df, cols): 
    for c in cols: 
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def add_calendar_feats(df):
    df["hour"]=df[TIME_COL].dt.hour; df["dow"]=df[TIME_COL].dt.dayofweek; df["month"]=df[TIME_COL].dt.month
    for col,period in [("hour",24),("dow",7),("month",12)]:
        df[f"{col}_sin"]=np.sin(2*np.pi*df[col]/period); df[f"{col}_cos"]=np.cos(2*np.pi*df[col]/period)
    return df

def build_feature_matrix(df,use_calendar=True):
    # 시간변화/정적 피처 확인
    for c in TIME_VARYING_FEATURES: 
        if c not in df.columns: raise ValueError(f"Missing {c}")
    for c in STATIC_NUMERIC_FEATURES:
        if c not in df.columns: raise ValueError(f"Missing {c}")
    if "river" not in df.columns: raise ValueError("Missing river")
    # 입력 매트릭스
    X_time=df[TIME_VARYING_FEATURES].values.astype(np.float32)
    S_num=df[STATIC_NUMERIC_FEATURES].values.astype(np.float32)
    river_idx=df["river"].astype(int).values; site_idx=df["site_idx"].astype(int).values
    if use_calendar: 
        cal=["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
        X_time=np.hstack([X_time,df[cal].values.astype(np.float32)])
    y=df[TARGET_COL].values.astype(np.float32)
    return X_time,S_num,site_idx,river_idx,y,df[TIME_COL].values,df[SITE_COL].values.astype(int),{
        "num_features_time":X_time.shape[1],
        "num_features_static":S_num.shape[1],
        "n_sites":df["site_idx"].nunique(),
        "n_rivers":df["river"].nunique()
    }

def add_features(df): return add_calendar_feats(df)

# ---------------------- Dataset ----------------------
class SeqDataset(Dataset):
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
                str(self.times[e-1]), int(self.sites_original[e-1]))

def make_indices_per_site(df,L,H):
    idx_list=[]; 
    for _,g in df.groupby("site_idx",sort=False):
        n=len(g); max_start=n-(L+H)
        if max_start<=0: continue
        idx_list.append(np.arange(g.index.min(),g.index.min()+max_start))
    return np.concatenate(idx_list) if idx_list else np.array([],dtype=int)

# ---------------------- 스케일러 ----------------------
class Scalers:
    def __init__(self): self.time_scaler=StandardScaler(); self.static_scaler=StandardScaler(); self.y_scaler=StandardScaler()
    def fit(self,Xt,Xs,y): self.time_scaler.fit(Xt); self.static_scaler.fit(Xs); self.y_scaler.fit(y.reshape(-1,1))
    def transform_time(self,X): return self.time_scaler.transform(X).astype(np.float32)
    def transform_static(self,S): return self.static_scaler.transform(S).astype(np.float32)
    def transform_y(self,y): return self.y_scaler.transform(y.reshape(-1,1)).astype(np.float32).squeeze(-1)
    def inverse_y(self,y): return self.y_scaler.inverse_transform(y.reshape(-1,1)).squeeze(-1)

# ---------------------- 모델 ----------------------
class Chomp1d(nn.Module):
    def __init__(self,c): super().__init__(); self.c=c
    def forward(self,x): return x[:,:,:-self.c].contiguous() if self.c>0 else x

class TemporalBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,dilation,dropout=0.2):
        super().__init__(); pad=(kernel_size-1)*dilation
        self.conv1=nn.Conv1d(in_ch,out_ch,kernel_size,padding=pad,dilation=dilation)
        self.chomp1=Chomp1d(pad); self.relu1=nn.ReLU(); self.drop1=nn.Dropout(dropout)
        self.conv2=nn.Conv1d(out_ch,out_ch,kernel_size,padding=pad,dilation=dilation)
        self.chomp2=Chomp1d(pad); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(dropout)
        self.down=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else None; self.relu=nn.ReLU()
    def forward(self,x):
        out=self.conv1(x); out=self.chomp1(out); out=self.relu1(out); out=self.drop1(out)
        out=self.conv2(out); out=self.chomp2(out); out=self.relu2(out); out=self.drop2(out)
        res=x if self.down is None else self.down(x); return self.relu(out+res)

class TCN(nn.Module):
    def __init__(self,num_features_time,num_features_static,n_sites,n_rivers,
                 site_emb_dim=8,river_emb_dim=4,channels=(64,64,128),kernel_size=3,dropout=0.2,horizon=24):
        super().__init__(); emb_in=site_emb_dim+river_emb_dim
        self.site_emb=nn.Embedding(n_sites,site_emb_dim); self.river_emb=nn.Embedding(max(n_rivers,1),river_emb_dim)
        ch_in=num_features_time+emb_in; layers=[]
        for i,ch_out in enumerate(channels):
            layers.append(TemporalBlock(ch_in,ch_out,kernel_size,2**i,dropout)); ch_in=ch_out
        self.tcn=nn.Sequential(*layers); self.pool=nn.AdaptiveAvgPool1d(1)
        self.static_mlp=nn.Sequential(nn.Linear(num_features_static,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(ch_in+32,ch_in),nn.ReLU(),nn.Linear(ch_in,horizon))
    def forward(self,x_time,s_static,site_id,river_id):
        x=x_time.transpose(1,2); B,_,L=x.shape
        x=torch.cat([x,
            self.site_emb(site_id).unsqueeze(-1).repeat(1,1,L),
            self.river_emb(river_id).unsqueeze(-1).repeat(1,1,L)],dim=1)
        z=self.tcn(x); z=self.pool(z).squeeze(-1); z_static=self.static_mlp(s_static)
        return self.head(torch.cat([z,z_static],dim=1))

# ---------------------- 학습/평가 ----------------------
def train_one_epoch(model,loader,optim,loss_fn):
    model.train(); total=0
    for x_time,s_static,sid,ridx,y,_,_ in loader:
        x_time,s_static,sid,ridx,y=x_time.to(DEVICE),s_static.to(DEVICE),sid.to(DEVICE),ridx.to(DEVICE),y.to(DEVICE)
        pred=model(x_time,s_static,sid,ridx); loss=loss_fn(pred,y)
        optim.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
        total+=loss.item()*x_time.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model,loader,loss_fn):
    model.eval(); total=0
    for x_time,s_static,sid,ridx,y,_,_ in loader:
        pred=model(x_time.to(DEVICE),s_static.to(DEVICE),sid.to(DEVICE),ridx.to(DEVICE)); 
        total+=loss_fn(pred,y.to(DEVICE)).item()*x_time.size(0)
    return total/len(loader.dataset)

def nse(y_true,y_pred): denom=np.sum((y_true-np.mean(y_true))**2); return np.nan if denom==0 else 1-np.sum((y_true-y_pred)**2)/denom
def kge(y_true,y_pred): 
    if len(y_true)<2: return np.nan
    r=np.corrcoef(y_true,y_pred)[0,1]; alpha=np.std(y_pred,ddof=1)/(np.std(y_true,ddof=1)+1e-12); beta=(np.mean(y_pred)+1e-12)/(np.mean(y_true)+1e-12)
    return 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)

@torch.no_grad()
def evaluate_full(model,loader,scalers,horizon):
    preds,trues,times,sites,horizons=[],[],[],[],[]
    for x_time,s_static,sid,ridx,y,base_time,site_o in loader:
        pred=model(x_time.to(DEVICE),s_static.to(DEVICE),sid.to(DEVICE),ridx.to(DEVICE)).cpu().numpy(); y_s=y.numpy()
        for i in range(pred.shape[0]):
            preds.append(scalers.inverse_y(pred[i])); trues.append(scalers.inverse_y(y_s[i]))
            base=pd.to_datetime(base_time[i])
            for h in range(horizon): times.append(base+pd.Timedelta(hours=h+1)); sites.append(int(site_o[i])); horizons.append(h+1)
    return np.array(preds),np.array(trues),pd.DataFrame({"time":times,"site":sites,"horizon":horizons})

def save_results(preds,trues,meta,out_dir="outputs",prefix="result"):
    os.makedirs(out_dir,exist_ok=True); y_true,y_pred=trues.flatten(),preds.flatten()
    metrics={"RMSE":float(np.sqrt(mean_squared_error(y_true,y_pred))),"MAE":float(mean_absolute_error(y_true,y_pred)),"NSE":float(nse(y_true,y_pred)),"KGE":float(kge(y_true,y_pred))}
    with open(os.path.join(out_dir,f"{prefix}_results.json"),"w",encoding="utf-8") as f: json.dump(metrics,f,indent=2,ensure_ascii=False)
    print(metrics); return metrics
