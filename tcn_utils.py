from pathlib import Path

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------- 설정 ----------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_COL = "ymd"
SITE_COL = "code_new"
TARGET_COL = "elev"

# 시계열 피쳐
TIME_VARYING_FEATURES = [
    "wtemp", "ec", "temp", "rainfall", "wind",
    "humidity", "pressure", "gtemp", "welect"
]

# 지점 피쳐
STATIC_NUMERIC_FEATURES = ["lat", "lon", "level"]
STATIC_CATEGORICAL_FEATURES = ["river"]     # label-encoded → embedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------- Parsers ----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Input CSV path")
    ap.add_argument("--l_window", type=int, default=336, help="Lookback length (hours)")
    ap.add_argument("--h_horizon", type=int, default=24, help="Forecast horizon (hours)")
    ap.add_argument("--epochs", type=int, default=10, help="Epochs (default 10)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_calendar", action="store_true", help="Disable calendar features")
    ap.add_argument("--output_dir", type=str, default="outputs")
    return ap.parse_args()


# ---------------------- IO & Preprocess ----------------------
# 로드 (CSV 읽고 시간순 정렬)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values([SITE_COL, TIME_COL]).reset_index(drop=True)
    return df

# river 라벨 인코딩
def encode_river(df: pd.DataFrame) -> pd.DataFrame:
    if "river" in df.columns:
        le = LabelEncoder()
        df["river"] = le.fit_transform(df["river"].astype(str))
    return df

# 문자열 숫자 float 변환
def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# hour, day, month 사인/코사인 주기형 변수 추가
def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df[TIME_COL].dt.hour   # 0~23
    df["dow"] = df[TIME_COL].dt.dayofweek   # 0~6 (day of week)
    df["month"] = df[TIME_COL].dt.month    # 1~12
    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2*np.pi*df[col]/period)
        df[f"{col}_cos"] = np.cos(2*np.pi*df[col]/period)
    return df

# trian, test 분리
def split_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp("2018-12-31 23:59:59")
    val_end   = pd.Timestamp("2020-12-31 23:59:59")
    train_df = df[df[TIME_COL] <= train_end].copy()
    val_df   = df[(df[TIME_COL] > train_end) & (df[TIME_COL] <= val_end)].copy()
    test_df  = df[df[TIME_COL] > val_end].copy()
    return train_df, val_df, test_df

# code_new 범위 0~N-1 변환 (site 임베딩)
def remap_sites(train_df: pd.DataFrame, *dfs: pd.DataFrame):
    unique_sites = sorted(train_df[SITE_COL].astype(int).unique().tolist())
    mapping = {s:i for i,s in enumerate(unique_sites)}
    def _map(df):
        df = df.copy()
        df["site_idx"] = df[SITE_COL].astype(int).map(mapping).astype(int)
        return df
    remapped = [_map(x) for x in (train_df,)+dfs]
    return mapping, remapped

# 칼럼 존재성 검증
def build_feature_matrix(df: pd.DataFrame, use_calendar: bool):
    # Validate presence
    for c in TIME_VARYING_FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing time-varying feature: {c}")
    for c in STATIC_NUMERIC_FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing static numeric feature: {c}")
    if "river" not in df.columns:
        raise ValueError("Missing static categorical feature: river")

    # 시간 변동 입력
    X_time = df[TIME_VARYING_FEATURES].values.astype(np.float32)

    # 정적 숫자 입력
    S_num = df[STATIC_NUMERIC_FEATURES].values.astype(np.float32)

    # 카테고리 id
    river_idx = df["river"].astype(int).values

    # site id
    site_idx = df["site_idx"].astype(int).values

    # 캘린더 특성 사용 여부 옵션
    if use_calendar:
        cal_cols = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
        for c in cal_cols:
            if c not in df.columns:
                raise ValueError(f"Calendar feature missing: {c}")
        X_cal = df[cal_cols].values.astype(np.float32)
        X_time = np.hstack([X_time, X_cal])

    # 타깃 (지하수 수위)
    y = df[TARGET_COL].values.astype(np.float32)    
    
    # 타임스탬프 (각 행의 시간)
    times = df[TIME_COL].values     
    
    # 원래 code_new 값
    sites_original = df[SITE_COL].values.astype(int)

    # 메타 정보 딕셔너리
    meta = {
        "num_features_time": X_time.shape[1],       # TCN 입력 채널 수 (시간 피쳐 개수)
        "num_features_static": S_num.shape[1],      # 정적 MLP 입력 차원 수
        "n_sites": int(df["site_idx"].nunique()),   # site 임베딩 크기
        "n_rivers": int(df["river"].nunique())      # river 임베딩 크기
    }
    return X_time, S_num, site_idx, river_idx, y, times, sites_original, meta

# ---------------------- Feature Engineering ----------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """캘린더 기반 파생변수 추가"""
    df = df.copy()
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df["hour"] = df[TIME_COL].dt.hour
    df["dow"] = df[TIME_COL].dt.dayofweek
    df["month"] = df[TIME_COL].dt.month

    for col, period in {"hour": 24, "dow": 7, "month": 12}.items():
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    return df

# ---------------------- 데이터셋 클래스 ----------------------
class SeqDataset(Dataset):
    def __init__(self,
                 X_time: np.ndarray,
                 S_num: np.ndarray,
                 site_idx: np.ndarray,
                 river_idx: np.ndarray,
                 y: np.ndarray,
                 times: np.ndarray,
                 sites_original: np.ndarray,
                 idx_by_site: np.ndarray,
                 L: int,
                 H: int):
        self.X_time = X_time
        self.S_num = S_num
        self.site_idx = site_idx
        self.river_idx = river_idx
        self.y = y
        self.times = times
        self.sites_original = sites_original
        self.L = L
        self.H = H
        self.indices = idx_by_site

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.indices[i]
        e = s + self.L
        t = e + self.H
        x_time = self.X_time[s:e]                 # (L, F_time)
        # Use last step's identity/static values
        s_num  = self.S_num[e-1]                  # (F_static,)
        sid    = self.site_idx[e-1]
        ridx   = self.river_idx[e-1]
        y      = self.y[e:t]                      # (H,)
        base_time = self.times[e-1]
        site_orig = self.sites_original[e-1]
        return (torch.from_numpy(x_time),
                torch.from_numpy(s_num.astype(np.float32)),
                torch.tensor(sid, dtype=torch.long),
                torch.tensor(ridx, dtype=torch.long),
                torch.from_numpy(y),
                str(base_time),
                int(site_orig))

def make_indices_per_site(df: pd.DataFrame, L: int, H: int) -> np.ndarray:
    idx_list = []
    for _, g in df.groupby("site_idx", sort=False):
        n = len(g)
        max_start = n - (L + H)
        if max_start <= 0: continue
        start = g.index.min()
        idxs  = np.arange(start, start + max_start)
        idx_list.append(idxs)
    if not idx_list:
        return np.array([], dtype=int)
    return np.concatenate(idx_list)


# ---------------------- 스케일링 ----------------------
# train 데이터 기준으로 표준화 (평균 0, 분산 1)
class Scalers:
    def __init__(self):
        self.time_scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
    def fit(self, X_time_tr, X_static_tr, y_tr):
        self.time_scaler.fit(X_time_tr)
        self.static_scaler.fit(X_static_tr)
        self.y_scaler.fit(y_tr.reshape(-1,1))
    def transform_time(self, X):
        return self.time_scaler.transform(X).astype(np.float32)
    def transform_static(self, S):
        return self.static_scaler.transform(S).astype(np.float32)
    def transform_y(self, y):
        return self.y_scaler.transform(y.reshape(-1,1)).astype(np.float32).squeeze(-1)
    def inverse_y(self, y_scaled):
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1,1)).squeeze(-1)


# ---------------------- 모델 정의 ----------------------
''' padding 잘라내서 causal convolution 보장 '''
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

''' TCN 기본 블록 (CNN 2층 + residual) '''
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

''' TCN 클래스 '''
class TCN(nn.Module):
    def __init__(self,
                 num_features_time: int,
                 num_features_static: int,
                 n_sites: int,
                 n_rivers: int,
                 use_site_embedding: bool = True,
                 site_emb_dim: int = 8,
                 river_emb_dim: int = 4,
                 channels=(64,64,64,64),
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 horizon: int = 24):
        super().__init__()
        self.use_site_embedding = use_site_embedding
        self.horizon = horizon

        emb_in = 0
        if use_site_embedding:
            self.site_emb = nn.Embedding(n_sites, site_emb_dim)
            emb_in += site_emb_dim
        self.river_emb = nn.Embedding(max(n_rivers,1), river_emb_dim) if n_rivers > 0 else None
        if self.river_emb is not None:
            emb_in += river_emb_dim

        in_ch = num_features_time + emb_in
        layers = []
        ch_in = in_ch
        for i, ch_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size, dilation, dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # static numeric features → small MLP
        self.static_mlp = nn.Sequential(
            nn.Linear(num_features_static, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # fuse: TCN(C) + Static(32) → horizon head
        self.head = nn.Sequential(
            nn.Linear(ch_in + 32, ch_in),
            nn.ReLU(),
            nn.Linear(ch_in, horizon)
        )

    def forward(self, x_time: torch.Tensor, s_static: torch.Tensor, site_id: torch.Tensor, river_id: torch.Tensor):
        # x_time: (B, L, F_time)
        x = x_time.transpose(1, 2)  # (B, F_time, L)
        B, _, L = x.shape

        emb_list = []
        if self.use_site_embedding:
            emb_site = self.site_emb(site_id).unsqueeze(-1).repeat(1,1,L)
            emb_list.append(emb_site)
        if self.river_emb is not None:
            emb_riv = self.river_emb(river_id).unsqueeze(-1).repeat(1,1,L)
            emb_list.append(emb_riv)
        if emb_list:
            x = torch.cat([x] + emb_list, dim=1)

        z = self.tcn(x)                 # (B, C, L)
        z = self.pool(z).squeeze(-1)    # (B, C)

        z_static = self.static_mlp(s_static)  # (B, 32)
        z_all = torch.cat([z, z_static], dim=1)
        out = self.head(z_all)          # (B, H)
        return out


# ---------------------- 학습/계산 ----------------------
''' forward -> loss -> backprop -> optimizer.step() '''
def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    total = 0.0
    for x_time, s_static, sid, ridx, y, _, _ in loader:
        x_time = x_time.to(DEVICE)
        s_static = s_static.to(DEVICE)
        sid = sid.to(DEVICE)
        ridx = ridx.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x_time, s_static, sid, ridx)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * x_time.size(0)
    return total / len(loader.dataset)

'''검증셋 MSE 계산'''
@torch.no_grad()    # 평가에는 기울기 필요 x
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0.0
    for x_time, s_static, sid, ridx, y, _, _ in loader:
        x_time = x_time.to(DEVICE)
        s_static = s_static.to(DEVICE)
        sid = sid.to(DEVICE)
        ridx = ridx.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x_time, s_static, sid, ridx)
        loss = loss_fn(pred, y)
        total += loss.item() * x_time.size(0)
    return total / len(loader.dataset)

''' 평가 지표 '''
# NSE (Nash-Sutcliffe efficiency)
def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0: return np.nan
    return 1 - np.sum((y_true - y_pred)**2) / denom

# KGE (Kling-Gupta efficiency)
def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2: return np.nan
    r = np.corrcoef(y_true, y_pred)[0,1]
    alpha = np.std(y_pred, ddof=1) / (np.std(y_true, ddof=1) + 1e-12)
    beta = (np.mean(y_pred) + 1e-12) / (np.mean(y_true) + 1e-12)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

# 전체 예측 (inverse scaling -> preds/trues 변환)
@torch.no_grad()    # 예측에는 기울기 필요 x
def evaluate_full(model, loader, scalers, horizon: int):
    model.eval()
    preds, trues = [], []
    times, sites, horizons = [], [], []
    for x_time, s_static, sid, ridx, y, base_time_strs, site_origs in loader:
        x_time = x_time.to(DEVICE)
        s_static = s_static.to(DEVICE)
        sid = sid.to(DEVICE)
        ridx = ridx.to(DEVICE)

        pred_s = model(x_time, s_static, sid, ridx).cpu().numpy()  # (B,H)
        y_s    = y.numpy()

        for i in range(pred_s.shape[0]):
            p = scalers.inverse_y(pred_s[i])
            t = scalers.inverse_y(y_s[i])
            preds.append(p); trues.append(t)

            base_time = pd.to_datetime(base_time_strs[i])
            site_o = int(site_origs[i].item() if hasattr(site_origs[i], "item") else site_origs[i])
            for h in range(horizon):
                times.append(base_time + pd.Timedelta(hours=h+1))
                sites.append(site_o)
                horizons.append(h+1)

    preds = np.array(preds)
    trues = np.array(trues)
    meta_df = pd.DataFrame({"time": times, "site": sites, "horizon": horizons})
    return preds, trues, meta_df

# 결과 저장
def save_results(preds, trues, meta_df: pd.DataFrame, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    y_true = trues.flatten()
    y_pred = preds.flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    nse_val = float(nse(y_true, y_pred))
    kge_val = float(kge(y_true, y_pred))
    metrics = {"RMSE": rmse, "MAE": mae, "NSE": nse_val, "KGE": kge_val}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # long-form predictions
    rows = []
    N, H = preds.shape
    k = 0
    for i in range(N):
        for h in range(H):
            rows.append({
                "time": meta_df.iloc[k]["time"],
                "site": int(meta_df.iloc[k]["site"]),
                "horizon": int(meta_df.iloc[k]["horizon"]),
                "true": trues[i,h],
                "pred": preds[i,h],
            })
            k += 1
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"[저장 완료] results.json, predictions.csv in {out_dir}")
    print(metrics)

def run_fold(train_df, val_df, args, fold_id, out_dir, patience=5):
    """
    train_df, val_df: site별 dataframe
    args: argparse 인자 (epochs, l_window, h_horizon 등 포함)
    fold_id: site 코드
    out_dir: 결과 저장 경로
    patience: early stopping patience
    """
    from tcn_train_full import SeqDataset, Scalers, TCN, make_indices_per_site, build_feature_matrix, train_one_epoch, evaluate, evaluate_full  # 기존 정의 활용

    os.makedirs(out_dir, exist_ok=True)

    # feature matrix
    Xtr_t, Str_s, tr_site, tr_riv, ytr, ttr, sor_tr, meta_tr = build_feature_matrix(train_df, True)
    Xva_t, Sva_s, va_site, va_riv, yva, tva, sor_va, _       = build_feature_matrix(val_df,   True)

    scalers = Scalers()
    scalers.fit(Xtr_t, Str_s, ytr)
    Xtr_t = scalers.transform_time(Xtr_t); Str_s = scalers.transform_static(Str_s); ytr_s = scalers.transform_y(ytr)
    Xva_t = scalers.transform_time(Xva_t); Sva_s = scalers.transform_static(Sva_s); yva_s = scalers.transform_y(yva)

    # index
    L, H = args.l_window, args.h_horizon
    idx_tr = make_indices_per_site(train_df, L, H)
    idx_va = make_indices_per_site(val_df, L, H)

    # dataset
    tr_ds = SeqDataset(Xtr_t, Str_s, tr_site, tr_riv, ytr_s, ttr, sor_tr, idx_tr, L, H)
    va_ds = SeqDataset(Xva_t, Sva_s, va_site, va_riv, yva_s, tva, sor_va, idx_va, L, H)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    # model
    channels = list(map(int, args.channels.split(",")))
    model = TCN(
        num_features_time=meta_tr["num_features_time"],
        num_features_static=meta_tr["num_features_static"],
        n_sites=meta_tr["n_sites"],
        n_rivers=meta_tr["n_rivers"],
        channels=channels,
        dropout=args.dropout,
        horizon=H,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # early stopping
    best_val = float("inf")
    patience_cnt = 0
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_ld, optimizer, loss_fn)
        va_loss = evaluate(model, va_ld, loss_fn)
        print(f"[Fold {fold_id} | Epoch {epoch}] train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[Fold {fold_id}] Early stopping at epoch {epoch}")
                break

    # load best
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # evaluate
    preds, trues, meta_df = evaluate_full(model, va_ld, scalers, horizon=H)
    y_true = trues.flatten()
    y_pred = preds.flatten()

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "NSE": float(nse(y_true, y_pred)),
        "KGE": float(kge(y_true, y_pred)),
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

# ---------------------- Main ----------------------
def main():
    args = parse_args()

    # CSV 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 로드 및 전처리
    df = load_data(csv_path)
    df = encode_river(df)
    df = coerce_numeric(df, TIME_VARYING_FEATURES + STATIC_NUMERIC_FEATURES)

    use_calendar = not args.no_calendar
    if use_calendar:
        df = add_calendar_feats(df)

    # 데이터 분할
    train_df, val_df, test_df = split_by_time(df)

    # site remap
    site_map, (train_df, val_df, test_df) = remap_sites(train_df, val_df, test_df)
    n_sites = len(site_map)

    # 매트릭스 생성
    Xtr_t, Str_s, tr_site, tr_riv, ytr, ttr, sor_tr, meta_tr = build_feature_matrix(train_df, use_calendar)
    Xva_t, Sva_s, va_site, va_riv, yva, tva, sor_va, _       = build_feature_matrix(val_df,   use_calendar)
    Xte_t, Ste_s, te_site, te_riv, yte, tte, sor_te, _       = build_feature_matrix(test_df,  use_calendar)

    # 스케일링
    scalers = Scalers()
    scalers.fit(Xtr_t, Str_s, ytr)
    Xtr_t = scalers.transform_time(Xtr_t); Str_s = scalers.transform_static(Str_s); ytr_s = scalers.transform_y(ytr)
    Xva_t = scalers.transform_time(Xva_t); Sva_s = scalers.transform_static(Sva_s); yva_s = scalers.transform_y(yva)
    Xte_t = scalers.transform_time(Xte_t); Ste_s = scalers.transform_static(Ste_s); yte_s = scalers.transform_y(yte)

    # 인덱스 생성
    L, H = args.l_window, args.h_horizon
    def _make_indices(df_block): return make_indices_per_site(df_block, L, H)
    idx_tr = _make_indices(train_df)
    idx_va = _make_indices(val_df)
    idx_te = _make_indices(test_df)

    # 데이터셋 및 데이터로더
    tr_ds = SeqDataset(Xtr_t, Str_s, tr_site, tr_riv, ytr_s, ttr, sor_tr, idx_tr, L, H)
    va_ds = SeqDataset(Xva_t, Sva_s, va_site, va_riv, yva_s, tva, sor_va, idx_va, L, H)
    te_ds = SeqDataset(Xte_t, Ste_s, te_site, te_riv, yte_s, tte, sor_te, idx_te, L, H)

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # 모델 정의
    model = TCN(
        num_features_time=meta_tr["num_features_time"],
        num_features_static=meta_tr["num_features_static"],
        n_sites=meta_tr["n_sites"],
        n_rivers=meta_tr["n_rivers"],
        use_site_embedding=True,
        site_emb_dim=8,
        river_emb_dim=4,
        channels=(64,64,64,64),
        kernel_size=3,
        dropout=0.2,
        horizon=H
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 학습
    best_va = float("inf")
    best_path = os.path.join(out_dir, "tcn_best.pt")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr_ld, optimizer, loss_fn)
        va_loss = evaluate(model, va_ld, loss_fn)
        print(f"[{epoch:03d}] train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss < best_va:
            best_va = va_loss
            torch.save(model.state_dict(), best_path)

    # 테스트 및 저장
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds, trues, meta_df = evaluate_full(model, te_ld, scalers, horizon=H)
    save_results(preds, trues, meta_df, out_dir=out_dir)
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()

