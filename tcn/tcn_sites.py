import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# tcn_utils 안에 있는 함수들 불러오기
from tcn_utils import (
    DEVICE,
    TIME_VARYING_FEATURES,
    STATIC_NUMERIC_FEATURES,
    encode_river,
    coerce_numeric,
    add_calendar_feats,
    build_feature_matrix,
    make_indices_per_site,
    SeqDataset,
    Scalers,
    TCN,
    train_one_epoch,
    evaluate,
    evaluate_full,
    nse,
    kge,
)

# ----------------------
# 예측 결과 시각화 함수
# ----------------------
def plot_site_timeseries(y_true, y_pred, times, site_id, out_dir):
    """한 사이트의 예측 vs 실제 곡선을 그려 PNG 저장"""
    plt.figure(figsize=(12, 4))
    plt.plot(times, y_true, label="True", alpha=0.7)
    plt.plot(times, y_pred, label="Pred", alpha=0.7)
    plt.title(f"Site {site_id} Prediction vs True (Time Series)")
    plt.xlabel("Time")
    plt.ylabel("Groundwater Level (elev)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/site{site_id}_timeseries.png")
    plt.close()


# ----------------------
# 1개 site 학습 + 평가
# ----------------------
def run_one_site(site_id, train_df, val_df, args):
    print(f"\n=== Training site {site_id} ===")

    # 데이터 분리
    ''' site 데이터만 추출
        해당 site만의 독립된 시계열 데이터 확보 '''
    tr_df = train_df[train_df["code_new"] == site_id].copy()
    va_df = val_df[val_df["code_new"] == site_id].copy()

    # 전처리
    tr_df = encode_river(tr_df)
    va_df = encode_river(va_df)
    tr_df = coerce_numeric(tr_df, TIME_VARYING_FEATURES + STATIC_NUMERIC_FEATURES)
    va_df = coerce_numeric(va_df, TIME_VARYING_FEATURES + STATIC_NUMERIC_FEATURES)

    # 캘린더 피쳐 추가
    if not args.no_calendar:
        tr_df = add_calendar_feats(tr_df)
        va_df = add_calendar_feats(va_df)

    # feature matrix 생성
    Xtr_t, Str_s, tr_site, tr_riv, ytr, ttr, sor_tr, meta_tr = build_feature_matrix(tr_df, not args.no_calendar)
    Xva_t, Sva_s, va_site, va_riv, yva, tva, sor_va, _       = build_feature_matrix(va_df, not args.no_calendar)

    # scaling
    ''' train 데이터 기준으로 평균, 표준편차 계산
        표준화 (mean = 0, std = 1)
        validation 데이터도 같은 스케일러로 변환'''
    scalers = Scalers()
    scalers.fit(Xtr_t, Str_s, ytr)
    Xtr_t = scalers.transform_time(Xtr_t)
    Str_s = scalers.transform_static(Str_s)
    ytr_s = scalers.transform_y(ytr)
    Xva_t = scalers.transform_time(Xva_t)
    Sva_s = scalers.transform_static(Sva_s)
    yva_s = scalers.transform_y(yva)

    # indices (윈도우 인덱스 생성)
    L, H = args.l_window, args.h_horizon
    idx_tr = make_indices_per_site(tr_df, L, H)
    idx_va = make_indices_per_site(va_df, L, H)

    # dataset & dataloader
    tr_ds = SeqDataset(Xtr_t, Str_s, tr_site, tr_riv, ytr_s, ttr, sor_tr, idx_tr, L, H)
    va_ds = SeqDataset(Xva_t, Sva_s, va_site, va_riv, yva_s, tva, sor_va, idx_va, L, H)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # model
    model = TCN(
        num_features_time=meta_tr["num_features_time"],
        num_features_static=meta_tr["num_features_static"],
        n_sites=meta_tr["n_sites"],
        n_rivers=meta_tr["n_rivers"],
        channels=tuple(map(int, args.channels.split(","))),
        dropout=args.dropout,
        horizon=H
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 학습 루프
    best_val = float("inf")
    best_path = os.path.join(args.output_dir, f"site{site_id}_best.pt")
    patience_counter = 0

    ''' 매 epoch마다 train/val MSE  출력
        validation loss가 개선될 때마다 모델 저장
        개선 안 되면 patience 증가 -> 일정 횟수 연속 악화 시 early stopping '''
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_ld, optimizer, loss_fn)
        va_loss = evaluate(model, va_ld, loss_fn)
        print(f"[Site {site_id} | Epoch {epoch}] train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Site {site_id}] Early stopping at epoch {epoch}")
                break

    # 예측 및 평가
    ''' 검증 데이터셋에 대해 예측 수행
        스케일 복원
        결과 시계열, 예측값/정답 배열 반환 '''
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds, trues, meta_df = evaluate_full(model, va_ld, scalers, horizon=H)

    # NSE, KGE 계산 (1-step 예측만)
    y_true = trues[:, 0]
    y_pred = preds[:, 0]
    nse_val = nse(y_true, y_pred)
    kge_val = kge(y_true, y_pred)

    # site별 시계열 그래프
    site_times = meta_df[meta_df["horizon"] == 1]["time"].values
    plot_site_timeseries(y_true, y_pred, site_times, site_id, args.output_dir)  # 예측 곡선 저장

    return {"site": site_id, "NSE": nse_val, "KGE": kge_val}


# ----------------------
# main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--l_window", type=int, default=180)
    ap.add_argument("--h_horizon", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--channels", type=str, default="64,64,128")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--output_dir", type=str, default="outputs_sites")
    ap.add_argument("--no_calendar", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load
    train_df = pd.read_csv(args.train_csv, parse_dates=["ymd"])
    val_df   = pd.read_csv(args.val_csv, parse_dates=["ymd"])

    results = []
    for site in sorted(train_df["code_new"].unique()):
        metrics = run_one_site(site, train_df, val_df, args)
        results.append(metrics)

    # 저장
    out_path = os.path.join(args.output_dir, "site_metrics.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n[저장 완료] site_metrics.csv in {args.output_dir}")


if __name__ == "__main__":
    main()
