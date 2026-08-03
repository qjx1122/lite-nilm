# -*- coding: utf-8 -*-
"""
[v14.6] 5 用户汇总整体评估指标视图 (分类任务 + 回归任务 × 训练/推理)
====================================================================

数据源 (批跑产物, gitignored):
  artifacts/trains/<uid>/train_daily_metrics.csv   (dataset=train/val/test)
  artifacts/infers/<uid>/inference_daily_metrics.csv

聚合口径 (micro-pooled, 逐日列精确可加):
  - 分类: TP/FP/FN/TN 求和 -> Acc=(TP+TN)/N, P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN)
  - 回归: MAE = Σ(MAE_d·n_d)/N; RMSE = sqrt(Σ(RMSE_d²·n_d)/N);
          SAE(带符号) = (ΣkWh_pred − ΣkWh_true)/ΣkWh_true  (+高估 / −低估)
  - AUC 不可池化, 仅报日均值 (标注 ~)

范围:
  训练侧 train/val/test 及合并; 推理侧 6月扩段(<2026-07-01)/7月段(>=2026-07-01) 及合并
注意:
  - 各用户 ON 阈值不同 (10/10/50/60/50W), 跨用户池化分类指标为混合阈值口径
  - U2844 的 8 个"全零无信息日" (TP=FP=FN=0) 只贡献 TN
  - OFF 日真值≈0 的日子 SAE 日口径为 n/a; 池化用电量和不受其影响
产出: artifacts/overall_metrics_view.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
USERS = [
    ("U842", "800080252842_4206894986488"),
    ("U2844", "800080252844_4206894986488"),
    ("U0778", "800080270778_4200903422131"),
    ("U0789", "800080270789_4206680982373"),
    ("U0800", "800080270800_4200904302272"),
]
TRAIN_SCOPES = ["train", "val", "test"]
INFER_SCOPES = ["6月扩段", "7月段"]


def pooled(df: pd.DataFrame) -> dict:
    """对一批逐日行做 micro 池化"""
    n = int(df["n_samples"].sum())
    tp, fp, fn, tn = (int(df[c].sum()) for c in ("TP", "FP", "FN", "TN"))
    conf = tp + fp + fn + tn
    acc = (tp + tn) / conf if conf else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    mae = float((df["MAE_W"] * df["n_samples"]).sum() / n) if n else np.nan
    rmse = float(np.sqrt((df["RMSE_W"] ** 2 * df["n_samples"]).sum() / n)) if n else np.nan
    kt, kp = float(df["kWh_true"].sum()), float(df["kWh_pred"].sum())
    sae_signed = (kp - kt) / kt if kt > 1e-9 else np.nan
    auc_day = float(df["AUC"].replace("", np.nan). dropna().astype(float).mean()) \
        if "AUC" in df.columns else np.nan
    return dict(days=len(df), n=n, TP=tp, FP=fp, FN=fn, TN=tn,
                Acc=acc, P=prec, R=rec, F1=f1, AUC=auc_day,
                MAE=mae, RMSE=rmse, SAE=sae_signed, kWh_true=kt, kWh_pred=kp)


def f3(v):
    return f"{v:.3f}" if pd.notna(v) else "n/a"


def f0(v):
    return f"{v:.0f}" if pd.notna(v) else "-"


def f1f(v):
    return f"{v:.1f}" if pd.notna(v) else "-"


def f2(v):
    return f"{v:.2f}" if pd.notna(v) else "-"


def cls_row(name, m):
    return (f"| {name} | {m['days']} | {m['n']} | {m['TP']}/{m['FP']}/{m['FN']}/{m['TN']} "
            f"| {f3(m['Acc'])} | {f3(m['P'])} | {f3(m['R'])} | {f3(m['F1'])} | {f3(m['AUC'])}~ |")


def reg_row(name, m):
    sae = f"{m['SAE']*100:+.1f}%" if pd.notna(m["SAE"]) else "n/a"
    return (f"| {name} | {m['days']} | {f1f(m['MAE'])} | {f1f(m['RMSE'])} "
            f"| {sae} | {f2(m['kWh_true'])} → {f2(m['kWh_pred'])} |")


CLS_HEAD = ("| 范围 | 天数 | 样本N | TP/FP/FN/TN | Acc | Precision | Recall | F1 | AUC(日均) |\n"
            "|---|---|---|---|---|---|---|---|---|")
REG_HEAD = ("| 范围 | 天数 | MAE_W | RMSE_W | SAE(带符号) | kWh 真值→预测 |\n"
            "|---|---|---|---|---|---|")


def main():
    per_user = {}  # uname -> {scope: pooled}
    all_train_rows, all_jun_rows, all_jul_rows = [], [], []
    thr_map = {}

    for uname, uid in USERS:
        tr = pd.read_csv(ROOT / f"artifacts/trains/{uid}/train_daily_metrics.csv")
        inf = pd.read_csv(ROOT / f"artifacts/infers/{uid}/inference_daily_metrics.csv")
        tr["date"] = pd.to_datetime(tr["date"])
        inf["date"] = pd.to_datetime(inf["date"])
        thr_map[uname] = float(tr["on_thr_w"].iloc[0])

        m = {}
        for ds in TRAIN_SCOPES:
            m[ds] = pooled(tr[tr["dataset"] == ds])
        m["训练合并"] = pooled(tr)
        jun = inf[inf["date"] < "2026-07-01"]
        jul = inf[inf["date"] >= "2026-07-01"]
        m["6月扩段"] = pooled(jun)
        m["7月段"] = pooled(jul)
        m["推理合并"] = pooled(inf)
        per_user[uname] = m
        all_train_rows.append(tr.assign(user=uname))
        all_jun_rows.append(jun.assign(user=uname))
        all_jul_rows.append(jul.assign(user=uname))

    all_tr = pd.concat(all_train_rows)
    all_jun = pd.concat(all_jun_rows)
    all_jul = pd.concat(all_jul_rows)
    all_inf = pd.concat([all_jun, all_jul])
    tot = {"训练合并": pooled(all_tr), "6月扩段": pooled(all_jun),
           "7月段": pooled(all_jul), "推理合并": pooled(all_inf)}
    # 训练三分的跨用户池化也给出
    for ds in TRAIN_SCOPES:
        tot[ds] = pooled(all_tr[all_tr["dataset"] == ds])

    out = []
    out.append("# 5 用户汇总整体评估指标 (分类任务 × 回归任务 · 训练 × 推理)\n")
    out.append(f"> 生成: {pd.Timestamp.now():%Y-%m-%d %H:%M} · 口径: micro-pooled 逐日精确可加 · "
               "AUC 为逐日均值(~标注, 不可池化)")
    out.append("> 各用户 ON 阈值: " + ", ".join(f"{u}={thr_map[u]:.0f}W" for u, _ in USERS)
               + " (跨用户池化为混合阈值口径)")
    out.append("> SAE 带符号: + = 高估, − = 低估; 推理分段: 6月扩段 < 2026-07-01 ≤ 7月段\n")

    # ========== 汇总表 ==========
    out.append("\n## 一、5 用户汇总 (micro-pooled)\n")
    out.append("\n### 分类任务 (ON/OFF 检测)\n")
    out.append(CLS_HEAD)
    for sc in ["train", "val", "test", "训练合并", "6月扩段", "7月段", "推理合并"]:
        out.append(cls_row(sc, tot[sc]))
    out.append("\n### 回归任务 (功率估计)\n")
    out.append(REG_HEAD)
    for sc in ["train", "val", "test", "训练合并", "6月扩段", "7月段", "推理合并"]:
        out.append(reg_row(sc, tot[sc]))

    # ========== 每用户明细 ==========
    for uname, _ in USERS:
        m = per_user[uname]
        out.append(f"\n## 二、{uname} (on_thr={thr_map[uname]:.0f}W)\n")
        out.append("\n### 分类\n")
        out.append(CLS_HEAD)
        for sc in ["train", "val", "test", "训练合并", "6月扩段", "7月段", "推理合并"]:
            out.append(cls_row(sc, m[sc]))
        out.append("\n### 回归\n")
        out.append(REG_HEAD)
        for sc in ["train", "val", "test", "训练合并", "6月扩段", "7月段", "推理合并"]:
            out.append(reg_row(sc, m[sc]))

    dst = ROOT / "artifacts/overall_metrics_view.md"
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"[OK] 汇总视图 -> {dst}")

    # 控制台速览: 汇总段
    print("\n".join(out[6:24]))


if __name__ == "__main__":
    sys.exit(main())
