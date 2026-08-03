# -*- coding: utf-8 -*-
"""
[v14.6] 5 用户『训练 + 推理』每日评估指标合并视图生成器
=========================================================
数据源 (均为批跑产物, gitignored):
  artifacts/trains/<uid>/train_daily_metrics.csv   (dataset 列: train/val/test)
  artifacts/infers/<uid>/inference_daily_metrics.csv (29 列, 含 n_bus_raw/n_branch_raw)
  artifacts/infers/<uid>/inference_result.csv      (逐点 y_true_W/y_pred, 备用)

产出:
  artifacts/daily_train_infer_metrics_view.md  (逐用户 数据质量卡 + 3 段逐日表)

口径:
  - n_bus_raw   : 总线当日 5min 原始样本数 (满日=288)
  - n_branch_raw: 分路当日 15min 原始样本数 (满日=96)
  - kWh 列      : 日累计电量 (15min 步长 -> sum*0.25/1000)
  - F1/SAE/MAE  : 见 metrics_utils 统一口径
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
USERS = {
    "U842": "800080252842_4206894986488",
    "U2844": "800080252844_4206894986488",
    "U0778": "800080270778_4200903422131",
    "U0789": "800080270789_4206680982373",
    "U0800": "800080270800_4200904302272",
}
TCOLS = ["date", "dataset", "n_samples", "F1", "Precision", "Recall",
         "MAE_W", "SAE", "kWh_true", "kWh_pred", "TP", "FP", "FN", "TN"]
ICOLS = ["date", "n_samples", "n_bus_raw", "n_branch_raw", "F1", "Precision", "Recall",
         "MAE_W", "SAE", "kWh_true", "kWh_pred", "TP", "FP", "FN",
         "is_on_day", "off_day_fp", "off_day_false_on_kWh", "on_only_mae_w"]


def fmt_pct(x):
    try:
        return f"{float(x)*100:.1f}" if abs(float(x)) <= 1.5 else f"{float(x):.1f}"
    except (TypeError, ValueError):
        return str(x)


def tbl(df: pd.DataFrame, cols) -> str:
    """精简 markdown 表"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%y-%m-%d")
    for c in df.columns:
        if c in ("F1", "Precision", "Recall"):
            df[c] = df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
        elif c in ("SAE",):
            # SAE 存储为比率 (0.308=30.8%; OFF 日记 None)
            df[c] = df[c].map(
                lambda v: "n/a" if pd.isna(v) or v == "" else f"{float(v)*100:.0f}%")
        elif c in ("MAE_W", "on_only_mae_w"):
            df[c] = df[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
        elif c in ("kWh_true", "kWh_pred", "off_day_false_on_kWh"):
            df[c] = df[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        elif c == "SAE_x":
            pass
    cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([header, sep] + rows)


def quality_card(uid: str, tr: pd.DataFrame, inf: pd.DataFrame, tf_cfg: dict) -> str:
    """数据质量速览"""
    lines = []
    tgt = (tf_cfg.get(uid, {}) or {}).get("target_col", "?")
    lines.append(f"- 目标分路列: `{tgt}`")

    # 训练侧
    for ds in ("train", "val", "test"):
        sub = tr[tr["dataset"] == ds]
        if len(sub):
            on_days = int((sub["kWh_true"] > 0).sum())
            lines.append(
                f"- 训练侧 {ds}: {len(sub)} 个有标签日 "
                f"(ON 日 {on_days} / OFF 日 {len(sub)-on_days})")
    # 推理侧
    if len(inf):
        inf = inf.copy()
        inf["date"] = pd.to_datetime(inf["date"])
        jun = inf[inf["date"] < "2026-07-01"]
        jul = inf[inf["date"] >= "2026-07-01"]
        for name, seg in (("6月扩段", jun), ("7月段", jul)):
            if not len(seg):
                lines.append(f"- 推理 {name}: 无有指标日")
                continue
            bus_full = int((seg["n_bus_raw"] >= 280).sum())
            br_full = int((seg["n_branch_raw"] >= 96).sum())
            zero_lab = int((seg["n_branch_raw"] == 0).sum())
            lines.append(
                f"- 推理 {name}: {len(seg)} 个有指标日; 总线满采集 {bus_full} 天, "
                f"分路满采集 {br_full} 天, 分路零样本 {zero_lab} 天")
        # 缺口日 (窗口内无指标)
        lines.append(
            f"- 推理日序范围: {inf['date'].min():%Y-%m-%d} ~ {inf['date'].max():%Y-%m-%d}")
    return "\n".join(lines)


def main():
    out = []
    out.append("# 5 用户逐日训练 + 推理评估指标合并视图 (v14.6)\n")
    out.append(f"> 生成: {pd.Timestamp.now():%Y-%m-%d %H:%M} · 批跑产物口径 · "
               "kWh 单位千瓦时 · SAE 单位 %\n")
    tf_cfg = json.load(open(ROOT / "data/time_filters.json", encoding="utf-8"))

    for uname, uid in USERS.items():
        tr_path = ROOT / f"artifacts/trains/{uid}/train_daily_metrics.csv"
        inf_path = ROOT / f"artifacts/infers/{uid}/inference_daily_metrics.csv"
        tr = pd.read_csv(tr_path) if tr_path.exists() else pd.DataFrame()
        inf = pd.read_csv(inf_path) if inf_path.exists() else pd.DataFrame()
        out.append(f"\n## {uname} ({uid})\n")
        out.append("**数据质量卡**\n")
        out.append(quality_card(uid, tr, inf, tf_cfg))

        if len(tr):
            out.append("\n**训练侧逐日 (train / val / test)**\n")
            tr_s = tr.sort_values(["dataset", "date"])
            for ds in ("train", "val", "test"):
                sub = tr_s[tr_s["dataset"] == ds]
                if len(sub):
                    out.append(f"\n*{ds} ({len(sub)} 天)*\n")
                    out.append(tbl(sub, TCOLS))
        if len(inf):
            inf = inf.copy()
            inf["date"] = pd.to_datetime(inf["date"])
            jun = inf[inf["date"] < "2026-07-01"].sort_values("date")
            jul = inf[inf["date"] >= "2026-07-01"].sort_values("date")
            out.append("\n**推理侧逐日**\n")
            if len(jun):
                out.append(f"\n*6月扩段 ({len(jun)} 天)*\n")
                out.append(tbl(jun, ICOLS))
            if len(jul):
                out.append(f"\n*7月段 ({len(jul)} 天)*\n")
                out.append(tbl(jul, ICOLS))
        out.append("\n---")

    dst = ROOT / "artifacts/daily_train_infer_metrics_view.md"
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"[OK] 视图 -> {dst}  ({len(out)} 段)")


if __name__ == "__main__":
    sys.exit(main())
