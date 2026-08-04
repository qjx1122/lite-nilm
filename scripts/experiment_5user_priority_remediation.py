# -*- coding: utf-8 -*-
"""
5-user priority remediation validation experiment.

Sequential variants:
  B0 baseline_current
  P0 + oracle branch-consistency force-off for U0789/U0800 (remove FP where branch true OFF)
  P1 + U842 focused P0-P3 optimized variant from experiment_u842_priority_fixes.py
  P2 + oracle branch-off for U0778 OFF/false-positive points
  P3 + U2844 low-true high-FP diagnostic guard
  P4 + daily reporting flags (coverage/no-positive/off-FP); metrics same as P3

Important:
  P0/P2/P3 use branch labels as diagnostic upper bounds for consistency guards,
  not production-ready rules. Production versions require train/val-only signal selection.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "five_user_priority_remediation"
DT_HOURS = 0.25
USER_LABELS = {
    "800080252842_4206894986488": "U842",
    "800080252844_4206894986488": "U2844",
    "800080270778_4200903422131": "U0778",
    "800080270789_4206680982373": "U0789",
    "800080270800_4200904302272": "U0800",
}
P0_USERS = {"800080270789_4206680982373", "800080270800_4200904302272"}
P2_USERS = {"800080270778_4200903422131"}
P3_USERS = {"800080252844_4206894986488"}


def _load_inference_predictions() -> pd.DataFrame:
    frames = []
    for uid, label in USER_LABELS.items():
        p = PROJECT_ROOT / "artifacts" / "infers" / uid / "inference_result.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, parse_dates=["time"])
        df["user_id"] = uid
        df["user"] = label
        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        df["state_pred_variant"] = df["state_pred_main"].astype(int)
        df["y_pred_variant"] = df["y_pred_W_main"].astype(float)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_u842_priority_variant() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        "prio", PROJECT_ROOT / "scripts" / "experiment_u842_priority_fixes.py")
    prio = importlib.util.module_from_spec(spec)
    sys.modules["prio"] = prio
    spec.loader.exec_module(prio)  # type: ignore[union-attr]
    df, _base, p1_base, raw_feat, p2_risk, *_ = prio._load_context()
    params = prio._fit_priority_params(df)
    p1_fixed = prio._apply_priority_p1_guard(p1_base, raw_feat, params, 3)
    import experiment_u842_p1_p2_p3 as ex  # local script
    opt = ex._combine_p1_with_p2_power(p1_fixed, p2_risk)
    opt = opt[opt["stage"] == "inference"].copy()
    opt["user_id"] = "800080252842_4206894986488"
    opt["user"] = "U842"
    opt["date"] = opt["time"].dt.strftime("%Y-%m-%d")
    return opt[["time", "user_id", "state_pred_variant", "y_pred_variant"]]


def _classification_metrics(y_true: np.ndarray, state: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, state, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    return {
        "Accuracy": float(accuracy_score(y_true, state)),
        "Precision": float(precision_score(y_true, state, zero_division=0)),
        "Recall": float(recall_score(y_true, state, zero_division=0)),
        "F1": float(f1_score(y_true, state, zero_division=0)),
        "AUC": float(auc) if not pd.isna(auc) else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    true_kwh = float(y_true.sum() * DT_HOURS / 1000.0)
    pred_kwh = float(pred.sum() * DT_HOURS / 1000.0)
    return {
        "MAE_W": float(mean_absolute_error(y_true, pred)),
        "RMSE_W": float(np.sqrt(mean_squared_error(y_true, pred))),
        "SAE": abs(pred_kwh - true_kwh) / true_kwh if true_kwh > 0 else np.nan,
        "kWh_true": true_kwh,
        "kWh_pred": pred_kwh,
        "kWh_err": pred_kwh - true_kwh,
    }


def _summarize(df: pd.DataFrame, variant: str) -> List[Dict[str, object]]:
    rows = []
    for user, g in df.groupby("user"):
        y = g["state_true"].astype(int).values
        s = g["state_pred_variant"].astype(int).values
        p = g["p_on_main"].astype(float).values
        cls = _classification_metrics(y, s, p)
        reg = _regression_metrics(g["y_true_W"].astype(float).values,
                                  g["y_pred_variant"].astype(float).values)
        rows.append({"variant": variant, "user": user, "user_id": g["user_id"].iloc[0], "n_samples": int(len(g)), **cls, **reg})
    # pooled
    g = df
    y = g["state_true"].astype(int).values
    s = g["state_pred_variant"].astype(int).values
    p = g["p_on_main"].astype(float).values
    rows.append({"variant": variant, "user": "ALL", "user_id": "ALL", "n_samples": int(len(g)),
                 **_classification_metrics(y, s, p),
                 **_regression_metrics(g["y_true_W"].astype(float).values, g["y_pred_variant"].astype(float).values)})
    return rows


def _daily_metrics(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for (user, date), g in df.groupby(["user", "date"]):
        y = g["state_true"].astype(int).values
        s = g["state_pred_variant"].astype(int).values
        p = g["p_on_main"].astype(float).values
        rows.append({"variant": variant, "user": user, "user_id": g["user_id"].iloc[0], "date": date,
                     "n_samples": int(len(g)),
                     **_classification_metrics(y, s, p),
                     **_regression_metrics(g["y_true_W"].astype(float).values, g["y_pred_variant"].astype(float).values)})
    return pd.DataFrame(rows)


def _apply_p0(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m = out["user_id"].isin(P0_USERS) & (out["state_true"].astype(int) == 0)
    out.loc[m, "state_pred_variant"] = 0
    out.loc[m, "y_pred_variant"] = 0.0
    out["p0_branch_off_applied"] = m.astype(int)
    return out


def _apply_p1_u842(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    u842 = _load_u842_priority_variant().rename(columns={
        "state_pred_variant": "u842_state", "y_pred_variant": "u842_pred"})
    out = out.merge(u842, on=["user_id", "time"], how="left", sort=False)
    m = out["user_id"].eq("800080252842_4206894986488") & out["u842_state"].notna()
    out.loc[m, "state_pred_variant"] = out.loc[m, "u842_state"].astype(int)
    out.loc[m, "y_pred_variant"] = out.loc[m, "u842_pred"].astype(float)
    return out.drop(columns=["u842_state", "u842_pred"])


def _apply_p2_u0778(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m = out["user_id"].isin(P2_USERS) & (out["state_true"].astype(int) == 0)
    out.loc[m, "state_pred_variant"] = 0
    out.loc[m, "y_pred_variant"] = 0.0
    out["p2_u0778_off_applied"] = m.astype(int)
    return out


def _apply_p3_u2844(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Diagnostic upper bound: for U2844 days with very low true kWh and many FP,
    # remove only false-positive points, preserving TP points. This simulates a perfect
    # branch-consistency guard for low-true/high-FP days.
    daily = _daily_metrics(out[out["user_id"].isin(P3_USERS)], "tmp")
    targets = daily[(daily["kWh_true"] < 1.0) & (daily["FP"] >= 10)][["user_id", "date"]]
    if len(targets) == 0:
        out["p3_u2844_lowtrue_fp_guard"] = 0
        return out
    target_keys = set(map(tuple, targets.values.tolist()))
    m = out.apply(lambda r: (r["user_id"], r["date"]) in target_keys and int(r["state_true"]) == 0, axis=1)
    out.loc[m, "state_pred_variant"] = 0
    out.loc[m, "y_pred_variant"] = 0.0
    out["p3_u2844_lowtrue_fp_guard"] = m.astype(int)
    return out


def _daily_flags(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["coverage"] = out["n_samples"] / 96.0
    out["is_on_day"] = out["kWh_true"] > 0.01
    out["no_positive_day"] = (out["TP"] + out["FP"] + out["FN"] == 0)
    out["off_day_fp"] = (~out["is_on_day"]) & (out["FP"] > 0)
    out["f1_lt_90_on"] = out["is_on_day"] & (out["F1"] < 0.9)
    out["sae_gt_20_on"] = out["is_on_day"] & (out["SAE"] > 0.2)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_inference_predictions()
    variants: Dict[str, pd.DataFrame] = {}
    variants["B0_baseline_current"] = base
    variants["P0_branch_off_U0789_U0800"] = _apply_p0(variants["B0_baseline_current"])
    variants["P1_add_U842_focused"] = _apply_p1_u842(variants["P0_branch_off_U0789_U0800"])
    variants["P2_add_U0778_off_guard"] = _apply_p2_u0778(variants["P1_add_U842_focused"])
    variants["P3_add_U2844_lowtrue_fp_guard"] = _apply_p3_u2844(variants["P2_add_U0778_off_guard"])
    variants["P4_reporting_flags"] = variants["P3_add_U2844_lowtrue_fp_guard"].copy()

    summary_rows: List[Dict[str, object]] = []
    daily_frames = []
    for name, vdf in variants.items():
        summary_rows.extend(_summarize(vdf, name))
        daily_frames.append(_daily_flags(_daily_metrics(vdf, name)))
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True)

    # counts
    cnt = []
    for (variant, user), g in daily.groupby(["variant", "user"]):
        on = g["is_on_day"]
        cnt.append({
            "variant": variant, "user": user,
            "days": int(len(g)), "on_days": int(on.sum()),
            "f1_lt_90_on_days": int(g["f1_lt_90_on"].sum()),
            "sae_gt_20_on_days": int(g["sae_gt_20_on"].sum()),
            "off_day_fp_days": int(g["off_day_fp"].sum()),
            "partial_days": int((g["coverage"] < 0.9).sum()),
        })
    counts = pd.DataFrame(cnt)

    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(OUT_DIR / "daily_problem_counts.csv", index=False, encoding="utf-8-sig")

    params = {
        "variants": list(variants.keys()),
        "warning": "P0/P2/P3 branch-off guards use inference branch labels as diagnostic upper bounds, not production rules.",
        "priority_order": ["P0", "P1", "P2", "P3", "P4"],
    }
    (OUT_DIR / "selected_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# 5用户 P0→P4 优先级优化验证实验报告\n")
    lines.append("> P0/P2/P3 中的 branch-off/low-true guards 使用 inference branch 标签作为诊断上限，不是生产规则。生产化需要 bus/branch 一致性信号和 train/val-only 选型。\n")
    lines.append("## 1. Pooled / per-user summary\n")
    cols = ["variant", "user", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]
    lines.append(summary[summary["user"].isin(["ALL", "U842", "U2844", "U0778", "U0789", "U0800"])][cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 2. Daily problem counts\n")
    lines.append(counts.to_markdown(index=False))
    lines.append("\n## 3. Key conclusions\n")
    lines.append("- P0 oracle branch-off for U0789/U0800 removes FP upper-bound and quantifies the potential of circuit consistency guards.\n")
    lines.append("- P1 U842 focused guards show large U842 improvement but remain experimental/focused, not production pipeline.\n")
    lines.append("- P2 U0778 off guard shows OFF false-block upper-bound improvement; high-temp power bias remains separate.\n")
    lines.append("- P3 U2844 low-true high-FP diagnostic guard addresses low-true FP days, but FN days still need recall/bus_guard sliding anchors.\n")
    lines.append("- P4 adds report flags (coverage/no-positive/off-FP) for monitoring; no metric change.\n")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] summary -> {OUT_DIR/'summary_metrics.csv'}")
    print(f"[OK] daily -> {OUT_DIR/'daily_metrics.csv'}")
    print(f"[OK] report -> {OUT_DIR/'report.md'}")
    print("\n[ALL summary]")
    print(summary[summary["user"] == "ALL"][["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
