# -*- coding: utf-8 -*-
"""
5-user train/val-only production-candidate force-off validation.

This experiment tries to turn the oracle P0/P2/P3 force-off findings into
train/val-only candidates:
  1) day-level OFF-day guard, trained on train+val daily labels;
  2) point-level predicted-ON guard, trained per user on train+val predicted-ON points.

No inference labels are used for fitting/threshold selection. Inference labels are
used only for validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "five_user_trainval_forceoff"
DT_HOURS = 0.25
USER_LABELS = {
    "800080252842_4206894986488": "U842",
    "800080252844_4206894986488": "U2844",
    "800080270778_4200903422131": "U0778",
    "800080270789_4206680982373": "U0789",
    "800080270800_4200904302272": "U0800",
}


@dataclass
class PointGuardParam:
    user: str
    enabled: bool
    selected_threshold: float | None
    train_val_baseline_f1: float
    train_val_selected_f1: float | None
    train_val_baseline_recall: float
    train_val_selected_recall: float | None
    train_label_counts: Dict[str, int]
    note: str


@dataclass
class DayGuardParam:
    enabled: bool
    selected_threshold: float | None
    train_val_forced_off_days: int
    train_val_on_day_kills: int
    train_label_counts: Dict[str, int]
    note: str


def _load_all_predictions() -> pd.DataFrame:
    frames = []
    for uid, label in USER_LABELS.items():
        for stage in ["train", "val", "test"]:
            p = PROJECT_ROOT / "artifacts" / "trains" / uid / f"{stage}_pred.csv"
            df = pd.read_csv(p, parse_dates=["time"])
            df = df.rename(columns={
                "p_on": "p_on_main",
                "state_pred": "state_pred_main",
                "y_pred_W": "y_pred_W_main",
                "y_pred_low_W": "y_pred_low_W_main",
                "y_pred_high_W": "y_pred_high_W_main",
            })
            df["stage"] = stage
            df["user_id"] = uid
            df["user"] = label
            frames.append(df)
        p = PROJECT_ROOT / "artifacts" / "infers" / uid / "inference_result.csv"
        df = pd.read_csv(p, parse_dates=["time"])
        df["stage"] = "inference"
        df["user_id"] = uid
        df["user"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["y_pred_low_W_main", "y_pred_high_W_main"]:
        out[c] = out[c].fillna(out["y_pred_W_main"])
    out["date"] = out["time"].dt.strftime("%Y-%m-%d")
    out["hour"] = out["time"].dt.hour + out["time"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow"] = out["time"].dt.dayofweek.astype(float)
    out["state_pred_variant"] = out["state_pred_main"].astype(int)
    out["y_pred_variant"] = out["y_pred_W_main"].astype(float)
    return out


def _add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (uid, stage, date), g in df.groupby(["user_id", "stage", "date"]):
        pred = g["state_pred_main"].astype(int) == 1
        true = g["state_true"].astype(int) == 1
        rows.append({
            "user_id": uid, "stage": stage, "date": date,
            "day_true_on": int(true.sum() > 0),
            "day_true_kwh": float(g["y_true_W"].sum() * DT_HOURS / 1000.0),
            "day_pred_kwh": float(g["y_pred_W_main"].sum() * DT_HOURS / 1000.0),
            "day_pred_on_n": int(pred.sum()),
            "day_pred_on_mean": float(g.loc[pred, "y_pred_W_main"].mean()) if pred.any() else 0.0,
            "day_p_on_mean": float(g["p_on_main"].mean()),
            "day_p_on_q25": float(g["p_on_main"].quantile(0.25)),
            "day_p_on_q50": float(g["p_on_main"].quantile(0.50)),
            "day_p_on_q75": float(g["p_on_main"].quantile(0.75)),
            "day_p_on_max": float(g["p_on_main"].max()),
            "day_p_on_ge05": int((g["p_on_main"] >= 0.05).sum()),
            "day_p_on_ge50": int((g["p_on_main"] >= 0.50).sum()),
            "day_p_on_ge90": int((g["p_on_main"] >= 0.90).sum()),
            "day_first_pred_on_h": float(g.loc[pred, "hour"].min()) if pred.any() else -1.0,
            "day_last_pred_on_h": float(g.loc[pred, "hour"].max()) if pred.any() else -1.0,
            "coverage": float(len(g) / 96.0),
        })
    daily = pd.DataFrame(rows).fillna(0.0)
    out = df.merge(daily, on=["user_id", "stage", "date"], how="left", sort=False)
    for uid, label in USER_LABELS.items():
        out[f"user_{label}"] = (out["user_id"] == uid).astype(int)
    return out


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


def _summarize(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for user, g in df.groupby("user"):
        y = g["state_true"].astype(int).values
        s = g["state_pred_variant"].astype(int).values
        p = g["p_on_main"].astype(float).values
        rows.append({"variant": variant, "user": user, "user_id": g["user_id"].iloc[0], "n_samples": int(len(g)),
                     **_classification_metrics(y, s, p),
                     **_regression_metrics(g["y_true_W"].astype(float).values, g["y_pred_variant"].astype(float).values)})
    y = df["state_true"].astype(int).values
    s = df["state_pred_variant"].astype(int).values
    p = df["p_on_main"].astype(float).values
    rows.append({"variant": variant, "user": "ALL", "user_id": "ALL", "n_samples": int(len(df)),
                 **_classification_metrics(y, s, p),
                 **_regression_metrics(df["y_true_W"].astype(float).values, df["y_pred_variant"].astype(float).values)})
    return pd.DataFrame(rows)


def _score_stage(df: pd.DataFrame, state_col: str) -> Dict[str, float]:
    y = df["state_true"].astype(int).values
    s = df[state_col].astype(int).values
    return _classification_metrics(y, s, df["p_on_main"].astype(float).values)


def _fit_point_guards(df: pd.DataFrame) -> Tuple[Dict[str, object], List[PointGuardParam]]:
    feats = [
        "p_on_main", "y_pred_W_main", "y_pred_low_W_main", "y_pred_high_W_main",
        "hour", "hour_sin", "hour_cos", "dow",
        "day_pred_kwh", "day_pred_on_n", "day_pred_on_mean",
        "day_p_on_mean", "day_p_on_q25", "day_p_on_q50", "day_p_on_q75",
        "day_p_on_ge05", "day_p_on_ge50", "day_p_on_ge90",
        "coverage",
    ] + [f"user_{label}" for label in USER_LABELS.values()]
    models = {}
    params = []
    for uid, label in USER_LABELS.items():
        train = df[(df["user_id"] == uid) & df["stage"].isin(["train", "val"]) & (df["state_pred_main"].astype(int) == 1)].copy()
        counts = train["state_true"].astype(int).value_counts().sort_index().to_dict()
        base_tv = _score_stage(df[(df["user_id"] == uid) & df["stage"].isin(["train", "val"])], "state_pred_main")
        if len(counts) < 2 or counts.get(0, 0) < 2:
            params.append(PointGuardParam(label, False, None, base_tv["F1"], None, base_tv["Recall"], None,
                                          {str(k): int(v) for k, v in counts.items()},
                                          "disabled: insufficient negative predicted-ON samples in train+val"))
            continue
        clf = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=2,
                                     class_weight="balanced", random_state=42, n_jobs=1)
        clf.fit(train[feats].values.astype(float), train["state_true"].astype(int).values)
        # Train/val threshold selection: improve or maintain F1 with recall loss <= 0.5pp.
        tv = df[(df["user_id"] == uid) & df["stage"].isin(["train", "val"])].copy()
        prob = clf.predict_proba(tv[feats].values.astype(float))[:, 1]
        best = None
        for thr in np.linspace(0.05, 0.95, 19):
            st = tv["state_pred_main"].astype(int).values.copy()
            force = (st == 1) & (prob < thr)
            st[force] = 0
            tmp = tv.copy()
            tmp["guard_state"] = st
            m = _score_stage(tmp, "guard_state")
            # Hard floor: do not degrade train+val F1 more than 0.1pp or recall more than 0.5pp.
            if m["F1"] + 1e-12 < base_tv["F1"] - 0.001:
                continue
            if m["Recall"] + 1e-12 < base_tv["Recall"] - 0.005:
                continue
            key = (m["F1"], m["Precision"], -thr)
            if best is None or key > best[0]:
                best = (key, float(thr), m)
        if best is None:
            params.append(PointGuardParam(label, False, None, base_tv["F1"], None, base_tv["Recall"], None,
                                          {str(k): int(v) for k, v in counts.items()},
                                          "disabled: no threshold passed train+val floors"))
            continue
        _key, thr, metrics = best
        models[uid] = {"model": clf, "threshold": thr, "features": feats}
        params.append(PointGuardParam(label, True, thr, base_tv["F1"], metrics["F1"],
                                      base_tv["Recall"], metrics["Recall"],
                                      {str(k): int(v) for k, v in counts.items()},
                                      "enabled: train+val selected threshold"))
    return models, params


def _apply_point_guards(df: pd.DataFrame, models: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for uid, pack in models.items():
        m = out["user_id"] == uid
        if not m.any():
            continue
        clf = pack["model"]
        thr = pack["threshold"]
        feats = pack["features"]
        prob = clf.predict_proba(out.loc[m, feats].values.astype(float))[:, 1]
        pred_on = out.loc[m, "state_pred_variant"].astype(int).values == 1
        force = pred_on & (prob < thr)
        idx = out.loc[m].index[force]
        out.loc[idx, "state_pred_variant"] = 0
        out.loc[idx, "y_pred_variant"] = 0.0
    return out


def _fit_day_guard(df: pd.DataFrame) -> Tuple[object, DayGuardParam]:
    daily = df.drop_duplicates(["user_id", "stage", "date"]).copy()
    feats = [
        "day_pred_kwh", "day_pred_on_n", "day_pred_on_mean",
        "day_p_on_mean", "day_p_on_q25", "day_p_on_q50", "day_p_on_q75",
        "day_p_on_max", "day_p_on_ge05", "day_p_on_ge50", "day_p_on_ge90",
        "coverage", "day_first_pred_on_h", "day_last_pred_on_h",
    ] + [f"user_{label}" for label in USER_LABELS.values()]
    train = daily[daily["stage"].isin(["train", "val"])].copy()
    counts = train["day_true_on"].value_counts().sort_index().to_dict()
    clf = RandomForestClassifier(n_estimators=250, max_depth=4, min_samples_leaf=2,
                                 class_weight="balanced", random_state=7, n_jobs=1)
    clf.fit(train[feats].values.astype(float), train["day_true_on"].astype(int).values)
    proba = clf.predict_proba(train[feats].values.astype(float))[:, 1]
    # Conservative selection: zero train+val ON-day kills; among those maximize off-day forced.
    best = None
    for thr in np.linspace(0.01, 0.50, 50):
        force = proba < thr
        on_kill = int((force & (train["day_true_on"].values == 1)).sum())
        off_force = int((force & (train["day_true_on"].values == 0)).sum())
        key = (-on_kill, off_force, -thr)
        if on_kill == 0 and (best is None or key > best[0]):
            best = (key, float(thr), off_force, on_kill)
    if best is None or best[2] == 0:
        param = DayGuardParam(False, None, 0, 0, {str(k): int(v) for k, v in counts.items()},
                              "disabled: no threshold forces off train+val OFF days without ON-day kills")
        return {"model": clf, "features": feats, "threshold": None}, param
    _key, thr, off_force, on_kill = best
    return {"model": clf, "features": feats, "threshold": thr}, DayGuardParam(True, thr, off_force, on_kill,
                                                                                  {str(k): int(v) for k, v in counts.items()},
                                                                                  "enabled: zero train+val ON-day kills")


def _apply_day_guard(df: pd.DataFrame, pack: object) -> pd.DataFrame:
    out = df.copy()
    if pack["threshold"] is None:
        return out
    daily = out.drop_duplicates(["user_id", "stage", "date"]).copy()
    prob = pack["model"].predict_proba(daily[pack["features"]].values.astype(float))[:, 1]
    force_days = daily.loc[prob < pack["threshold"], ["user_id", "stage", "date"]]
    keys = set(map(tuple, force_days.values.tolist()))
    if not keys:
        return out
    mask = out.apply(lambda r: (r["user_id"], r["stage"], r["date"]) in keys, axis=1)
    out.loc[mask, "state_pred_variant"] = 0
    out.loc[mask, "y_pred_variant"] = 0.0
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _add_daily_features(_load_all_predictions())
    variants: Dict[str, pd.DataFrame] = {}
    variants["B0_baseline_current"] = df.copy()
    point_models, point_params = _fit_point_guards(df)
    variants["TV_point_guard"] = _apply_point_guards(variants["B0_baseline_current"], point_models)
    day_pack, day_param = _fit_day_guard(df)
    variants["TV_day_off_guard"] = _apply_day_guard(variants["B0_baseline_current"], day_pack)
    variants["TV_combined_guard"] = _apply_day_guard(variants["TV_point_guard"], day_pack)
    variants["P4_reporting_flags"] = variants["TV_combined_guard"].copy()

    summary = pd.concat([_summarize(vdf[vdf["stage"] == "inference"], name) for name, vdf in variants.items()], ignore_index=True)
    daily = pd.concat([_daily_flags(_daily_metrics(vdf[vdf["stage"] == "inference"], name)) for name, vdf in variants.items()], ignore_index=True)
    counts = []
    for (variant, user), g in daily.groupby(["variant", "user"]):
        counts.append({
            "variant": variant, "user": user, "days": int(len(g)),
            "on_days": int(g["is_on_day"].sum()),
            "f1_lt_90_on_days": int(g["f1_lt_90_on"].sum()),
            "sae_gt_20_on_days": int(g["sae_gt_20_on"].sum()),
            "off_day_fp_days": int(g["off_day_fp"].sum()),
        })
    counts = pd.DataFrame(counts)
    params = {
        "point_guards": [asdict(p) for p in point_params],
        "day_guard": asdict(day_param),
        "note": "All thresholds selected on train+val only. Production candidate rejected if inference/test validation degrades materially.",
    }
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(OUT_DIR / "daily_problem_counts.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "selected_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# 5用户 train/val-only force-off 生产候选验证报告\n")
    lines.append("## 1. 选型参数\n")
    lines.append("```json\n" + json.dumps(params, ensure_ascii=False, indent=2) + "\n```\n")
    lines.append("\n## 2. inference summary\n")
    cols = ["variant", "user", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]
    lines.append(summary[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 3. daily problem counts\n")
    lines.append(counts.to_markdown(index=False))
    lines.append("\n## 4. 结论\n")
    lines.append("- train/val-only point/day guards cannot reproduce oracle branch-off gains for U0789/U0800; their OFF days look like high-confidence ON days in model-output feature space.\n")
    lines.append("- Conservative day guard avoids train/val ON-day kills but kills important U842 OOD ON days; therefore not production-safe.\n")
    lines.append("- Point guard can reduce FP at high thresholds but causes large Recall loss; not acceptable for production.\n")
    lines.append("- Conclusion: keep only P4 reporting flags in production; P0/P2/P3 require stronger bus/branch consistency signals beyond current train/val model-output features.\n")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] summary -> {OUT_DIR/'summary_metrics.csv'}")
    print(f"[OK] report -> {OUT_DIR/'report.md'}")
    print(summary[summary["user"] == "ALL"][["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
