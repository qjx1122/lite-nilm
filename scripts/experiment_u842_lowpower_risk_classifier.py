# -*- coding: utf-8 -*-
"""
U842 low-power-long risk recognizer experiment.

Purpose:
  Train/validate a stronger day-level recognizer for low-power long ON risk using
  suggested features:
    - rawbus-to-baseline ratios
    - raw73 ON mean/std
    - morning/mid/evening raw73 and p_on distributions
    - base_on_mean/rawbus_on_mean interactions
    - closeness to train+val low-power-long clusters

Important:
  - Train labels use train+val only.
  - test/inference labels are validation only.
  - This experiment proves whether the recognizer is production-ready; it does not
    tune on OOD/July.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_u842_p1_p2_p3 as ex  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "u842_lowpower_risk_classifier"
LOWPOWER_MEAN_THR_W = 400.0
MIN_ON_HOURS = 10.0


@dataclass
class AnchorParams:
    humid_rh_min: float
    humid_anchor_w: float
    dry_anchor_w: float
    lowpower_label_true_on_h_min: float
    lowpower_label_true_on_mean_max_w: float
    note: str


def _load_priority_current():
    spec = importlib.util.spec_from_file_location(
        "prio", PROJECT_ROOT / "scripts/experiment_u842_priority_fixes.py")
    prio = importlib.util.module_from_spec(spec)
    sys.modules["prio"] = prio
    spec.loader.exec_module(prio)  # type: ignore[union-attr]
    df, base, p1_base, raw_feat, p2_risk, *_ = prio._load_context()
    params = prio._fit_priority_params(df)
    current = ex._combine_p1_with_p2_power(
        prio._apply_priority_p1_guard(p1_base, raw_feat, params, 3),
        p2_risk,
    )
    return df, base, raw_feat, current


def _fit_anchor_params(df: pd.DataFrame) -> AnchorParams:
    rows = []
    for (_stage, _date), g in df[df["stage"].isin(["train", "val"])].groupby(["stage", "date"]):
        on = g["state_true"].astype(int) == 1
        if on.sum() == 0:
            continue
        true_on_h = float(on.sum() * ex.DT_HOURS)
        true_on_mean = float(g.loc[on, "y_true_W"].mean())
        if true_on_h >= MIN_ON_HOURS and true_on_mean < LOWPOWER_MEAN_THR_W:
            rows.append({
                "rh_mean": float(g["rh_mean"].iloc[0]),
                "values": g.loc[on, "y_true_W"].astype(float).values,
            })
    if not rows:
        raise RuntimeError("No train+val low-power-long samples found")
    humid_values = np.concatenate([r["values"] for r in rows if r["rh_mean"] >= 80.0])
    dry_values = np.concatenate([r["values"] for r in rows if r["rh_mean"] < 80.0])
    humid_anchor = float(np.median(humid_values)) if len(humid_values) else float(np.median(np.concatenate([r["values"] for r in rows])))
    dry_anchor = float((np.quantile(dry_values, 0.60) + np.quantile(dry_values, 0.65)) / 2.0) if len(dry_values) else humid_anchor
    return AnchorParams(
        humid_rh_min=80.0,
        humid_anchor_w=humid_anchor,
        dry_anchor_w=dry_anchor,
        lowpower_label_true_on_h_min=MIN_ON_HOURS,
        lowpower_label_true_on_mean_max_w=LOWPOWER_MEAN_THR_W,
        note="anchors from train+val low-power-long ON samples only",
    )


def _build_daily_feature_table(df: pd.DataFrame, base: pd.DataFrame,
                               raw_feat: pd.DataFrame, current: pd.DataFrame,
                               anchors: AnchorParams) -> pd.DataFrame:
    rows = []
    raw73 = "raw_load_iden_data73"
    for (stage, date), g0 in current.groupby(["stage", "date"]):
        g = g0.reset_index(drop=True)
        b = base[(base["stage"] == stage) & (base["date"] == date)].reset_index(drop=True)
        gf = raw_feat[(raw_feat["stage"] == stage) & (raw_feat["date"] == date)].reset_index(drop=True)
        if len(g) != len(gf) or len(g) != len(b):
            continue
        true_on = g["state_true"].astype(int) == 1
        pred_on = g["state_pred_variant"].astype(int) == 1
        true_kwh = float(g["y_true_W"].sum() * ex.DT_HOURS / 1000.0)
        current_kwh = float(g["y_pred_variant"].sum() * ex.DT_HOURS / 1000.0)
        base_kwh = float(b["y_pred_variant"].sum() * ex.DT_HOURS / 1000.0)
        true_on_h = float(true_on.sum() * ex.DT_HOURS)
        true_on_mean = float(g.loc[true_on, "y_true_W"].mean()) if true_on.any() else 0.0
        rh = float(g["rh_mean"].iloc[0])
        anchor_w = anchors.humid_anchor_w if rh >= anchors.humid_rh_min else anchors.dry_anchor_w
        anchor_pred = g["y_pred_variant"].astype(float).values.copy()
        anchor_pred[pred_on.values] = anchor_w
        anchor_kwh = float(anchor_pred.sum() * ex.DT_HOURS / 1000.0)
        hour = g["time"].dt.hour + g["time"].dt.minute / 60.0
        row: Dict[str, object] = {
            "stage": stage,
            "date": date,
            "true_kwh": true_kwh,
            "base_kwh": base_kwh,
            "current_kwh": current_kwh,
            "anchor_kwh": anchor_kwh,
            "abs_current_err": abs(current_kwh - true_kwh),
            "abs_anchor_err": abs(anchor_kwh - true_kwh),
            "anchor_improves": int(abs(anchor_kwh - true_kwh) < abs(current_kwh - true_kwh)),
            "lowpower_long_label": int(true_on_h >= MIN_ON_HOURS and true_on_mean < LOWPOWER_MEAN_THR_W),
            "true_on_h": true_on_h,
            "true_on_mean_w": true_on_mean,
            "coverage": float(len(g) / 96.0),
            "n_samples": int(len(g)),
            "pred_on_n": int(pred_on.sum()),
            "current_on_mean": float(g.loc[pred_on, "y_pred_variant"].mean()) if pred_on.any() else 0.0,
            "base_on_mean": float(b.loc[pred_on, "y_pred_variant"].mean()) if pred_on.any() else 0.0,
            "p_on_mean": float(g["p_on_main"].mean()),
            "p_on_std": float(g["p_on_main"].std(ddof=0)),
            "p_on_q25": float(g["p_on_main"].quantile(0.25)),
            "p_on_q50": float(g["p_on_main"].quantile(0.50)),
            "p_on_q75": float(g["p_on_main"].quantile(0.75)),
            "p_on_ge02": int((g["p_on_main"] >= 0.02).sum()),
            "p_on_ge05": int((g["p_on_main"] >= 0.05).sum()),
            "p_on_ge10": int((g["p_on_main"] >= 0.10).sum()),
            "p_on_ge30": int((g["p_on_main"] >= 0.30).sum()),
            "p_on_ge45": int((g["p_on_main"] >= 0.45).sum()),
            "p_on_ge57": int((g["p_on_main"] >= 0.57).sum()),
            "rh_mean": rh,
            "temp_mean": float(g["temp_mean"].iloc[0]),
            "raw73_day_mean": float(gf[raw73].mean()) if raw73 in gf else 0.0,
            "raw73_day_std": float(gf[raw73].std(ddof=0)) if raw73 in gf else 0.0,
            "raw73_on_mean": float(gf.loc[pred_on.values, raw73].mean()) if pred_on.any() and raw73 in gf else 0.0,
            "raw73_on_std": float(gf.loc[pred_on.values, raw73].std(ddof=0)) if pred_on.any() and raw73 in gf else 0.0,
            "rawbus_to_current_ratio": anchor_kwh / current_kwh if current_kwh > 1e-9 else 0.0,
            "anchor_delta_kwh": anchor_kwh - current_kwh,
            "raw73_to_current_on": (float(gf.loc[pred_on.values, raw73].mean()) / (float(g.loc[pred_on, "y_pred_variant"].mean()) + 1e-6)) if pred_on.any() and raw73 in gf else 0.0,
            "dist_humid_low_cluster": 0.0,
            "dist_dry_low_cluster": 0.0,
        }
        # approximate distance to train+val low-power long clusters
        row["dist_humid_low_cluster"] = abs(float(row["current_on_mean"]) - anchors.humid_anchor_w) + abs(float(row["rh_mean"]) - 80.0) * 2 + abs(float(row["temp_mean"]) - 27.0) * 5
        row["dist_dry_low_cluster"] = abs(float(row["current_on_mean"]) - anchors.dry_anchor_w) + abs(float(row["rh_mean"]) - 59.0) * 2 + abs(float(row["temp_mean"]) - 26.5) * 5
        for name, h0, h1 in [("morn", 9, 12), ("mid", 12, 17), ("eve", 17, 22)]:
            m = (hour >= h0) & (hour < h1)
            row[f"p_on_{name}_mean"] = float(g.loc[m, "p_on_main"].mean()) if m.any() else 0.0
            row[f"raw73_{name}_mean"] = float(gf.loc[m.values, raw73].mean()) if m.any() and raw73 in gf else 0.0
            row[f"pred_{name}_mean"] = float(g.loc[m & pred_on, "y_pred_variant"].mean()) if (m & pred_on).any() else 0.0
            row[f"pred_{name}_n"] = int((m & pred_on).sum())
        rows.append(row)
    tab = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return tab


def _feature_cols(tab: pd.DataFrame) -> List[str]:
    banned = {"stage", "date", "true_kwh", "abs_current_err", "abs_anchor_err", "anchor_improves", "lowpower_long_label", "true_on_h", "true_on_mean_w"}
    return [c for c in tab.columns if c not in banned]


def _apply_anchor_by_days(current: pd.DataFrame, tab: pd.DataFrame, anchors: AnchorParams,
                          use_mask: np.ndarray, variant: str) -> pd.DataFrame:
    out = current.copy()
    selected = tab[use_mask].copy()
    triggers = []
    for _, r in selected.iterrows():
        mask = ((out["stage"] == r["stage"]) & (out["date"] == r["date"]) &
                (out["state_pred_variant"].astype(int) == 1))
        if not mask.any():
            continue
        anchor = anchors.humid_anchor_w if float(r["rh_mean"]) >= anchors.humid_rh_min else anchors.dry_anchor_w
        old_kwh = float(out.loc[(out["stage"] == r["stage"]) & (out["date"] == r["date"]), "y_pred_variant"].sum() * ex.DT_HOURS / 1000.0)
        out.loc[mask, "y_pred_variant"] = anchor
        new_kwh = float(out.loc[(out["stage"] == r["stage"]) & (out["date"] == r["date"]), "y_pred_variant"].sum() * ex.DT_HOURS / 1000.0)
        triggers.append({
            "variant": variant, "stage": r["stage"], "date": r["date"],
            "anchor_w": anchor, "old_kwh": old_kwh, "new_kwh": new_kwh,
            "lowpower_long_label": int(r["lowpower_long_label"]),
            "anchor_improves": int(r["anchor_improves"]),
        })
    out.attrs["triggers"] = triggers
    return out


def _variant_metrics(vdf: pd.DataFrame, name: str):
    return ex._summarize(vdf, name), ex._daily_metrics(vdf, name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, base, raw_feat, current = _load_priority_current()
    anchors = _fit_anchor_params(df)
    tab = _build_daily_feature_table(df, base, raw_feat, current, anchors)
    feat_cols = _feature_cols(tab)
    train = tab[tab["stage"].isin(["train", "val"]) & (tab["true_kwh"] > 0.01)].copy()

    models = {
        "rf_lowpower_thr_0p4": (RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=1, class_weight="balanced", random_state=1, n_jobs=1), 0.4),
        "extra_lowpower_thr_0p4": (ExtraTreesClassifier(n_estimators=200, max_depth=3, min_samples_leaf=1, class_weight="balanced", random_state=2, n_jobs=1), 0.4),
        "logreg_lowpower_thr_0p5": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)), 0.5),
    }

    variants: Dict[str, pd.DataFrame] = {"current_optimized": current}
    clf_rows = []
    trigger_rows = []
    for name, (model, thr) in models.items():
        model.fit(train[feat_cols], train["lowpower_long_label"].astype(int))
        proba = model.predict_proba(tab[feat_cols])[:, 1]
        use = proba >= thr
        vdf = _apply_anchor_by_days(current, tab, anchors, use, name)
        variants[name] = vdf
        for stage in ["train", "val", "test", "inference"]:
            sub = tab[(tab["stage"] == stage) & (tab["true_kwh"] > 0.01)]
            if len(sub) == 0:
                continue
            p = proba[sub.index]
            pred = p >= thr
            y = sub["lowpower_long_label"].astype(int).values
            clf_rows.append({
                "model": name, "stage": stage, "threshold": thr,
                "n_days": int(len(sub)), "true_positive_days": int(y.sum()),
                "pred_positive_days": int(pred.sum()),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "predicted_dates": ",".join(sub.loc[pred, "date"].astype(str).tolist()),
            })
        for row in vdf.attrs.get("triggers", []):
            trigger_rows.append(row)

    # Also include oracle upper-bound for 6/19+6/27 only.
    oracle_use = ((tab["stage"] == "inference") & (tab["date"].isin(["2026-06-19", "2026-06-27"]))).values
    oracle = _apply_anchor_by_days(current, tab, anchors, oracle_use, "oracle_target_0619_0627")
    variants["oracle_target_0619_0627"] = oracle
    trigger_rows.extend(oracle.attrs.get("triggers", []))

    summary_rows = []
    daily_frames = []
    for name, vdf in variants.items():
        srows, ddf = _variant_metrics(vdf, name)
        summary_rows.extend(srows)
        daily_frames.append(ddf)
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    clf_metrics = pd.DataFrame(clf_rows)
    triggers = pd.DataFrame(trigger_rows)

    tab.to_csv(OUT_DIR / "daily_feature_table.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    clf_metrics.to_csv(OUT_DIR / "classifier_metrics.csv", index=False, encoding="utf-8-sig")
    triggers.to_csv(OUT_DIR / "trigger_log.csv", index=False, encoding="utf-8-sig")
    params = {
        "anchors": asdict(anchors),
        "feature_cols": feat_cols,
        "train_label_counts": train["lowpower_long_label"].value_counts().sort_index().to_dict(),
        "models": {k: {"threshold": v[1], "class": type(v[0]).__name__} for k, v in models.items()},
        "note": "Models trained on train+val low_power_long labels; test/inference labels validation only. Oracle is upper-bound only.",
    }
    (OUT_DIR / "selected_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# U842 低功率长时风险识别器训练/验证报告\n")
    lines.append("## 1. 参数与标签\n")
    lines.append("```json\n" + json.dumps(params, ensure_ascii=False, indent=2, default=str) + "\n```\n")
    lines.append("\n## 2. 分类器日级识别效果\n")
    lines.append(clf_metrics.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 3. 应用锚点后的整体指标\n")
    cols = ["variant", "stage", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]
    lines.append(summary[summary["stage"].isin(["train_val", "test", "inference"])][cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 4. 6/19 与 6/27 日级效果\n")
    target = daily[(daily["stage"] == "inference") & daily["date"].isin(["2026-06-19", "2026-06-27"])]
    lines.append(target[["variant", "date", "F1", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 5. 触发日志\n")
    lines.append(triggers.to_markdown(index=False, floatfmt=".3f") if len(triggers) else "无触发")
    lines.append("\n## 6. 结论\n")
    lines.append("- train+val 低功率长时正样本只有 3 天，且不能代表 6/19 的 'raw bus 高 + baseline 高 + true≈200W' 强错配。\n")
    lines.append("- 所有训练出的识别器都无法可靠命中 6/19；能命中的多为 6/8、6/9 或全OFF高湿日，说明建议特征仍不足。\n")
    lines.append("- oracle 证明锚点可修 6/19/6/27，但生产识别器尚不合格；需要补充同型样本或引入更直接的业务/总线分解特征。\n")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] summary -> {OUT_DIR/'summary_metrics.csv'}")
    print(f"[OK] classifier -> {OUT_DIR/'classifier_metrics.csv'}")
    print(f"[OK] report -> {OUT_DIR/'report.md'}")
    print("\n[classifier metrics]")
    print(clf_metrics.to_string(index=False))
    print("\n[inference summary]")
    print(summary[summary["stage"] == "inference"][["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_pred", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
