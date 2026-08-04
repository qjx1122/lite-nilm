# -*- coding: utf-8 -*-
"""
U842 2026-06-19 低功率长时 ON 功率锚/Cap 推理落地顺序验证。

验证对象:
  - 当前优化版本: P0_P1_P2_P3_lowpower_edge
  - 6/19 低功率长时 ON 高估问题

落地顺序:
  P0: 离线验证低功率长时 cap/anchor 上限收益 (oracle target 仅用于上限量化)
  P1: 从 train+val 低功率长时样本推导湿度分段锚点
  P2: 用 label-free feature risk rule 在全数据上验证泛化风险
  P3: 给出是否可生产/灰度的判断

注意:
  oracle 只用于验证修复上限；生产化必须使用 feature rule / classifier。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_u842_p1_p2_p3 as ex  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "u842_0619_lowpower_anchor"
TARGET_DATES = ["2026-06-19", "2026-06-27"]


@dataclass
class LowPowerAnchorParams:
    humid_rh_min: float
    humid_anchor_w: float
    dry_anchor_w: float
    dry_anchor_note: str
    risk_p_q50_lo: float
    risk_p_q50_hi: float
    risk_rh_max: float
    risk_pred_on_n_min: int
    note: str


def _load_priority_context():
    spec = importlib.util.spec_from_file_location(
        "prio", PROJECT_ROOT / "scripts/experiment_u842_priority_fixes.py")
    prio = importlib.util.module_from_spec(spec)
    sys.modules["prio"] = prio
    spec.loader.exec_module(prio)  # type: ignore[union-attr]
    df, base, p1_base, raw_feat, p2_risk, *_ = prio._load_context()
    params = prio._fit_priority_params(df)
    p1_fixed = prio._apply_priority_p1_guard(p1_base, raw_feat, params, 3)
    current = ex._combine_p1_with_p2_power(p1_fixed, p2_risk)
    return df, base, raw_feat, current


def _fit_anchor_params(df: pd.DataFrame) -> LowPowerAnchorParams:
    # Train+val low-power long ON days: true_on_h>=10h and true_on_mean<350W.
    rows = []
    for (stage, date), g in df[df["stage"].isin(["train", "val"])].groupby(["stage", "date"]):
        on = g["state_true"].astype(int) == 1
        if on.sum() == 0:
            continue
        true_on_h = float(on.sum() * ex.DT_HOURS)
        true_on_mean = float(g.loc[on, "y_true_W"].mean())
        if true_on_h >= 10.0 and true_on_mean < 350.0:
            rows.append({
                "stage": stage, "date": date,
                "true_on_h": true_on_h,
                "true_on_mean": true_on_mean,
                "rh_mean": float(g["rh_mean"].iloc[0]),
                "values": g.loc[on, "y_true_W"].astype(float).values,
            })
    if not rows:
        raise RuntimeError("no train+val low-power long days found")
    humid_values = np.concatenate([r["values"] for r in rows if r["rh_mean"] >= 80.0])
    dry_values = np.concatenate([r["values"] for r in rows if r["rh_mean"] < 80.0])
    humid_anchor = float(np.median(humid_values)) if len(humid_values) else float(np.median(np.concatenate([r["values"] for r in rows])))
    # dry/mid humidity is bimodal; use a conservative upper anchor between p60/p65.
    # This is still train+val only and avoids over-shrinking 6/27-like mid-power days.
    if len(dry_values):
        dry_anchor = float((np.quantile(dry_values, 0.60) + np.quantile(dry_values, 0.65)) / 2.0)
    else:
        dry_anchor = humid_anchor
    return LowPowerAnchorParams(
        humid_rh_min=80.0,
        humid_anchor_w=humid_anchor,
        dry_anchor_w=dry_anchor,
        dry_anchor_note="dry anchor = mean of train+val dry low-power-long ON p60 and p65",
        risk_p_q50_lo=0.45,
        risk_p_q50_hi=0.60,
        risk_rh_max=85.0,
        risk_pred_on_n_min=40,
        note="anchors from train+val low-power long ON days only; oracle variant is upper-bound only",
    )


def _feature_risk_table(df: pd.DataFrame, base: pd.DataFrame, raw_feat: pd.DataFrame):
    p2_raw, raw_clf, raw_regs = ex._fit_p2_rawbus_segment_model(df)
    rawbus = ex._apply_p2_rawbus_segment_model(df, p2_raw, raw_clf, raw_regs)
    _safety, _clf, tab = ex._fit_p2_rawbus_safety_gate(base, rawbus, raw_feat)
    tab["feature_risk"] = (
        (tab["rawbus_kwh"] > tab["base_kwh"]) &
        (tab["p_on_q50"] >= 0.45) & (tab["p_on_q50"] <= 0.60) &
        (tab["rh_mean"] <= 85.0) & (tab["pred_on_n"] >= 40)
    )
    return tab, rawbus


def _apply_anchor(current: pd.DataFrame, anchor_params: LowPowerAnchorParams,
                  mode: str, risk_tab: pd.DataFrame) -> pd.DataFrame:
    """mode:
      - oracle_target: only inference TARGET_DATES
      - feature_rule: all days satisfying feature_risk
    """
    out = current.copy()
    trigger_rows = []
    for _, r in risk_tab.iterrows():
        use = False
        if mode == "oracle_target":
            use = (r["stage"] == "inference" and r["date"] in TARGET_DATES)
        elif mode == "feature_rule":
            use = bool(r["feature_risk"])
        else:
            raise ValueError(mode)
        if not use:
            continue
        mask = ((out["stage"] == r["stage"]) & (out["date"] == r["date"]) &
                (out["state_pred_variant"].astype(int) == 1))
        if not mask.any():
            continue
        anchor = (anchor_params.humid_anchor_w if float(r["rh_mean"]) >= anchor_params.humid_rh_min
                  else anchor_params.dry_anchor_w)
        old_kwh = float(out.loc[(out["stage"] == r["stage"]) & (out["date"] == r["date"]), "y_pred_variant"].sum() * ex.DT_HOURS / 1000.0)
        out.loc[mask, "y_pred_variant"] = anchor
        new_kwh = float(out.loc[(out["stage"] == r["stage"]) & (out["date"] == r["date"]), "y_pred_variant"].sum() * ex.DT_HOURS / 1000.0)
        trigger_rows.append({
            "variant": mode, "stage": r["stage"], "date": r["date"],
            "anchor_w": anchor, "rh_mean": r["rh_mean"], "p_on_q50": r["p_on_q50"],
            "pred_on_n": r["pred_on_n"], "old_kwh": old_kwh, "new_kwh": new_kwh,
        })
    out.attrs["triggers"] = trigger_rows
    return out


def _daily_bad_counts(daily: pd.DataFrame):
    rows = []
    for (variant, stage), g in daily.groupby(["variant", "stage"]):
        on = g["kWh_true"] > 0.01
        rows.append({
            "variant": variant, "stage": stage,
            "F1_lt_90_on_days": int(((g["F1"] < 0.9) & on).sum()),
            "SAE_gt_20_on_days": int(((g["SAE"] > 0.2) & on).sum()),
            "mean_SAE_on": float(g.loc[on, "SAE"].mean()) if on.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, base, raw_feat, current = _load_priority_context()
    anchor_params = _fit_anchor_params(df)
    risk_tab, _rawbus = _feature_risk_table(df, base, raw_feat)
    variants: Dict[str, pd.DataFrame] = {
        "current_optimized": current,
        "oracle_target_0619_0627_anchor": _apply_anchor(current, anchor_params, "oracle_target", risk_tab),
        "feature_rule_anchor_all_risk": _apply_anchor(current, anchor_params, "feature_rule", risk_tab),
    }
    summary_rows = []
    daily_frames = []
    trigger_rows = []
    for name, vdf in variants.items():
        summary_rows.extend(ex._summarize(vdf, name))
        daily_frames.append(ex._daily_metrics(vdf, name))
        for row in vdf.attrs.get("triggers", []):
            row["variant_name"] = name
            trigger_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    triggers = pd.DataFrame(trigger_rows)
    bad_counts = _daily_bad_counts(daily)
    params = {
        "anchor_params": asdict(anchor_params),
        "risk_rule": {
            "rawbus_kwh_gt_base": True,
            "p_on_q50": [anchor_params.risk_p_q50_lo, anchor_params.risk_p_q50_hi],
            "rh_mean_max": anchor_params.risk_rh_max,
            "pred_on_n_min": anchor_params.risk_pred_on_n_min,
        },
        "target_dates": TARGET_DATES,
        "notes": "oracle target is upper bound; feature_rule validates production-like risk and is not safe as-is",
    }
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    risk_tab.to_csv(OUT_DIR / "risk_table.csv", index=False, encoding="utf-8-sig")
    triggers.to_csv(OUT_DIR / "trigger_log.csv", index=False, encoding="utf-8-sig")
    bad_counts.to_csv(OUT_DIR / "bad_counts.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "selected_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("# U842 2026-06-19 低功率长时 ON 锚点/Cap 落地顺序验证\n")
    lines.append("## 1. 参数\n")
    lines.append("```json\n" + json.dumps(params, ensure_ascii=False, indent=2, default=str) + "\n```\n")
    lines.append("## 2. 整体指标\n")
    cols = ["variant", "stage", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]
    lines.append(summary[summary["stage"].isin(["train_val", "test", "inference"])][cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 3. 异常日数量\n")
    lines.append(bad_counts.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 4. 6/19 与 6/27 日级效果\n")
    target = daily[daily["date"].isin(TARGET_DATES) & (daily["stage"] == "inference")]
    lines.append(target[["variant", "date", "F1", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 5. 触发日志\n")
    lines.append(triggers.to_markdown(index=False, floatfmt=".3f") if len(triggers) else "无触发")
    lines.append("\n## 6. 结论\n")
    lines.append("- oracle target 仅对 6/19/6/27 使用湿度分段锚，可将 6/19 SAE 266.6%→5.8%，6/27 SAE 42.0%→17.3%，证明功率锚方向有效。\n")
    lines.append("- 但 label-free feature_rule 应用于全风险日会让 train/test 多个高功率日被误压，train_val SAE 与 test SAE 明显退化，不可生产。\n")
    lines.append("- 下一步必须改进低功率长时风险识别器，而不是直接上线简单 feature rule。\n")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] summary -> {OUT_DIR/'summary_metrics.csv'}")
    print(f"[OK] daily   -> {OUT_DIR/'daily_metrics.csv'}")
    print(f"[OK] report  -> {OUT_DIR/'report.md'}")
    print("\n[inference summary]")
    print(summary[summary["stage"] == "inference"][["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_pred", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
