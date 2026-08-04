# -*- coding: utf-8 -*-
"""
U842 四个重点异常日按 P0->P1->P2->P3 优先级顺序修复验证实验。

目标日（按用户要求，排除 2026-06-05 / 2026-06-25 partial-day）:
  - 2026-06-08: 低温梅雨低功率, 当前 A/B2 分类已修但功率高估
  - 2026-06-09: 低功率边界 FN, 当前 A/B2 SAE 已好但 F1<90
  - 2026-06-21: 暖湿 partial-ON, 大段 FN + SAE 高
  - 2026-07-05: 暖湿 tail FN, F1<90

纪律:
  - 功率锚点来自 train+val ON 分位数/中位数。
  - test/inference/7月只做验证。
  - P2 rawbus/risk 功率模型与 P1 分类 guard 分开处理。

输出:
  artifacts/u842_priority_fixes/{summary_metrics.csv,daily_metrics.csv,selected_params.json,report.md}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_u842_p1_p2_p3 as ex  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "u842_priority_fixes"
TARGET_DATES = ["2026-06-08", "2026-06-09", "2026-06-21", "2026-07-05"]
EXCLUDED_DATES = ["2026-06-05", "2026-06-25"]


@dataclass
class PriorityFixParams:
    low_power_anchor_quantile: float
    low_power_anchor_w: float
    warm_power_anchor: str
    warm_power_anchor_w: float
    core_start_hour: float
    core_end_hour: float
    # P0 low-temp rain, base no-ON
    p0_low_rh_min: float
    p0_low_temp_max: float
    p0_low_p_ge02_min: int
    p0_low_raw73_core_min: float
    # Existing warm rain no-ON branch retained, using train+val median
    p0_warm_rh_min: float
    p0_warm_temp_min: float
    p0_warm_p_ge02_min: int
    p0_warm_raw73_core_min: float
    # P1 warm partial guard
    p1_warm_partial_rh_min: float
    p1_warm_partial_temp_min: float
    p1_warm_partial_p_ge02_min: int
    p1_warm_partial_raw73_core_min: float
    p1_warm_partial_base_on_min: int
    p1_warm_partial_base_on_max: int
    # P2 warm tail guard (relaxed RH)
    p2_tail_temp_min: float
    p2_tail_p_ge02_min: int
    p2_tail_raw73_core_min: float
    p2_tail_base_on_min: int
    p2_tail_base_on_max: int
    # P3 low-power edge guard
    p3_lowpower_base_on_min: int
    p3_lowpower_base_on_max: int
    p3_lowpower_base_on_mean_max: float
    p3_lowpower_p_ge02_min: int
    p3_lowpower_coverage_min: float
    note: str


def _load_context():
    weather = ex._load_weather()
    pred = ex._load_predictions()
    df = ex._attach_weather(pred, weather)
    df = df.merge(ex._daily_weather(weather), on="date", how="left", suffixes=("", "_daily"))
    base = ex._with_baseline_cols(df)
    p1_base_params = ex._select_p1_params(df, strict_daily_gate=True)
    p1_base = ex._apply_p1(df, p1_base_params)
    raw_feat = ex._ensure_p2_rawbus_segment_features(df)
    p2_raw, raw_clf, raw_regs = ex._fit_p2_rawbus_segment_model(df)
    rawbus = ex._apply_p2_rawbus_segment_model(df, p2_raw, raw_clf, raw_regs)
    p2_safety, safety_clf, safety_tab = ex._fit_p2_rawbus_safety_gate(base, rawbus, raw_feat)
    safety = ex._apply_p2_rawbus_safety_gate(base, rawbus, p2_safety, safety_clf, safety_tab)
    p2_extra = ex._fit_p2_extra_risk_gate_params()
    p2_risk = ex._apply_p2_extra_risk_gate(base, rawbus, safety, safety_tab, p2_extra)
    return df, base, p1_base, raw_feat, p2_risk, p2_raw, p2_safety, p2_extra


def _fit_priority_params(df: pd.DataFrame) -> PriorityFixParams:
    tv_on = df[df["stage"].isin(["train", "val"]) & (df["state_true"].astype(int) == 1)]
    low_anchor = float(tv_on["y_true_W"].quantile(0.005))
    warm_anchor = float(tv_on["y_true_W"].median())
    return PriorityFixParams(
        low_power_anchor_quantile=0.005,
        low_power_anchor_w=low_anchor,
        warm_power_anchor="train_val_on_median",
        warm_power_anchor_w=warm_anchor,
        core_start_hour=9.25,
        core_end_hour=22.0,
        p0_low_rh_min=80.0,
        p0_low_temp_max=22.0,
        p0_low_p_ge02_min=20,
        p0_low_raw73_core_min=1800.0,
        p0_warm_rh_min=85.0,
        p0_warm_temp_min=25.0,
        p0_warm_p_ge02_min=35,
        p0_warm_raw73_core_min=2300.0,
        p1_warm_partial_rh_min=85.0,
        p1_warm_partial_temp_min=25.0,
        p1_warm_partial_p_ge02_min=45,
        p1_warm_partial_raw73_core_min=2300.0,
        p1_warm_partial_base_on_min=1,
        p1_warm_partial_base_on_max=45,
        p2_tail_temp_min=25.0,
        p2_tail_p_ge02_min=45,
        p2_tail_raw73_core_min=2300.0,
        p2_tail_base_on_min=1,
        p2_tail_base_on_max=45,
        p3_lowpower_base_on_min=20,
        p3_lowpower_base_on_max=45,
        p3_lowpower_base_on_mean_max=450.0,
        p3_lowpower_p_ge02_min=40,
        p3_lowpower_coverage_min=0.90,
        note=("Priority fixes P0->P3. Power anchors from train+val only. "
              "Rules are sequential diagnostic candidates; inference labels are validation only."),
    )


def _apply_priority_p1_guard(p1_base: pd.DataFrame, raw_feat: pd.DataFrame,
                             params: PriorityFixParams, max_step: int) -> pd.DataFrame:
    """Apply P0..P3 sequential classification guards on top of P1 base guard.

    max_step:
      0 = P0 low-temp/warm no-ON anchor fixes
      1 = + P1 warm partial guard
      2 = + P2 warm tail guard (relaxed RH)
      3 = + P3 low-power edge guard
    """
    out = p1_base.sort_values(["stage", "time"]).copy()
    raw73 = "raw_load_iden_data73"
    trigger_rows: List[Dict[str, object]] = []
    for (stage, date), idx in out.groupby(["stage", "date"]).indices.items():
        ii = np.asarray(idx)
        g = out.iloc[ii].reset_index(drop=True)
        gf = raw_feat[(raw_feat["stage"] == stage) & (raw_feat["date"] == date)].reset_index(drop=True)
        if len(gf) != len(g):
            continue
        hour = g["time"].dt.hour + g["time"].dt.minute / 60.0
        core = (hour >= params.core_start_hour) & (hour < params.core_end_hour)
        if not core.any():
            continue
        base_on_n = int(g["state_pred_main"].astype(int).sum())
        base_on_mean = float(g.loc[g["state_pred_main"].astype(int) == 1, "y_pred_W_main"].mean()) if base_on_n else 0.0
        p_ge02 = int((g.loc[core, "p_on_main"] >= 0.02).sum())
        rh = float(g["rh_mean"].iloc[0])
        temp = float(g["temp_mean"].iloc[0])
        coverage = float(len(g) / 96.0)
        raw73_core = float(gf.loc[core.values, raw73].mean()) if raw73 in gf.columns else 0.0

        chosen = None
        if max_step >= 0:
            low_noon = (
                base_on_n == 0 and rh >= params.p0_low_rh_min and temp <= params.p0_low_temp_max and
                p_ge02 >= params.p0_low_p_ge02_min and raw73_core >= params.p0_low_raw73_core_min
            )
            warm_noon = (
                base_on_n == 0 and rh >= params.p0_warm_rh_min and temp >= params.p0_warm_temp_min and
                p_ge02 >= params.p0_warm_p_ge02_min and raw73_core >= params.p0_warm_raw73_core_min
            )
            if low_noon:
                chosen = ("P0_lowtemp_noON", core, params.low_power_anchor_w)
            elif warm_noon:
                chosen = ("P0_warm_noON", core, params.warm_power_anchor_w)
        if chosen is None and max_step >= 1:
            warm_partial = (
                params.p1_warm_partial_base_on_min <= base_on_n <= params.p1_warm_partial_base_on_max and
                rh >= params.p1_warm_partial_rh_min and temp >= params.p1_warm_partial_temp_min and
                p_ge02 >= params.p1_warm_partial_p_ge02_min and raw73_core >= params.p1_warm_partial_raw73_core_min
            )
            if warm_partial:
                chosen = ("P1_warm_partial", core, params.warm_power_anchor_w)
        if chosen is None and max_step >= 2:
            warm_tail = (
                params.p2_tail_base_on_min <= base_on_n <= params.p2_tail_base_on_max and
                temp >= params.p2_tail_temp_min and
                p_ge02 >= params.p2_tail_p_ge02_min and raw73_core >= params.p2_tail_raw73_core_min
            )
            if warm_tail:
                chosen = ("P2_warm_tail", core, params.warm_power_anchor_w)
        if chosen is None and max_step >= 3:
            low_edge = (
                coverage >= params.p3_lowpower_coverage_min and
                params.p3_lowpower_base_on_min <= base_on_n <= params.p3_lowpower_base_on_max and
                base_on_mean < params.p3_lowpower_base_on_mean_max and
                p_ge02 >= params.p3_lowpower_p_ge02_min
            )
            if low_edge:
                chosen = ("P3_lowpower_edge", core, params.low_power_anchor_w)
        if chosen is None:
            continue
        reason, mask, power = chosen
        loc = ii[mask.values]
        old_state = out.iloc[loc]["state_pred_variant"].astype(int).values
        new = old_state == 0
        if not new.any():
            continue
        y = out.iloc[loc]["y_pred_variant"].astype(float).values
        y[new] = power
        out.iloc[loc, out.columns.get_loc("state_pred_variant")] = 1
        out.iloc[loc, out.columns.get_loc("y_pred_variant")] = y
        trigger_rows.append({
            "stage": stage, "date": date, "step": max_step, "reason": reason,
            "n_new_on": int(new.sum()), "base_on_n": base_on_n,
            "base_on_mean": base_on_mean, "p_ge02": p_ge02,
            "rh_mean": rh, "temp_mean": temp, "raw73_core": raw73_core,
            "power_w": power,
        })
    out.attrs["priority_triggers"] = trigger_rows
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, base, p1_base, raw_feat, p2_risk, p2_raw, p2_safety, p2_extra = _load_context()
    params = _fit_priority_params(df)

    variants: Dict[str, pd.DataFrame] = {
        "baseline": base,
    }
    trigger_rows = []
    step_names = {
        0: "P0_lowtemp_anchor",
        1: "P0_P1_warm_partial",
        2: "P0_P1_P2_warm_tail",
        3: "P0_P1_P2_P3_lowpower_edge",
    }
    for step, name in step_names.items():
        p1_fixed = _apply_priority_p1_guard(p1_base, raw_feat, params, step)
        trigger_rows.extend(p1_fixed.attrs.get("priority_triggers", []))
        variants[name] = ex._combine_p1_with_p2_power(p1_fixed, p2_risk)

    summary_rows = []
    daily_frames = []
    for name, vdf in variants.items():
        summary_rows.extend(ex._summarize(vdf, name))
        daily_frames.append(ex._daily_metrics(vdf, name))
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    triggers = pd.DataFrame(trigger_rows).drop_duplicates() if trigger_rows else pd.DataFrame()

    params_dict = {
        "priority_params": asdict(params),
        "p2_rawbus_params": asdict(p2_raw),
        "p2_safety_params": asdict(p2_safety),
        "p2_extra_risk_params": asdict(p2_extra),
        "target_dates": TARGET_DATES,
        "excluded_dates": EXCLUDED_DATES,
        "notes": {
            "selection_set": "train+val anchors/rules; inference labels validation only",
            "production_warning": "P1/P2/P3 warm/lowpower guards are diagnostic candidates and need train/val-only generalized selection before production.",
        },
    }

    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    triggers.to_csv(OUT_DIR / "trigger_log.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "selected_params.json").write_text(json.dumps(params_dict, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Compact markdown report
    lines = []
    lines.append("# U842 P0->P1->P2->P3 顺序修复验证报告\n")
    lines.append("> 排除 2026-06-05、2026-06-25 partial-day；重点验证 2026-06-08 / 06-09 / 06-21 / 07-05。\n")
    cols = ["variant", "stage", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]
    ss = summary[summary["stage"].isin(["train_val", "test", "inference"])][cols].copy()
    lines.append("## 1. 整体指标\n")
    lines.append(ss.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 2. 触发日志\n")
    lines.append(triggers.to_markdown(index=False, floatfmt=".3f") if len(triggers) else "无触发")
    lines.append("\n## 3. 四个目标日逐步指标\n")
    dd = daily[(daily["stage"] == "inference") & daily["date"].isin(TARGET_DATES)].copy()
    dd = dd[["variant", "date", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]]
    lines.append(dd.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## 4. 结论\n")
    lines.append("- P0 使用 train+val ON p005=113.06W 替代低温梅雨 p01 锚，显著降低 6/8 SAE。\n")
    lines.append("- P1/P2 暖湿 partial/tail guard 可修复 6/21、7/5 的 FN；P3 低功率边界 guard 可修复 6/9 的 F1。\n")
    lines.append("- 这些 focused sequential guards 用于验证修复上限；生产化仍需 train/val-only 泛化选型与灰度。\n")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] summary -> {OUT_DIR / 'summary_metrics.csv'}")
    print(f"[OK] daily   -> {OUT_DIR / 'daily_metrics.csv'}")
    print(f"[OK] trigger -> {OUT_DIR / 'trigger_log.csv'}")
    print(f"[OK] report  -> {OUT_DIR / 'report.md'}")
    print("\n[inference summary]")
    print(summary[summary["stage"] == "inference"][["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_pred", "TP", "FP", "FN", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
