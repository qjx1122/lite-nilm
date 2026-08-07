# -*- coding: utf-8 -*-
"""
Build 5-user data/config profile and an auto-config blueprint.

This is a read-only profiler: it does not replace data/time_filters.json.
It separates main_model / guard / calibration / evaluation concerns and emits
artifacts/auto_config_blueprint.{json,md} for review.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "data" / "time_filters.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "auto_config_blueprint"
USER_LABELS = {
    "800080252842_4206894986488": "U842",
    "800080252844_4206894986488": "U2844",
    "800080270778_4200903422131": "U0778",
    "800080270789_4206680982373": "U0789",
    "800080270800_4200904302272": "U0800",
}


def _date_set(ranges: List[List[str]]) -> Set[str]:
    out: Set[str] = set()
    for lo, hi in ranges or []:
        for ts in pd.date_range(pd.Timestamp(lo), pd.Timestamp(hi), freq="D"):
            out.add(ts.strftime("%Y-%m-%d"))
    return out


def _load_branch_daily(uid: str, target_col: str) -> pd.DataFrame:
    from sys import path
    path.insert(0, str(PROJECT_ROOT / "scripts"))
    from feature_utils import load_branch_csv  # noqa: WPS433
    frames = []
    for root in [PROJECT_ROOT / "data" / "trains" / uid, PROJECT_ROOT / "data" / "infers" / uid]:
        for p in root.glob("*.csv"):
            if p.name.startswith("e241_"):
                continue
            df = load_branch_csv(p, target_col=target_col)
            if target_col not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
            frames.append(df[["time", "date", target_col]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    g = df.groupby("date").agg(
        n_branch=(target_col, "size"),
        kwh=(target_col, lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum()) * 0.25 / 1000.0),
        on_n=(target_col, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
        on_mean_w=(target_col, lambda s: float(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0].mean()) if (pd.to_numeric(s, errors="coerce") > 0).any() else 0.0),
    ).reset_index()
    return g


def _suggest_on_thr(daily: pd.DataFrame, configured_thr: float) -> Dict[str, object]:
    # Conservative heuristic: keep configured value, report data-derived scale for review.
    on_days = daily[daily["kwh"] > 0.01]
    med_on = float(on_days["on_mean_w"].median()) if len(on_days) else 0.0
    if med_on <= 150:
        category = "low_power"
    elif med_on <= 500:
        category = "mid_power"
    else:
        category = "high_power"
    return {"configured_on_thr_w": configured_thr, "median_on_mean_w": med_on, "power_category": category,
            "suggested_on_thr_w": configured_thr,
            "note": "threshold generation should use raw OFF noise p99 + margin; keep configured value until noise profiler is added"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    profile_rows = []
    blueprint = {"global_defaults": {"holdout_ood_start": "2026-07-01", "sample_period_min": 15}, "users": {}}
    for uid, label in USER_LABELS.items():
        u = cfg[uid]
        target = u.get("target_col", "p1")
        co = u.get("common_overrides", {})
        configured_thr = float(u.get("on_thr_w", co.get("on_thr_w", 10.0)))
        daily = _load_branch_daily(uid, target)
        train_dates = _date_set(u.get("train", {}).get("include", []))
        infer_dates = _date_set(u.get("infer", {}).get("include", [])) - _date_set(u.get("infer", {}).get("exclude", []))
        train_daily = daily[daily["date"].isin(train_dates)] if len(daily) else pd.DataFrame()
        infer_daily = daily[daily["date"].isin(infer_dates)] if len(daily) else pd.DataFrame()
        thr = _suggest_on_thr(train_daily, configured_thr) if len(train_daily) else _suggest_on_thr(daily, configured_thr)
        train_on = int((train_daily["kwh"] > 0.01).sum()) if len(train_daily) else 0
        train_off = int((train_daily["kwh"] <= 0.01).sum()) if len(train_daily) else 0
        infer_on = int((infer_daily["kwh"] > 0.01).sum()) if len(infer_daily) else 0
        infer_off = int((infer_daily["kwh"] <= 0.01).sum()) if len(infer_daily) else 0
        guard_need = []
        if train_off < 3:
            guard_need.append("insufficient_main_off_days")
        if label in ("U0789", "U0800", "U0778"):
            guard_need.append("force_off_guard_candidate")
        if label == "U842":
            guard_need.append("lowprob_rain_guard_candidate")
        if label == "U2844":
            guard_need.append("bus_guard_sliding_anchor_candidate")
        profile_rows.append({
            "user": label, "uid": uid, "target_col": target,
            "configured_on_thr_w": configured_thr,
            "train_days": len(train_dates), "train_on_days": train_on, "train_off_days": train_off,
            "infer_days": len(infer_dates), "infer_on_days": infer_on, "infer_off_days": infer_off,
            "bus_guard": bool(u.get("bus_guard", {}).get("enabled", False)),
            "power_temp_calib": bool(u.get("power_temp_calib", {}).get("enable", False)),
            "guard_need": ";".join(guard_need),
        })
        blueprint["users"][uid] = {
            "label": label,
            "target": {"target_col": target, **thr},
            "main_model": {
                "train_include": u.get("train", {}).get("include", []),
                "split_policy": "auto_stratified_by_on_off_power_weather",
                "data_sufficiency": {"train_on_days": train_on, "train_off_days": train_off},
            },
            "guard": {
                "enabled_candidates": guard_need,
                "train_include": [],
                "val_include": [],
                "note": "guard dates must be selected separately; do not merge guard samples into main_model train",
            },
            "calibration": {
                "calib_stats_include": u.get("calib_stats_include", []),
                "power_temp_calib": u.get("power_temp_calib", {}),
            },
            "evaluation": {
                "inference_include": u.get("infer", {}).get("include", []),
                "inference_exclude": u.get("infer", {}).get("exclude", []),
                "ood_holdout": [["2026-07-01", "2026-12-31"]],
            },
        }
    prof = pd.DataFrame(profile_rows)
    prof.to_csv(OUT_DIR / "user_profile.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "auto_config_blueprint.json").write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 5-user auto config blueprint\n", "## User profile\n", prof.to_markdown(index=False),
          "\n## Notes\n", "- This is a blueprint, not production config.\n- Main-model data, guard data, calibration data and evaluation holdout are separated.\n- Guard samples must not be merged into main-model training.\n"]
    (OUT_DIR / "auto_config_blueprint.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] {OUT_DIR / 'auto_config_blueprint.json'}")
    print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
