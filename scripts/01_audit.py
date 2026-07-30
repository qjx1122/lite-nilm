# -*- coding: utf-8 -*-
"""
Step 1: 数据勘察
- 加载总线 + 分路 CSV, 输出形状/时段/采样率/缺失率/统计画像
- 全程详细日志, 结果落盘 CSV
"""
import pandas as pd
import numpy as np
from common import (BUS_CSV, BR_CSV, ARTIFACT_DIR, SENT_VALUE, TARGET_COL,
                    get_logger, Timer)
from feature_utils import load_bus_csv, load_branch_csv

log = get_logger("audit")


def main():
    log.info("=" * 70)
    log.info("Step 1: 数据勘察 启动")
    log.info(f"  总线 CSV: {BUS_CSV}")
    log.info(f"  分路 CSV: {BR_CSV}")
    log.info("=" * 70)

    # ---------- 1. 加载 ----------
    with Timer("加载总线 CSV", log):
        bus, data_cols = load_bus_csv(BUS_CSV)
        log.info(f"  总线形状: {bus.shape}, 电参量列数: {len(data_cols)}")

    with Timer("加载分路 CSV", log):
        br = load_branch_csv(BR_CSV)
        log.info(f"  分路形状: {br.shape}")

    # ---------- 2. 总线统计 ----------
    log.info("-" * 70)
    log.info("【总线数据画像】")
    log.info(f"  时间范围: {bus['event_time'].min()}  ->  {bus['event_time'].max()}")
    log.info(f"  时长    : {bus['event_time'].max() - bus['event_time'].min()}")
    dt = bus["event_time"].diff().dropna()
    log.info(f"  采样间隔(s)  中位/均/最小/最大: "
             f"{dt.median().total_seconds():.0f} / "
             f"{dt.mean().total_seconds():.1f} / "
             f"{dt.min().total_seconds():.0f} / "
             f"{dt.max().total_seconds():.0f}")
    log.debug(f"  采样间隔 Top5:\n{dt.value_counts().head(5).to_string()}")

    # ---------- 3. 分路统计 ----------
    log.info("-" * 70)
    log.info("【分路数据画像】")
    log.info(f"  时间范围: {br['time'].min()}  ->  {br['time'].max()}")
    dt_br = br["time"].diff().dropna()
    log.info(f"  采样间隔(s)  中位: {dt_br.median().total_seconds():.0f}")
    if TARGET_COL not in br.columns:
        log.warning(f"  分路 CSV 缺少目标列 '{TARGET_COL}', "
                    f"实际列: {list(br.columns)}")
    else:
        log.info(f"  {TARGET_COL} 统计: min={br[TARGET_COL].min()}, "
                 f"max={br[TARGET_COL].max()}, "
                 f"mean={br[TARGET_COL].mean():.2f}, "
                 f">0比例={(br[TARGET_COL] > 0).mean()*100:.1f}%")

    # ---------- 4. 76 列电参量画像 ----------
    log.info("-" * 70)
    log.info("【76 列电参量统计画像】(逐列计算)")
    with Timer("电参量列扫描", log):
        rows = []
        for i, c in enumerate(data_cols, 1):
            s = bus[c]
            rows.append({
                "col": c,
                "min": float(s.min()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().any() else None,
                "median": float(s.median()) if s.notna().any() else None,
                "miss_pct": round(float(s.isna().mean() * 100), 2),
                "nunique": int(s.nunique(dropna=True)),
            })
            if i % 20 == 0:
                log.debug(f"  已扫描 {i}/{len(data_cols)} 列")
        summ = pd.DataFrame(rows)

    out_path = ARTIFACT_DIR / "bus_columns_summary.csv"
    summ.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  已保存电参量画像 -> {out_path}")

    # 标记将被剔除的列 (全空/零方差)
    drop_mask = (summ["miss_pct"] >= 100) | (summ["nunique"] <= 1)
    log.info(f"  全空或零方差列: {int(drop_mask.sum())} / {len(summ)} "
             f"(后续训练将剔除)")

    # ---------- 5. 重叠时段 ----------
    log.info("-" * 70)
    log.info("【时间对齐情况】")
    s = max(bus['event_time'].min(), br['time'].min())
    e = min(bus['event_time'].max(), br['time'].max())
    log.info(f"  重叠时段: {s}  ~  {e}  (时长 {e - s})")

    log.info("=" * 70)
    log.info("Step 1 完成")


if __name__ == "__main__":
    main()
