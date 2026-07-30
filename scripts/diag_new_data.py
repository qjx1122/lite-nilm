# -*- coding: utf-8 -*-
"""
深度诊断新数据 (2026-04-29 ~ 2026-05-17)
- 与训练集做完整画像对比
- 分析功率分布、季节属性、特征漂移
- 定位 SAE 偏高的根本原因
"""
import pandas as pd
import numpy as np
from common import DATA_DIR, ON_THR_W, ON_THR_BUSINESS_W, get_logger    # v6.13 解耦
from feature_utils import load_bus_csv, load_branch_csv, resample_and_align
from expert_utils import assign_season, SEASON_LABELS

log = get_logger("diag_new", log_to_file=False)

# 文件路径
NEW_BUS = DATA_DIR / "e241_800080252842_4206894986488-Ch1-20260429-0517.csv"
NEW_BR  = DATA_DIR / "4206894986488-260429-0517.csv"
OLD_BUS = DATA_DIR / "e241_800080252842_4206894986488-Ch1-20250710-260428-1.csv"
OLD_BR  = DATA_DIR / "4206894986488-250710-260428.csv"


def profile(label, bus_path, br_path):
    log.info("=" * 72)
    log.info(f"【{label}】")
    log.info("=" * 72)
    bus, data_cols = load_bus_csv(bus_path)
    br = load_branch_csv(br_path)
    log.info(f"  总线 shape={bus.shape}, 分路 shape={br.shape}")
    log.info(f"  总线时段: {bus['event_time'].min()}  ~  {bus['event_time'].max()}")
    log.info(f"  分路时段: {br['time'].min()}  ~  {br['time'].max()}")
    log.info(f"  总线时长: {bus['event_time'].max()-bus['event_time'].min()}")

    # 对齐
    df = resample_and_align(bus, br, keep_cols=data_cols)
    log.info(f"  对齐后(15min): {df.shape}")

    y = df["y_ac"].values
    state = (y >= ON_THR_BUSINESS_W).astype(int)    # v6.13: BUSINESS
    log.info(f"  ON 占比: {state.mean()*100:.1f}%")

    # 季节分布
    seasons = assign_season(df.index)
    log.info(f"  月份分布:")
    mc = pd.DatetimeIndex(df.index).to_period("M").value_counts().sort_index()
    for m, n in mc.items():
        log.info(f"    {m}: {n:>5}")
    log.info(f"  季节分布(按 expert_utils.SEASON_MAP):")
    for sea in SEASON_LABELS:
        m = seasons == sea
        n = int(m.sum())
        on_n = int((m & (state == 1)).sum())
        log.info(f"    {sea:<11}: 总={n:<5} ON={on_n:<5}")

    # 功率分布
    on_mask = state == 1
    if on_mask.any():
        y_on = y[on_mask]
        log.info(f"  ON 功率统计:")
        log.info(f"    均值={y_on.mean():.1f}W, 中位={np.median(y_on):.1f}W, "
                 f"std={y_on.std():.1f}W")
        log.info(f"    范围=[{y_on.min():.0f}, {y_on.max():.0f}]")
        log.info(f"    分位 P10/P25/P50/P75/P90/P95: " +
                 ", ".join([f"{np.percentile(y_on,p):.0f}W" for p in [10,25,50,75,90,95]]))
        # 功率分桶
        log.info(f"  ON 功率分桶:")
        bins = [0, 200, 400, 500, 600, 700, 800, 900, 1000]
        for i in range(len(bins)-1):
            mm = (y_on >= bins[i]) & (y_on < bins[i+1])
            n = int(mm.sum())
            if n > 0:
                log.info(f"    [{bins[i]:>4}, {bins[i+1]:>4})  n={n:<5} "
                         f"({n/len(y_on)*100:.1f}%)")
    return df, y, state


def main():
    log.info("\n" + "#" * 72)
    log.info("新数据 vs 训练数据 深度对比诊断")
    log.info("#" * 72)

    # 旧 (训练用)
    df_old, y_old, s_old = profile("旧数据 (训练集来源)", OLD_BUS, OLD_BR)

    # 新 (推理出问题)
    df_new, y_new, s_new = profile("新数据 (推理出 SAE=22.6%)", NEW_BUS, NEW_BR)

    # 关键对比
    log.info("\n" + "=" * 72)
    log.info("【两组数据对比 - 关键差异】")
    log.info("=" * 72)
    on_old = s_old == 1
    on_new = s_new == 1

    log.info(f"\n  指标               旧数据(train)       新数据(infer)")
    log.info(f"  ON 样本数          {on_old.sum():<18}{on_new.sum()}")
    if on_old.any() and on_new.any():
        log.info(f"  ON 功率 均值       {y_old[on_old].mean():<18.1f}{y_new[on_new].mean():.1f} W")
        log.info(f"  ON 功率 中位       {np.median(y_old[on_old]):<18.1f}{np.median(y_new[on_new]):.1f} W")
        log.info(f"  ON 功率 std        {y_old[on_old].std():<18.1f}{y_new[on_new].std():.1f} W")
        log.info(f"  ON 功率 max        {y_old[on_old].max():<18.0f}{y_new[on_new].max():.0f} W")
        log.info(f"  ON 功率 P95        {np.percentile(y_old[on_old],95):<18.0f}{np.percentile(y_new[on_new],95):.0f} W")

    # 季节对比
    log.info(f"\n  季节分布对比:")
    seasons_old = assign_season(df_old.index)
    seasons_new = assign_season(df_new.index)
    log.info(f"  {'季节':<12}{'旧 ON 数':<18}{'新 ON 数':<18}{'差异':<10}")
    for sea in SEASON_LABELS:
        n_old = int(((seasons_old == sea) & (s_old == 1)).sum())
        n_new = int(((seasons_new == sea) & (s_new == 1)).sum())
        log.info(f"  {sea:<12}{n_old:<18}{n_new:<18}")

    # 月份级 ON 功率对比 (关键: 是否同月份的真实功率也变了?)
    log.info(f"\n  按月份的 ON 功率均值对比 (检测同期数据是否漂移):")
    log.info(f"  {'月份':<12}{'旧数据 (n, 均值W)':<25}{'新数据 (n, 均值W)':<25}")
    months_old = pd.DatetimeIndex(df_old.index).to_period("M")
    months_new = pd.DatetimeIndex(df_new.index).to_period("M")
    all_months = sorted(set(list(months_old.unique()) + list(months_new.unique())))
    for m in all_months:
        old_mask = (months_old == m) & on_old
        new_mask = (months_new == m) & on_new
        old_str = f"({int(old_mask.sum())}, {y_old[old_mask].mean():.1f}W)" if old_mask.any() else "(-)"
        new_str = f"({int(new_mask.sum())}, {y_new[new_mask].mean():.1f}W)" if new_mask.any() else "(-)"
        log.info(f"  {str(m):<12}{old_str:<25}{new_str:<25}")

    # 5 月细分到旬, 看温度趋势是否反映在功率上
    log.info(f"\n  5 月 ON 功率按旬细分 (检测是否随气温升高功率攀升):")
    if (months_new == pd.Period("2026-05")).any():
        df_may = df_new[months_new == pd.Period("2026-05")].copy()
        df_may["day"] = df_may.index.day
        df_may["旬"] = pd.cut(df_may["day"], bins=[0, 10, 20, 31],
                            labels=["上旬(1-10)", "中旬(11-20)", "下旬(21-31)"])
        for xun in ["上旬(1-10)", "中旬(11-20)", "下旬(21-31)"]:
            sub = df_may[df_may["旬"] == xun]
            on_sub = sub["y_ac"] >= ON_THR_BUSINESS_W    # v6.13: BUSINESS
            if on_sub.any():
                y_sub = sub.loc[on_sub, "y_ac"]
                log.info(f"    {xun:<15} n={int(on_sub.sum()):<4} "
                         f"均值={y_sub.mean():.0f}W "
                         f"中位={y_sub.median():.0f}W "
                         f"max={y_sub.max():.0f}W")


if __name__ == "__main__":
    main()
