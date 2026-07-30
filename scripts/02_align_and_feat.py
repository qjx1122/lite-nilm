# -*- coding: utf-8 -*-
"""
Step 2: 时间对齐 + 特征相关性分析

v6.12.3: 新增 --exclude-dates 参数
  用法: python 02_align_and_feat.py --exclude-dates 2025-06-30,2025-07-01,2025-07-02,2025-07-03
  用途: 训练数据清洗, 排除工作模式不一致的污染日期
"""
import argparse
import pandas as pd
import numpy as np
from common import (BUS_CSV, BR_CSV, ARTIFACT_DIR, RESAMPLE, TARGET_COL,
                    get_logger, Timer)
from feature_utils import load_bus_csv, load_branch_csv, resample_and_align

log = get_logger("align")


def main():
    # v6.12.3: 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dates", default="",
                        help="逗号分隔的训练数据排除日期 (YYYY-MM-DD), 用于剔除工作模式不一致的污染段")
    # [v12] 新增: 通用时段过滤规格 (JSON 字符串, include+exclude)
    parser.add_argument("--time-filter-spec", default="",
                        help="[v12] JSON 字符串, 时段过滤规格 "
                             "{'include':[[start,end],...], 'exclude':[[start,end],...]}. "
                             "支持任意时段 (非整天), 闭区间. 与 --exclude-dates 可同时使用 "
                             "(exclude-dates 视为整天 exclude, 附加到 spec 的 exclude 上)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Step 2: 时间对齐 + 特征相关性")
    log.info("=" * 70)

    with Timer("加载数据", log):
        bus, data_cols = load_bus_csv(BUS_CSV)
        # [v13.16] 传 TARGET_COL 让 load_branch_csv 按需物化复合列 (如 "p1+p2")
        br = load_branch_csv(BR_CSV, target_col=TARGET_COL)
        log.info(f"  总线 {bus.shape}, 分路 {br.shape}")
        if TARGET_COL and "+" in TARGET_COL:
            log.info(f"  [v13.16] 检测到复合 target_col={TARGET_COL!r}, "
                     f"已物化为 br['{TARGET_COL}'] = "
                     f"{' + '.join(TARGET_COL.split('+'))} 逐行求和")

        # v6.12.3: 数据清洗 - 排除指定日期 (向后兼容, 内部转成 time_filter spec)
        legacy_exclude_ranges = []
        if args.exclude_dates.strip():
            excl_dates = [pd.Timestamp(d.strip()).date()
                          for d in args.exclude_dates.split(",") if d.strip()]
            log.info(f"  [v6.12.3 数据清洗] 排除日期 (legacy): {excl_dates}")
            for d in excl_dates:
                legacy_exclude_ranges.append([
                    d.strftime("%Y-%m-%d"),
                    d.strftime("%Y-%m-%d"),
                ])

        # [v12] 新时段过滤 (统一路径)
        try:
            from time_filter_utils import (
                cli_arg_to_spec, apply_time_filter, parse_ranges, spec_summary,
            )
            spec = cli_arg_to_spec(args.time_filter_spec)
            # 把 legacy --exclude-dates 合并到 spec 的 exclude
            if legacy_exclude_ranges:
                if spec is None:
                    spec = {"include": [], "exclude": parse_ranges(legacy_exclude_ranges)}
                else:
                    spec = {
                        "include": spec["include"],
                        "exclude": spec["exclude"] + parse_ranges(legacy_exclude_ranges),
                    }
            if spec is not None:
                log.info(f"  [v12 时段过滤] 规格: {spec_summary(spec)}")
                bus = apply_time_filter(bus, "event_time", spec, "bus", logger=log)
                br  = apply_time_filter(br,  "time",       spec, "branch", logger=log)
        except ImportError:
            log.warning("  [v12 时段过滤] time_filter_utils 未找到, 跳过 (仅 legacy --exclude-dates 生效)")

    with Timer("筛选有效电参量列", log):
        keep_cols = []
        for c in data_cols:
            s = bus[c]
            if s.isna().all() or s.nunique(dropna=True) <= 1:
                continue
            keep_cols.append(c)
        log.info(f"  可用电参量列: {len(keep_cols)} / {len(data_cols)}")

    with Timer(f"重采样到 {RESAMPLE} 并对齐分路标签", log):
        df = resample_and_align(bus, br, keep_cols=keep_cols)
        log.info(f"  对齐后样本: {len(df)}, 列数: {df.shape[1]}")

    with Timer("计算特征-标签 Pearson 相关性", log):
        corr = df[keep_cols].corrwith(df["y_ac"]).abs().sort_values(ascending=False)

    corr_path = ARTIFACT_DIR / "feature_corr_with_ac.csv"
    corr.to_csv(corr_path, header=["abs_corr"], encoding="utf-8-sig")
    log.info(f"  相关性排序 -> {corr_path}")
    log.info(f"  Top-10:\n{corr.head(10).to_string()}")

    out_path = ARTIFACT_DIR / "aligned_15min.csv"
    df.to_csv(out_path, encoding="utf-8-sig")
    log.info(f"  对齐数据 -> {out_path}  shape={df.shape}")

    # 标签分布
    log.info("-" * 70)
    log.info("【标签 y_ac 分布】")
    log.info(f"  零样本占比: {(df['y_ac'] == 0).mean()*100:.1f}%")
    if (df['y_ac'] > 0).any():
        log.info(f"  >0 时均值: {df.loc[df['y_ac']>0, 'y_ac'].mean():.2f} W, "
                 f"中位: {df.loc[df['y_ac']>0, 'y_ac'].median():.2f} W")
    log.info(f"  峰值: {df['y_ac'].max():.0f} W")
    log.info("Step 2 完成")


if __name__ == "__main__":
    main()
