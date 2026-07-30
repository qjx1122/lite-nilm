# -*- coding: utf-8 -*-
"""
合并多份历史数据为统一的训练 CSV
- 自动按时间排序去重
- 兼容新旧总线/分路 CSV (不同列结构会取公共列)
"""
import pandas as pd
from pathlib import Path
from common import DATA_DIR, get_logger
from time_utils import parse_timestamps, format_timestamp

log = get_logger("merge", log_to_file=False)


def merge_bus_files(input_files, output_path):
    """合并多个总线 CSV (支持 -/. 多种时间分隔符)"""
    dfs = []
    for f in input_files:
        df = pd.read_csv(f, encoding="utf-8")
        df["event_time"] = parse_timestamps(df["event_time"], logger=log)
        df = df.dropna(subset=["event_time"])
        log.info(f"  [BUS] {Path(f).name}: {df.shape}, "
                 f"时段 {df['event_time'].min()} ~ {df['event_time'].max()}")
        dfs.append(df)
    # 取所有 CSV 的公共列 (避免某些 CSV 列结构差异)
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    common_cols = [c for c in dfs[0].columns if c in common_cols]
    log.info(f"  [BUS] 公共列数: {len(common_cols)}")
    merged = pd.concat([d[common_cols] for d in dfs], ignore_index=True)
    merged = merged.sort_values("event_time").drop_duplicates(subset=["event_time"])
    log.info(f"  [BUS] 合并后: {merged.shape}, "
             f"时段 {merged['event_time'].min()} ~ {merged['event_time'].max()}")
    # 写出时把时间格式转回原始格式
    merged["event_time"] = merged["event_time"].dt.strftime("%Y/%m/%d %H:%M:%S")
    merged.to_csv(output_path, index=False, encoding="utf-8")
    log.info(f"  [BUS] ✓ -> {output_path}")


def merge_branch_files(input_files, output_path):
    """合并多个分路 CSV (支持 -/. 多种时间分隔符)"""
    dfs = []
    for f in input_files:
        df = pd.read_csv(f, encoding="utf-8")
        df["time"] = parse_timestamps(df["time"], logger=log)
        df = df.dropna(subset=["time"])
        log.info(f"  [BR ] {Path(f).name}: {df.shape}, "
                 f"时段 {df['time'].min()} ~ {df['time'].max()}")
        dfs.append(df)
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    common_cols = [c for c in dfs[0].columns if c in common_cols]
    merged = pd.concat([d[common_cols] for d in dfs], ignore_index=True)
    merged = merged.sort_values("time").drop_duplicates(subset=["time"])
    log.info(f"  [BR ] 合并后: {merged.shape}, "
             f"时段 {merged['time'].min()} ~ {merged['time'].max()}")
    merged["time"] = merged["time"].dt.strftime("%Y/%m/%d %H:%M:%S")
    merged.to_csv(output_path, index=False, encoding="utf-8")
    log.info(f"  [BR ] ✓ -> {output_path}")


def main():
    log.info("=" * 72)
    log.info("合并多份历史数据为统一训练 CSV")
    log.info("=" * 72)
    user_id="4206894986488"
    terminal_id = "800080252844"

    # new_data_dir = "数据6.4-6.11"

    # #合并dst_data2使用的原始data时间：src_data1、src_data2、dst_time1
    # src_time1 = "260521-260603"
    # src_time2 = "260604-260611"
    # dst_time1 = "260521-260611"
    # channel = "Ch1"

    # #合并dst_data2使用的原始data时间:、src_time3、dst_time1
    # src_time3 = "260612-260629"
    # dst_time2 = "260521-260629"

    #new_data_dir = "数据6.4-6.11"
    new_data_dir = "infers"
    #合并dst_data2使用的原始data时间：src_data1、src_data2、dst_time1
    src_time1 = "250710-260629"
    src_time2 = "260701-260715"
    dst_time1 = "250710-260715"
    channel = "Ch1"
    
    #合并dst_data2使用的原始data时间:、src_time3、dst_time1
    src_time3 = "260612-260629"
    dst_time2 = "250710-260629"

    # 合并test
    #合并data-1-1
    bus_files = [
        DATA_DIR / "trains" / f"{terminal_id}_{user_id}" / f"e241_{terminal_id}_{user_id}-{channel}-{src_time1}-1.csv",
        DATA_DIR / new_data_dir / f"{terminal_id}_{user_id}" / f"e241_{terminal_id}_{user_id}-{channel}-{src_time2}-1.csv",
    ]
    br_files = [
        DATA_DIR / "trains" / f"{terminal_id}_{user_id}" / f"{user_id}-{src_time1}.csv",
        DATA_DIR / new_data_dir / f"{terminal_id}_{user_id}" / f"{user_id}-{src_time2}.csv",
    ]
    bus_out = DATA_DIR /  "trains" / f"{terminal_id}_{user_id}" / f"e241_{terminal_id}_{user_id}-{channel}-{dst_time1}-1.csv"
    br_out  = DATA_DIR /  "trains" / f"{terminal_id}_{user_id}" / f"{user_id}-{dst_time1}.csv"

    #合并data-2-1
    # bus_files = [
    #     DATA_DIR / "trains" / f"{terminal_id}_{user_id}" / f"e241_{terminal_id}_{user_id}-Ch1-{dst_time1}-1.csv",
    #     DATA_DIR / "infers" / f"{terminal_id}_{user_id}" / f"e241_{terminal_id}_{user_id}-Ch1-{src_time3}-1.csv",
    # ]
    # br_files = [
    #     DATA_DIR / "trains" / f"{terminal_id}_{user_id}" / f"{user_id}-{dst_time1}.csv",
    #     DATA_DIR / "infers" / f"{terminal_id}_{user_id}" / f"{user_id}-{src_time3}.csv",
    # ]
    # bus_out = DATA_DIR /  "trains" / f"{terminal_id}_{user_id}" / f"e241_{terminal_id}_{user_id}-Ch1-{dst_time2}-1.csv"
    # br_out  = DATA_DIR /  "trains" / f"{terminal_id}_{user_id}" / f"{user_id}-{dst_time2}.csv"


    #合并data-1
    # bus_files = [
    #     DATA_DIR / "trains" / "800080270870_4200008003791" / "e241_800080270870_4200008003791-Ch1-260521-260603-1.csv",
    #     DATA_DIR / "数据6.4-6.11" / "800080270870_4200008003791" / "e241_800080270870_4200008003791-Ch1-260604-260611-1.csv",
    # ]
    # br_files = [
    #     DATA_DIR / "trains" / "800080270870_4200008003791" / "4200008003791-260521-260603.csv",
    #     DATA_DIR / "数据6.4-6.11" / "800080270870_4200008003791" / "4200008003791-260604-260611.csv",
    # ]
    # bus_out = DATA_DIR /  "trains" / "800080270870_4200008003791" / "e241_800080270870_4200008003791-Ch1-260521-260611-1.csv"
    # br_out  = DATA_DIR /  "trains" / "800080270870_4200008003791" / "4200008003791-260521-260611.csv"

    # #合并data-2
    # bus_files = [
    #     DATA_DIR / "trains" / "800080270870_4200008003791" / "e241_800080270870_4200008003791-Ch1-260521-260611-1.csv",
    #     DATA_DIR / "infers" / "800080270870_4200008003791" / "e241_800080270870_4200008003791-Ch1-260612-260629-1.csv",
    # ]
    # br_files = [
    #     DATA_DIR / "trains" / "800080270870_4200008003791" / "4200008003791-260521-260611.csv",
    #     DATA_DIR / "infers" / "800080270870_4200008003791" / "4200008003791-260612-260629.csv",
    # ]
    # bus_out = DATA_DIR /  "trains" / "800080270870_4200008003791" / "e241_800080270870_4200008003791-Ch1-260521-260629-1.csv"
    # br_out  = DATA_DIR /  "trains" / "800080270870_4200008003791" / "4200008003791-260521-260629.csv"

    #合并data-3
    # bus_files = [
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-260512-260520-1.csv",
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-260521-260603-1.csv",
    # ]
    # br_files = [
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "4206810972139-260512-260520.csv",
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "4206810972139-260521-260603.csv",
    # ]
    # bus_out = DATA_DIR /  "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-250512-260603-1.csv"
    # br_out  = DATA_DIR /  "trains" / "800080270856_4206810972139" / "4206810972139-250512-260603.csv"

    #合并data-4
    # bus_files = [
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-250512-260603-1.csv",
    #     DATA_DIR / "数据6.4-6.11" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-260604-260611-1.csv",
    # ]
    # br_files = [
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "4206810972139-250512-260603.csv",
    #     DATA_DIR / "数据6.4-6.11" / "800080270856_4206810972139" / "4206810972139-260604-260611.csv",
    # ]
    # bus_out = DATA_DIR /  "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-250512-260611-1.csv"
    # br_out  = DATA_DIR /  "trains" / "800080270856_4206810972139" / "4206810972139-250512-260611.csv"

    #合并data-5
    # bus_files = [
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-250512-260611-1.csv",
    #     DATA_DIR / "infers" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-260612-260629-1.csv",
    # ]
    # br_files = [
    #     DATA_DIR / "trains" / "800080270856_4206810972139" / "4206810972139-250512-260611.csv",
    #     DATA_DIR / "infers" / "800080270856_4206810972139" / "4206810972139-260612-260629.csv",
    # ]
    # bus_out = DATA_DIR /  "trains" / "800080270856_4206810972139" / "e241_800080270856_4206810972139-Ch1-250512-260629-1.csv"
    # br_out  = DATA_DIR /  "trains" / "800080270856_4206810972139" / "4206810972139-250512-260629.csv"
    
    log.info("\n[1] 合并总线 CSV")
    merge_bus_files(bus_files, bus_out)
    log.info("\n[2] 合并分路 CSV")
    merge_branch_files(br_files, br_out)

    log.info("\n" + "=" * 72)
    log.info("合并完成。请将 common.py 中的 BUS_CSV/BR_CSV 改为:")
    log.info(f"  BUS_CSV = DATA_DIR / 'merged_bus.csv'")
    log.info(f"  BR_CSV  = DATA_DIR / 'merged_branch.csv'")
    log.info("然后清空 artifacts/models/logs 重训。")


if __name__ == "__main__":
    main()
