# -*- coding: utf-8 -*-
"""
v6.14 标签清洗器: 用 d87 启动信号确定真实空调起点, 启动前小负荷强制归 0

物理依据 (用户4 提出):
  - d87 是空调启动的真实物理信号 (双向尖峰)
  - 一段"小->大"功率跃迁中, 启动前的小负荷可能是其他设备误叠加
  - 若启动点 (d87 大幅变化) 之前 N 步存在小负荷, 应判定为非空调

清洗逻辑:
  1. 找出所有 d87 启动点 (|d87| >= D87_STARTUP_THR)
  2. 对每个启动点 ts_startup:
     - 寻找该启动点之前最近的"小负荷段" (p1 ∈ [P_SMALL_MIN, P_SMALL_MAX])
     - 这段小负荷被认为是"非空调干扰", 强制归 0
  3. 输出清洗后的 branch DataFrame, p1 列被修改
  4. 同时输出清洗诊断报告

参数:
  D87_STARTUP_THR: d87 启动信号阈值 (从 bundle d87_guard_meta 取或用 50)
  P_SMALL_MIN/MAX: 小负荷判定区间 (默认 (5, 50])
  LOOKBACK_MIN: 启动点前看多少分钟内的小负荷 (默认 60min = 4 步)
"""
import pandas as pd
import numpy as np
from pathlib import Path


def clean_branch_labels(bus_df: pd.DataFrame, br_df: pd.DataFrame, tcol: str,
                        d87_startup_thr: float = 50.0,
                        p_small_min: float = 5.0,
                        p_small_max: float = 50.0,
                        lookback_min: float = 120.0,
                        logger=None) -> tuple[pd.DataFrame, dict]:
    """
    清洗 branch 标签: 在 d87 启动点之前的小负荷段强制归 0
    
    参数:
        bus_df: 总线 5min 数据 (含 event_time 列与 load_iden_data87)
        br_df:  分路 15min 数据 (含 time 列与 tcol)
        tcol:   目标列名 ('p1' 或 'p2' 等)
        d87_startup_thr: d87 启动信号阈值 (|d87| >= 该值算启动点)
        p_small_min/max: 小负荷区间
        lookback_min: 启动点前回看的时长 (分钟), 此窗口内的小负荷段被归 0
    
    返回:
        (cleaned_br, report)
          cleaned_br: 清洗后的 branch DataFrame
          report:     诊断字典 {events_n, cleaned_steps, original_kwh, cleaned_kwh, ...}
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 准备数据
    bus = bus_df.copy()
    if 'event_time' in bus.columns and not isinstance(bus.index, pd.DatetimeIndex):
        bus['ts'] = pd.to_datetime(bus['event_time'])
        bus = bus.set_index('ts').sort_index()
    
    br = br_df.copy()
    if 'time' in br.columns and not isinstance(br.index, pd.DatetimeIndex):
        br['ts'] = pd.to_datetime(br['time'])
        br = br.set_index('ts').sort_index()
    
    if 'load_iden_data87' not in bus.columns:
        log("[label_cleaner] bus 缺 load_iden_data87, 跳过清洗")
        return br_df, {"enabled": False, "reason": "no d87 column"}
    
    if tcol not in br.columns:
        log(f"[label_cleaner] branch 缺 {tcol}, 跳过清洗")
        return br_df, {"enabled": False, "reason": f"no {tcol} column"}
    
    # 1. 找所有 d87 启动点
    d87_abs = bus['load_iden_data87'].abs()
    startup_ts_list = bus.index[d87_abs >= d87_startup_thr].tolist()
    log(f"[label_cleaner] d87 启动点 (|d87|>={d87_startup_thr}): {len(startup_ts_list)} 个")
    
    # 2. 对每个启动点, 找其前 lookback_min 内的"小负荷段", 标记为待清洗
    p_original = br[tcol].copy()
    cleaned_mask = pd.Series(False, index=br.index)
    
    for startup_ts in startup_ts_list:
        # 该启动点对应的 15min 步 (向下取整到 15min)
        # 启动点前 lookback_min 内的步
        window_start = startup_ts - pd.Timedelta(minutes=lookback_min)
        window_end = startup_ts
        # 找该窗口内的 branch 步
        mask = (br.index >= window_start) & (br.index < window_end)
        for ts in br.index[mask]:
            p = br.loc[ts, tcol]
            if p_small_min <= p <= p_small_max:
                cleaned_mask.loc[ts] = True
    
    # 3. 同时: 所有"启动点完全之外的小负荷段"也需要清洗
    # (即: 该 branch 步附近 ±lookback_min 内没有 d87 启动点 + p 在小负荷区间)
    # 这样可以处理"全天没启动但有 21W 持续"的情况
    if startup_ts_list:
        startup_arr = pd.DatetimeIndex(startup_ts_list)
    else:
        startup_arr = pd.DatetimeIndex([])
    
    extra_cleaned = 0
    for ts in br.index:
        if cleaned_mask.loc[ts]:
            continue
        p = br.loc[ts, tcol]
        if not (p_small_min <= p <= p_small_max):
            continue
        # 看该步前后 lookback_min 内是否有启动点
        win_start = ts - pd.Timedelta(minutes=lookback_min)
        win_end = ts + pd.Timedelta(minutes=lookback_min)
        nearby_startups = startup_arr[(startup_arr >= win_start) & (startup_arr <= win_end)]
        if len(nearby_startups) == 0:
            cleaned_mask.loc[ts] = True
            extra_cleaned += 1
    
    # 4. 执行清洗
    cleaned_br = br.copy()
    cleaned_br.loc[cleaned_mask, tcol] = 0.0
    
    # 5. 诊断报告
    n_cleaned = int(cleaned_mask.sum())
    orig_kwh = float(p_original.sum() * 0.25 / 1000)
    cleaned_kwh = float(cleaned_br[tcol].sum() * 0.25 / 1000)
    n_small_total = int(((p_original >= p_small_min) & (p_original <= p_small_max)).sum())
    
    report = {
        "enabled": True,
        "d87_startup_thr": d87_startup_thr,
        "p_small_range": [p_small_min, p_small_max],
        "lookback_min": lookback_min,
        "n_startup_points": len(startup_ts_list),
        "n_small_loads_total": n_small_total,
        "n_cleaned_to_zero": n_cleaned,
        "cleaned_in_startup_window": n_cleaned - extra_cleaned,
        "cleaned_isolated": extra_cleaned,
        "original_kwh": round(orig_kwh, 3),
        "cleaned_kwh": round(cleaned_kwh, 3),
        "kwh_reduction": round(orig_kwh - cleaned_kwh, 3),
    }
    
    log(f"[label_cleaner] 启动点数: {report['n_startup_points']}")
    log(f"[label_cleaner] 小负荷段总数: {n_small_total}")
    log(f"[label_cleaner] 强制归 0 步数: {n_cleaned} "
        f"(启动窗口内 {n_cleaned-extra_cleaned} + 孤立小负荷 {extra_cleaned})")
    log(f"[label_cleaner] kWh 变化: {orig_kwh:.3f} -> {cleaned_kwh:.3f} ({-report['kwh_reduction']:+.3f})")
    
    # 复原 cleaned_br 为带 'time' 列的格式 (与原 br_df 一致)
    if 'time' in br_df.columns:
        cleaned_br_out = cleaned_br.reset_index()
        if 'index' in cleaned_br_out.columns and 'ts' not in cleaned_br_out.columns:
            cleaned_br_out = cleaned_br_out.rename(columns={'index': 'time'})
        elif 'ts' in cleaned_br_out.columns:
            cleaned_br_out = cleaned_br_out.rename(columns={'ts': 'time'})
        # 保留与原 br 一致的列顺序
        cols_keep = [c for c in br_df.columns if c in cleaned_br_out.columns]
        cleaned_br_out = cleaned_br_out[cols_keep]
    else:
        cleaned_br_out = cleaned_br
    
    return cleaned_br_out, report


if __name__ == "__main__":
    # 测试用户1 训练数据清洗效果
    bus = pd.read_csv('/home/user/uploads/e241_800080270848_4206671776099-Ch1-250612-250708-1.csv')
    br = pd.read_csv('/home/user/uploads/4206671776099-250612-250708.csv')
    cleaned, rep = clean_branch_labels(bus, br, 'p1')
    print(f"\n报告: {rep}")
