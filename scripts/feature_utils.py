# -*- coding: utf-8 -*-
"""
特征工程模块 (v5 升级版)

- 训练/推理共享, 保证特征构造一致性
- v5 新增: 12 维温度衍生特征 (依赖 weather_utils 拉取的温度 DataFrame)
"""
import numpy as np
import pandas as pd
from common import SENT_VALUE, RESAMPLE, TARGET_COL
from time_utils import parse_timestamps


# ============================================================
# 数据加载 (与 v4.2 保持一致, 时间解析改用 time_utils 支持多种格式)
# ============================================================
def load_bus_csv(path):
    """加载总线 CSV, 标准化时间列, 处理 INT32_MIN 缺测"""
    df = pd.read_csv(path, encoding="utf-8")
    df["event_time"] = parse_timestamps(df["event_time"])
    df = df.dropna(subset=["event_time"]).sort_values("event_time")
    df = df.reset_index(drop=True)
    data_cols = [c for c in df.columns if c.startswith("load_iden_data")]
    df[data_cols] = df[data_cols].replace(SENT_VALUE, np.nan)
    return df, data_cols


def load_branch_csv(path, target_col: str = None):
    """加载分路 CSV (标签)

    [v13.16] 支持复合目标列语义:
      若 target_col 含 '+' (例 "p1+p2" / "p1+p2+p3"),
      加载后自动新增一列, 列名就是复合字符串, 值 = 各分量按行求和.
      下游代码 (resample_and_align / label_cleaner / analyze_on_periods)
      把它当普通列名使用即可, 无需感知复合语义.

    Args:
        path:        分路 CSV 路径
        target_col:  目标列名 (可选). 若为 None 或不含 '+', 保持旧行为不物化;
                     若含 '+' 且所有分量都存在, 新增复合列.

    Returns:
        DataFrame (若触发物化, 会多出 target_col 命名的一列)
    """
    df = pd.read_csv(path, encoding="utf-8")
    df["time"] = parse_timestamps(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # [v13.16] 复合列物化: 只在 target_col 显式含 '+' 时触发, 避免误伤旧调用
    if target_col and isinstance(target_col, str) and "+" in target_col:
        composite = "".join(target_col.split()).lower()   # 归一化去空白+小写
        parts = composite.split("+")
        # 分量合法性校验 (与 time_filter_utils.get_user_target_col 同源正则)
        import re as _re
        if not all(_re.fullmatch(r"p\d+", p) for p in parts):
            raise ValueError(
                f"[v13.16] load_branch_csv: target_col={target_col!r} 含非法分量, "
                f"每个分量必须 pN 格式 (N ≥ 0 整数)"
            )
        # 检查每个分量都在 CSV 里
        missing = [p for p in parts if p not in df.columns]
        if missing:
            raise KeyError(
                f"[v13.16] 分路 CSV 缺少 target_col={target_col!r} 的分量列 "
                f"{missing}, 实际列: {df.columns.tolist()}"
            )
        # 强制数值化后逐行相加 (原始列如含空字符串 → NaN, sum(skipna=False) 传播 NaN
        # 避免"部分分量缺采样点"时被静默补 0 导致电量偏低)
        parts_df = df[parts].apply(pd.to_numeric, errors="coerce")
        df[composite] = parts_df.sum(axis=1, skipna=False)

    return df


def resample_and_align(bus_df, branch_df=None, keep_cols=None,
                       max_gap_steps: int = 2):
    """
    将总线数据 resample 到 15min, 可选与分路标签 inner-join。
    返回的 DataFrame index 为 DatetimeIndex
    
    参数:
        max_gap_steps : 允许 ffill 填充的最大连续 NaN 步数 (默认 2 = 30 分钟)
                        防止"无中生有": 总线长时间停采 (如停电/故障) 时,
                        不应填充, 而是删除这些样本.
                        典型场景:
                          - max_gap_steps=2  (30min):  容忍小漏点, 严禁长间隙
                          - max_gap_steps=4  (1h):     更宽松
                          - max_gap_steps=0:           完全不 ffill, 任何 NaN 直接删除
    """
    if keep_cols is None:
        keep_cols = [c for c in bus_df.columns if c.startswith("load_iden_data")]

    # ---- v6.12.a: 关键尖峰列保留 5min 极值聚合 (方案 A 精选版) ----
    # 物理依据 (硬证据): load_iden_data87 在空调启动时呈现 5min 单步负向冲击
    #   (训练集 75 个 ON 事件: |d87|>=144; 全天 OFF 7 天: |d87|<=35).
    #   但默认 resample("15min").mean() 会把 -224 的尖峰平滑到 -62,
    #   导致模型完全无法学到这一可靠的物理签名 (-> 推理 SAE 46.87% Bug 根因).
    # 因此对 SPIKE_COLS 额外保留 5min 内的 max/min/abs_max 三个聚合.
    # 处理 keep_cols 中可能已包含衍生列名 (例如推理时 top_cols 含 _max5):
    SPIKE_SUFFIXES = ("_max5", "_min5", "_absmax5")
    raw_keep_cols = [c for c in keep_cols if not c.endswith(SPIKE_SUFFIXES)]
    # 从 keep_cols 中推断出 spike 源列 (如 keep_cols 含 load_iden_data87_max5,
    # 则需要 load_iden_data87 作为源)
    requested_spike_bases = set()
    for c in keep_cols:
        if c.endswith(SPIKE_SUFFIXES):
            for suf in SPIKE_SUFFIXES:
                if c.endswith(suf):
                    requested_spike_bases.add(c[: -len(suf)])
                    break
    # 默认 SPIKE_COLS (训练阶段会自动注入)
    DEFAULT_SPIKE_COLS = ["load_iden_data87"]
    spike_cols_all = sorted(set(DEFAULT_SPIKE_COLS) | requested_spike_bases)
    # 源列只需在 bus_df 中存在 (推理时 top_cols 可能只含派生列 _max5)
    spike_cols_present = [c for c in spike_cols_all if c in bus_df.columns]

    # 1) 主分支: raw_keep_cols 的 mean 聚合
    bus_for_mean = bus_df.set_index("event_time")[raw_keep_cols]
    bus_rs = bus_for_mean.resample(RESAMPLE, label="left", closed="left").mean()

    # 2) 旁路: spike 源列的 max/min/absmax 聚合 (独立 resample, 不依赖 raw_keep_cols)
    extras = []
    for sc in spike_cols_present:
        s = bus_df.set_index("event_time")[sc]
        rs = s.resample(RESAMPLE, label="left", closed="left")
        extras.append(rs.max().rename(f"{sc}_max5"))
        extras.append(rs.min().rename(f"{sc}_min5"))
        extras.append(s.abs().resample(RESAMPLE, label="left", closed="left")
                       .max().rename(f"{sc}_absmax5"))
    if extras:
        bus_rs = pd.concat([bus_rs] + extras, axis=1)

    # 后续 ffill/dropna 用扩展后的列集合
    keep_cols = list(bus_rs.columns)

    if branch_df is not None:
        if TARGET_COL not in branch_df.columns:
            raise KeyError(
                f"分路 CSV 缺少目标列 '{TARGET_COL}', "
                f"实际列: {list(branch_df.columns)}. "
                f"请检查 common.py 中 TARGET_COL 配置或 CSV 列名."
            )
        br_idx = branch_df.set_index("time")[[TARGET_COL]] \
                          .rename(columns={TARGET_COL: "y_ac"})
        df = bus_rs.join(br_idx, how="inner").dropna(subset=["y_ac"])
    else:
        df = bus_rs.copy()

    # 关键修复 (v6.7): 限制 ffill 距离, 防止总线长间隙被无声填充
    # 旧版 .ffill().bfill() 会把短期快照"广播"到整天, 造成虚假预测
    n_before = len(df)
    if max_gap_steps > 0:
        df[keep_cols] = df[keep_cols].ffill(limit=max_gap_steps) \
                                     .bfill(limit=max_gap_steps)
    # 删除仍有 NaN 的行 (即长间隙时段)
    df = df.dropna(how="any", subset=keep_cols)
    n_after = len(df)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        import logging
        logging.getLogger("nilm").info(
            f"  [align] 因总线长间隙(>{max_gap_steps*15}min) 丢弃 "
            f"{n_dropped} 行 ({n_dropped/max(n_before,1)*100:.1f}%)"
        )
    return df


# ============================================================
# 特征工程 (v5: 60 + 12 = 72 维)
# ============================================================
def build_features(df: pd.DataFrame, top_cols: list,
                   weather_df: pd.DataFrame = None,
                   temp_power_lut: dict = None) -> pd.DataFrame:
    """
    构造最终特征矩阵 (训练/推理严格一致)
    输入:
        df:             对齐后的总线数据 (含 top_cols), DatetimeIndex
        top_cols:       训练时选定的 Top-25 原始电参量列名
        weather_df:     可选, 15min 粒度温度 DataFrame
        temp_power_lut: 可选 (v6 新增), 温度-功率历史查找表, 用于构造漂移特征

    输出:
        X_df: 60 / 72 / 77 / 128 维特征 DataFrame (v6.11 升级)
            - 25 原始电参量
            - 10 一阶差分 (Top-10)
            - 10 滚动统计 (Top-5 × {mean_4, std_4})
            -  6 滞后项 (Top-3 × {lag1, lag2})
            -  9 时间特征 (含 sin/cos 周期编码)
            -[12 温度特征 - v5, 当 weather_df 给定时]
            -[ 5 漂移特征 - v6, 当 temp_power_lut 给定时]
            -- v6.11 新增 突变感知特征 (针对漏识别事件 7/12, 4/11, 4/30):
            - 20 多步差分     (Top-10 × {d3, d6})           - 捕捉 45min/90min 突变
            - 10 绝对差分     (Top-5 × {abs_d1, abs_d3})    - 突变幅度信号
            -  6 突变方向标志 (Top-3 × {up_d1, down_d1})    - 显式区分开/关机
            -  6 窗口极差     (Top-3 × {range_4, range_12}) - 1h/3h 波动范围
            -  9 EMA 跨尺度   (Top-3 × {ema_2, ema_24, ratio_ema}) - 短长期对比
    """
    X = df[top_cols].copy()

    # ---- v6.12.a: 5min 极值聚合作为新特征 (方案 A 精选版) ----
    # 由 resample_and_align 注入的 _max5/_min5/_absmax5 列, 直接作为原始特征加入.
    # 这些列保留了 5min 内的尖峰信息 (如 d87 启动冲击 -224W),
    # 弥补默认 15min mean 平滑掉的关键物理签名.
    SPIKE_SUFFIXES = ("_max5", "_min5", "_absmax5")
    spike_feat_cols = [c for c in df.columns if c.endswith(SPIKE_SUFFIXES)]
    for c in spike_feat_cols:
        if c not in X.columns:
            X[c] = df[c].values

    # ---- v6.12.3: d87 启动事件二值特征 (方案 #2) ----
    # 物理依据 (89 个真实启动事件验证):
    #   - 96% 启动事件 d87 单步出现 |Δd87| > 50 的瞬态尖峰
    #   - 4% 启动事件 d87 是正向尖峰 (用户2 5/23: d87 +176)
    #   - 维持 ON/OFF 时段 d87 全部稳定在 ±20 噪声内
    # 设计目的:
    #   把"启动这一瞬间"显式做成模型可看的特征, 让 d87 实际进入决策
    #   (之前 _max5/_min5 是聚合, 模型仅能看到"今天极值", 无法区分启动 vs 维持)
    D87_SRC = "load_iden_data87"
    D87_MIN5 = f"{D87_SRC}_min5"   # 来自上面 SPIKE_SUFFIXES 注入
    D87_MAX5 = f"{D87_SRC}_max5"
    if D87_MIN5 in df.columns and D87_MAX5 in df.columns:
        # (a) 当步内 5min 跨度 (max-min): 启动瞬态的核心指标, ON/OFF 维持期都接近 0
        d87_jump = (df[D87_MAX5] - df[D87_MIN5]).fillna(0)
        X["d87_jump_abs5"] = d87_jump

        # (b) 该步 |d87| 极值 (双向, 不分 min/max)
        d87_amax = np.maximum(df[D87_MAX5].abs(), df[D87_MIN5].abs()).fillna(0)
        X["d87_amax5"] = d87_amax

        # (c) 启动事件二值标志: |d87| 超过 50 (固定阈值, 与训练用户 OFF 段 P1=-32 对应)
        EVENT_THR = 50.0
        X["d87_event_neg"] = (df[D87_MIN5] < -EVENT_THR).astype(np.int8).fillna(0)
        X["d87_event_pos"] = (df[D87_MAX5] >  EVENT_THR).astype(np.int8).fillna(0)
        X["d87_event_any"] = ((X["d87_event_neg"] | X["d87_event_pos"]) > 0).astype(np.int8)

        # (d) 近 3 步 (45min) 滚动启动事件标志 (启动后 ON 维持期保持记忆)
        X["d87_event_recent_3"] = X["d87_event_any"].rolling(
            window=3, min_periods=1).max().fillna(0).astype(np.int8)

        # (e) 启动尖峰强度比 (d87_jump 与 _max5 + _min5 绝对和的比值)
        # 启动时 d87_jump 远大于该步原始波动; 维持时两者接近
        # 用 |max5|+|min5| 作为"该步整体波动"分母, 不依赖原始 d87 列
        denom = (df[D87_MAX5].abs() + df[D87_MIN5].abs() + 5.0).fillna(5.0)
        X["d87_spike_ratio"] = (d87_jump / denom).fillna(0)

    # --- (1) 一阶差分 (前 10 列) ---
    for c in top_cols[:10]:
        X[f"{c}_d1"] = df[c].diff().fillna(0)

    # --- (2) 滚动统计 (前 5 列, 窗口 4=1h) ---
    for c in top_cols[:5]:
        X[f"{c}_rm4"] = df[c].rolling(window=4, min_periods=1).mean()
        X[f"{c}_rs4"] = df[c].rolling(window=4, min_periods=1).std().fillna(0)

    # --- (3) 滞后项 (前 3 列, lag1 / lag2) ---
    for c in top_cols[:3]:
        X[f"{c}_lag1"] = df[c].shift(1).bfill()
        X[f"{c}_lag2"] = df[c].shift(2).bfill()

    # ============================================================
    # v6.11 突变感知特征 (针对漏识别事件 7/12, 4/11, 4/30)
    # ============================================================
    # 根因背景:
    #   事件 1 (7/12 08:15): 总线从 0 -> 847W 一步突变, lag/rolling 全在 OFF 区
    #   事件 2 (4/11 17:45): 功率从 720W 降到 430W, p_on 在阈值附近抖动
    #   事件 3 (4/30 14:45): 关机前衰减带 466->314W, 模型彻底"认为已关"
    # 设计原则:
    #   1. 多尺度差分 (1/3/6 步): 覆盖 15min/45min/90min 时间窗
    #   2. 绝对值 + 方向: 让 GBDT 显式区分"突变幅度"与"开/关方向"
    #   3. 窗口极差: 反映短时波动, 区分稳定段 vs 切换段
    #   4. EMA 跨尺度比: 提供"近期 vs 昨日"的对比信号
    # ============================================================

    # --- (3a) 多步差分 (前 10 列 × {d3, d6} = 20 维) ---
    # d3 = 45min 前差分, d6 = 90min 前差分; 捕捉滞后于 d1 的累积突变
    for c in top_cols[:10]:
        X[f"{c}_d3"] = df[c].diff(periods=3).fillna(0)
        X[f"{c}_d6"] = df[c].diff(periods=6).fillna(0)

    # --- (3b) 绝对差分 (前 5 列 × {abs_d1, abs_d3} = 10 维) ---
    # 突变幅度比方向更通用 (开机/关机都是大幅变化)
    for c in top_cols[:5]:
        X[f"{c}_abs_d1"] = df[c].diff().abs().fillna(0)
        X[f"{c}_abs_d3"] = df[c].diff(periods=3).abs().fillna(0)

    # --- (3c) 突变方向标志 (前 3 列 × {up_d1, down_d1} = 6 维) ---
    # 显式区分: up = 突然增加 (开机), down = 突然减少 (关机/降档)
    # 阈值用 50W (远低于事件 1 的 847W 和事件 2 的 250W 跌幅, 但足以过滤噪声)
    JUMP_THRESHOLD_W = 50.0
    for c in top_cols[:3]:
        d1 = df[c].diff().fillna(0)
        X[f"{c}_up_d1"]   = (d1 >  JUMP_THRESHOLD_W).astype(np.int8)
        X[f"{c}_down_d1"] = (d1 < -JUMP_THRESHOLD_W).astype(np.int8)

    # --- (3d) 窗口极差 (前 3 列 × {range_4, range_12} = 6 维) ---
    # range = max - min, 反映窗口内波动幅度
    # 稳定 ON / 稳定 OFF 段 range 都很小; 切换段 range 突增
    for c in top_cols[:3]:
        r4_max = df[c].rolling(window=4, min_periods=1).max()
        r4_min = df[c].rolling(window=4, min_periods=1).min()
        X[f"{c}_range_4"] = (r4_max - r4_min).fillna(0)
        r12_max = df[c].rolling(window=12, min_periods=1).max()
        r12_min = df[c].rolling(window=12, min_periods=1).min()
        X[f"{c}_range_12"] = (r12_max - r12_min).fillna(0)

    # --- (3e) EMA 跨尺度 (前 3 列 × {ema_2, ema_24, ratio_ema} = 9 维) ---
    # ema_2  = 30min 半衰期, 灵敏跟踪当前状态
    # ema_24 = 6h 半衰期, 反映"今天上午平均水平"
    # ratio  = ema_2 / ema_24, > 1.2 表示当前显著高于近期基线 (可能开机)
    #                          < 0.8 表示当前显著低于近期基线 (可能关机/降档)
    for c in top_cols[:3]:
        ema2  = df[c].ewm(halflife=2,  adjust=False).mean()
        ema24 = df[c].ewm(halflife=24, adjust=False).mean()
        X[f"{c}_ema_2"]  = ema2
        X[f"{c}_ema_24"] = ema24
        # 避免除 0: 分母 + 1.0 (典型功率量级 100~1000W, 加 1 无影响)
        X[f"{c}_ratio_ema"] = (ema2 / (ema24 + 1.0)).fillna(1.0)

    # --- (4) 时间特征 (含周期编码 + 季节) ---
    ts = df.index
    X["hour"] = ts.hour
    X["dow"]  = ts.dayofweek
    X["is_evening"] = ((ts.hour >= 18) | (ts.hour <= 1)).astype(int)
    X["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    X["sin_hour"] = np.sin(2 * np.pi * ts.hour / 24)
    X["cos_hour"] = np.cos(2 * np.pi * ts.hour / 24)
    X["month"]    = ts.month
    doy = ts.dayofyear
    X["sin_doy"]  = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"]  = np.cos(2 * np.pi * doy / 365.25)

    # --- (5) [v5] 温度特征 (12 维, 仅当 weather_df 提供) ---
    if weather_df is not None and not weather_df.empty:
        X = _add_weather_features(X, df.index, weather_df)

    # --- (6) [v6] 漂移感知特征 (5 维, 仅当 temp_power_lut 提供) ---
    if temp_power_lut is not None:
        from drift_features import build_drift_features
        drift = build_drift_features(df, top_cols,
                                     weather_df=weather_df,
                                     temp_power_lut=temp_power_lut)
        for c in drift.columns:
            X[c] = drift[c].values

    return X


def assert_no_nan_features(X_df, stage_name: str = "unknown", logger=None,
                           raise_on_nan: bool = True) -> dict:
    """[v13.7] 特征矩阵 NaN 硬检测 - 03/04/05 三阶段共用

    动机 (硬证据):
      270758 用户配置2 推理时报错 `ValueError: Input X contains NaN. GradientBoostingClassifier
      does not accept missing values encoded as NaN natively.` 定位到 05_inference.py L262
      `p_on = clf.predict_proba(X_s)[:, 1]`.

      根因: X_df 里存在 NaN 但推理路径没做防线. NaN 来源可能:
        (a) feature_utils.build_features L213 `rolling(...).mean()` 无 fillna
        (b) _add_weather_features `daily.reindex(ts_index.normalize(), method='ffill')`
            当 weather_df 起点晚于 ts_index 起点时会产 NaN
        (c) weather API 返回的 apparent_temperature / humidity 字段本身含 NaN
        (d) drift_features 某些边界组合

      Linux 沙箱恰好不复现是因为 pandas 版本 + 缓存气象数据完整. Windows 用户环境
      因 pandas/numpy 微版本差异触发. 为保数据质量透明, 加硬检测:
        - 检测到 NaN -> WARN 打印 (列名 / NaN 数 / 首个 NaN 时间戳/行号) + raise ValueError

    参数:
        X_df:          build_features() 返回的 DataFrame (n_rows x n_features)
        stage_name:    诊断日志前缀, 如 "train" / "evaluate" / "inference"
        logger:        Logger 对象, 若 None 则用 print
        raise_on_nan:  True (默认) 检测到 NaN 抛异常; False 仅记录不抛

    返回:
        dict {n_nan_cols, n_nan_rows, first_col, first_ts, all_nan_cols}
        无 NaN 时 dict 全 0 / 空.

    抛出:
        ValueError: 检测到 NaN 且 raise_on_nan=True
    """
    import pandas as _pd  # 局部 import 避免顶层 pd 依赖
    import numpy as _np

    _log = (logger.warning if logger is not None else print)
    _info = (logger.info if logger is not None else print)

    if X_df is None or len(X_df) == 0:
        _info(f"  [v13.7 NaN 检测/{stage_name}] X_df 为空, 跳过检测")
        return {"n_nan_cols": 0, "n_nan_rows": 0, "first_col": None,
                "first_ts": None, "all_nan_cols": []}

    na_counts = X_df.isna().sum()
    bad_cols = na_counts[na_counts > 0].sort_values(ascending=False)
    n_bad_cols = int(len(bad_cols))

    if n_bad_cols == 0:
        _info(f"  [v13.7 NaN 检测/{stage_name}] [OK] X_df shape={X_df.shape}, 无 NaN")
        return {"n_nan_cols": 0, "n_nan_rows": 0, "first_col": None,
                "first_ts": None, "all_nan_cols": []}

    # 有 NaN: 详细定位
    row_has_nan = X_df.isna().any(axis=1)
    n_bad_rows = int(row_has_nan.sum())
    first_col = bad_cols.index[0]
    first_row_idx = X_df.index[X_df[first_col].isna()][0]

    # WARN 打印详情 (Top-10 列)
    top10 = bad_cols.head(10).to_dict()
    _log(f"  [v13.7 NaN 检测/{stage_name}] [FAIL] X_df 含 NaN!")
    _log(f"    - 特征矩阵 shape       : {X_df.shape}")
    _log(f"    - 含 NaN 的列数         : {n_bad_cols} / {X_df.shape[1]}")
    _log(f"    - 含 NaN 的行数         : {n_bad_rows} / {X_df.shape[0]}")
    _log(f"    - Top-10 NaN 列 (col:count): {top10}")
    _log(f"    - 首个 NaN 列          : {first_col}")
    _log(f"    - 首个 NaN 行索引/时间戳: {first_row_idx}")
    _log(f"    诊断建议:")
    _log(f"    (a) 若列名以 _rm4/_ema/_lag 结尾: feature_utils rolling/ewm/shift 边界产 NaN")
    _log(f"    (b) 若列名含 temp_/apparent/humidity: 气象数据头/尾对齐问题 (检查 _add_weather_features)")
    _log(f"    (c) 若列名含 power_recent_/temp_power_residual: drift_features 边界 (检查 build_drift_features)")
    _log(f"    (d) 请把上述定位信息发给算法组; 或用 --time-filter-spec 排除边界日期后重试")

    result = {
        "n_nan_cols": n_bad_cols,
        "n_nan_rows": n_bad_rows,
        "first_col": first_col,
        "first_ts": first_row_idx,
        "all_nan_cols": list(bad_cols.index),
    }

    if raise_on_nan:
        raise ValueError(
            f"[v13.7/{stage_name}] X_df 含 NaN, 不能送入 sklearn GBM. "
            f"详情: {n_bad_cols} 列 / {n_bad_rows} 行有 NaN, "
            f"首个 NaN = '{first_col}' @ {first_row_idx}. "
            f"参见上方 WARN 日志的诊断建议."
        )
    return result


def _add_weather_features(X: pd.DataFrame, ts_index: pd.DatetimeIndex,
                          weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    添加 12 维温度衍生特征 (v5)
    设计原则: 既覆盖即时温度, 也覆盖统计/偏差/趋势/物理派生
    """
    # 1) 即时对齐温度 (3 维)
    w = weather_df.reindex(ts_index, method="nearest")
    X["temp_2m"]            = w["temperature_2m"].values
    X["apparent_temp"]      = w["apparent_temperature"].values
    X["humidity"]           = w["relative_humidity_2m"].values

    # 2) 当日统计 (3 维): 当日 max/min/mean
    daily = weather_df["temperature_2m"].resample("D").agg(["max", "min", "mean"])
    daily_reidx = daily.reindex(ts_index.normalize(), method="ffill")
    X["temp_day_max"]       = daily_reidx["max"].values
    X["temp_day_min"]       = daily_reidx["min"].values
    X["temp_day_mean"]      = daily_reidx["mean"].values

    # 3) 偏差与跨度 (2 维)
    #    日较差 = max - min  反映晴/阴
    X["temp_diurnal_range"] = X["temp_day_max"] - X["temp_day_min"]
    #    季节常值偏离 = 当日均温 - 该月历史均值 (用 doy 周期编码近似)
    #    简化为 与 30 年武汉气候同期均值的偏离
    seasonal_baseline = _monthly_climatology_for_index(ts_index)
    X["temp_deviation"]     = X["temp_day_mean"] - seasonal_baseline

    # 4) 趋势 (2 维)
    #    短期变化 = 3h 温度变化, 反映急升急降
    temp_3h = w["temperature_2m"].diff(periods=12).fillna(0)   # 3h = 12 × 15min
    X["temp_change_3h"]     = temp_3h.values
    #    长期变化 = 24h 温度变化
    temp_24h = w["temperature_2m"].diff(periods=96).fillna(0)  # 24h = 96 × 15min
    X["temp_change_24h"]    = temp_24h.values

    # 5) 物理派生 (2 维): 制冷/制热度时
    #    CDH = max(0, T - 26)  超过 26℃ 部分, 直接驱动制冷负荷
    X["cooling_degree"]     = np.clip(X["temp_2m"].values - 26.0, 0, None)
    #    HDH = max(0, 16 - T)  低于 16℃ 部分, 驱动制热负荷
    X["heating_degree"]     = np.clip(16.0 - X["temp_2m"].values, 0, None)

    return X


def _monthly_climatology_for_index(ts_index: pd.DatetimeIndex) -> np.ndarray:
    """返回每个时间戳对应的"30 年武汉同期气候平均"温度"""
    from weather_utils import MONTHLY_AVG_TEMP_WUHAN
    return np.array([MONTHLY_AVG_TEMP_WUHAN[m] for m in ts_index.month])
