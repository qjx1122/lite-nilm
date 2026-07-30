# -*- coding: utf-8 -*-
"""[v13.5+ / v13.6] 分路数据开机时段分析脚本

功能:
  加载指定用户 (或指定 CSV) 的分路数据, 根据 ON_THR_W 阈值二值化,
  按自然日拆分连续 ON 段, 输出:
    (1) 段级明细 CSV: 每段起止 + 统计量
    (2) [v13.6] 每日汇总 CSV: 按日聚合总时长/总电量/开机段数

段级明细列 (与用户示例对齐, 追加统计):
  being_time, end_time, <target_col>, duration_min, min_w, mean_w, peak_w, energy_kwh
    - being_time / end_time: 该 ON 段的起止 (含端点, 格式 YYYY/M/D H:MM:SS)
    - <target_col>: 恒为 1 (仅输出 ON 段)
    - duration_min: 时长 (分钟, 含末点采样区间)
    - min_w:  [v13.16] 该段最小瞬时功率 (W); OFF 天行 = 全天最小 (待机下限)
    - mean_w: 该段平均功率 (W)
    - peak_w: 该段峰值功率 (W)
    - energy_kwh: 该段电量 (kWh) = sum(w * dt_h)

[v13.6] 每日汇总列 (--daily-out 或默认与 --out 同目录):
  date, n_segments, total_on_min, total_on_hours,
  first_on_time, last_off_time, min_w, mean_w, peak_w, energy_kwh
    - date: 自然日 (YYYY-MM-DD)
    - n_segments: 该日 ON 段数
    - total_on_min / total_on_hours: 该日总开机时长
    - first_on_time / last_off_time: 首次开机 / 末次关机时间 (HH:MM:SS)
    - min_w:  [v13.16] ON 天=各 ON 段 min_w 的最小 (=开机期间最低瞬时功率); OFF 天=全天最小
    - mean_w: 加权平均功率 (按 ON 采样点算术平均)
    - peak_w: 该日峰值
    - energy_kwh: 该日累计用电

设计要点:
  1. 严格按每个采样点判定, 不合并 (raw_strict) - 一遇 < ON_THR_W 立即断段
  2. 跨自然日的 ON 段按 00:00 拆分为多行, 便于"每日"统计
  3. 三层优先级链读取 on_thr_w 和 target_col (与 v13.1/v13.4/v13.5 一致):
     配置文件 user_id → _default → common.py 全局默认
  4. 支持双调用模式:
     a) --user <folder_name> --stage train|infer [--config <path>]
        自动定位 data/trains|infers/<folder_name>/ 下的分路 CSV
     b) --br-csv <path> [--target-col pN] [--on-thr-w <W>]
        显式指定 CSV 路径 (target_col 未给则读第 1 个 pN 列)

用法示例:
  # 模式 A: 用户 ID + 阶段 (读配置文件的 target_col 和 on_thr_w)
  python analyze_on_periods.py \\
      --user 800080270708_4206602981958 \\
      --stage train \\
      --config ../data/time_filters.example.json \\
      --out ../artifacts/270708_train_on_periods.csv

  # 模式 B: 显式 CSV + 阈值 (不需要配置文件)
  python analyze_on_periods.py \\
      --br-csv ../data/trains/800080270708_4206602981958/4206602981958-260612-260629.csv \\
      --target-col p1 \\
      --on-thr-w 50 \\
      --out ../artifacts/270708_p1_on_periods.csv

依赖: pandas, common.py (读默认阈值), time_filter_utils.py (可选, 用于读配置)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 默认阈值兜底 (与 common.py 一致 = 10.0W)
try:
    from common import ON_THR_W as _DEFAULT_ON_THR_W
except Exception:
    _DEFAULT_ON_THR_W = 10.0


# ---------- 工具函数 ----------

# 单个 pN 列 (向后兼容)
_RE_PN = re.compile(r"^p\d+$", re.IGNORECASE)
# [v13.16] 单列或复合 (+连接): p1 / p2 / p1+p2 / p1+p2+p3
_RE_PN_COMPOSITE = re.compile(r"^p\d+(\+p\d+)*$", re.IGNORECASE)


def _normalize_target_col(s: str) -> str:
    """[v13.16] 归一化: 去所有空白 + 小写. 允许 "P1 + p2" -> "p1+p2"."""
    return "".join(str(s).split()).lower()


def _materialize_composite_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """[v13.16] 若 target_col 含 '+', 在 df 上新增复合列 (值 = 分量按行求和).

    与 feature_utils.load_branch_csv 的物化逻辑等价 (语义单一来源).

    Args:
        df:         已加载的 DataFrame (必须已含所有分量列)
        target_col: 目标列名; 若不含 '+' 直接返回原 df

    Returns:
        df (原对象或添加了复合列的新对象)
    """
    if not target_col or "+" not in target_col:
        return df
    composite = _normalize_target_col(target_col)
    parts = composite.split("+")
    missing = [p for p in parts if p not in df.columns]
    if missing:
        raise KeyError(
            f"[v13.16] analyze_on_periods: 分路数据缺少 target_col={target_col!r} "
            f"的分量列 {missing}, 实际列: {df.columns.tolist()}"
        )
    if composite not in df.columns:
        parts_df = df[parts].apply(pd.to_numeric, errors="coerce")
        df = df.copy()
        df[composite] = parts_df.sum(axis=1, skipna=False)
    return df


def _fmt_ts(ts: pd.Timestamp) -> str:
    """输出格式与用户示例一致: 2026/5/21 9:00:00 (无前导 0)."""
    return f"{ts.year}/{ts.month}/{ts.day} {ts.hour}:{ts.minute:02d}:{ts.second:02d}"


def _locate_br_csv(user_id: str, stage: str) -> Path:
    """[模式 A] 在 data/trains|infers/<user_id>/ 下定位分路 CSV.
    分路命名: <user>-<start>-<end>[-1|-infer].csv (无 e241_ 前缀).
    """
    assert stage in ("train", "infer"), f"stage 必须是 train/infer, 收到 {stage!r}"
    sub = "trains" if stage == "train" else "infers"
    d = PROJECT_ROOT / "data" / sub / user_id
    if not d.exists():
        raise FileNotFoundError(f"目录不存在: {d}")

    # 分路 CSV: 不以 e241_ 开头
    candidates = sorted(
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv" and not f.name.startswith("e241_")
    )
    if not candidates:
        raise FileNotFoundError(f"目录 {d} 下未找到分路 CSV (非 e241_ 开头的 .csv)")
    if len(candidates) > 1:
        print(f"[WARN] 目录 {d} 下有多个分路 CSV, 取字典序第 1 个: {candidates[0].name}")
    return candidates[0]


def _resolve_target_col(
    br_csv: Path,
    explicit: Optional[str],
    config: Optional[dict],
    user_id: Optional[str],
) -> str:
    """target_col 三层优先级链:
    1. 命令行 --target-col (最高)
    2. 配置文件 config[user_id].target_col 或 config[_default].target_col
    3. 分路 CSV 第 1 个 pN 列 (兜底)

    [v13.16] 支持复合 target_col ('pA+pB[+pC...]'), 例 'p1+p2'.
    兜底路径不构造复合列 (无从推断用户意图), 只返回第 1 个 pN.
    """
    # (1) 命令行显式
    if explicit:
        col = _normalize_target_col(explicit)
        if not _RE_PN_COMPOSITE.match(col):
            raise ValueError(
                f"--target-col 格式必须 pN 或 pA+pB[+pC...] (N ≥ 0), "
                f"收到 {explicit!r}"
            )
        return col

    # (2) 配置文件
    if config is not None and user_id is not None:
        try:
            from time_filter_utils import get_user_target_col
            col = get_user_target_col(config, user_id)
            if col:
                return col
        except Exception as e:
            print(f"[WARN] 读配置 target_col 失败, 回退到 CSV 第 1 列 pN: {e}")

    # (3) 兜底: CSV 第 1 个 pN 列
    cols = pd.read_csv(br_csv, nrows=1).columns.tolist()
    p_cols = [c for c in cols if _RE_PN.match(c.strip())]
    if not p_cols:
        raise ValueError(f"分路 CSV {br_csv} 未找到任何 pN 列 (实际列: {cols})")
    return p_cols[0].strip().lower()


def _resolve_on_thr_w(
    explicit: Optional[float],
    config: Optional[dict],
    user_id: Optional[str],
) -> float:
    """on_thr_w 三层优先级链:
    1. 命令行 --on-thr-w
    2. 配置文件 user_id → _default 的 on_thr_w
    3. common.py 默认 (=10.0)
    """
    if explicit is not None:
        v = float(explicit)
        if not (0 < v <= 5000):
            raise ValueError(f"--on-thr-w 必须在 (0, 5000] W, 收到 {v}")
        return v

    if config is not None and user_id is not None:
        try:
            from time_filter_utils import get_user_common_overrides
            ov = get_user_common_overrides(config, user_id)
            if "on_thr_w" in ov:
                return float(ov["on_thr_w"])
        except Exception as e:
            print(f"[WARN] 读配置 on_thr_w 失败, 回退到 common 默认: {e}")

    return float(_DEFAULT_ON_THR_W)


# ---------- 核心算法 ----------

def compute_on_periods(
    df: pd.DataFrame,
    target_col: str,
    on_thr_w: float,
    split_by_day: bool = True,
    date_labels: dict = None,
) -> pd.DataFrame:
    """从时间序列分路数据中提取连续 ON 段.

    参数:
        df:  必含 'time' 列 (可解析时间戳) 和 target_col 功率列
        target_col: 目标分路列名, 如 'p1'
        on_thr_w: ON 判定阈值 (W), state = (w >= on_thr_w)
        split_by_day: 跨自然日 ON 段是否按 00:00 拆分为多行
        date_labels: [v13.10] 可选, dict {"yyyy-mm-dd": "train"/"val"/"test"/"未使用"/"used"/"excluded"}
                    若提供, 输出 CSV 追加 `dataset` 列, 每段/每日按 being_time 的日期查找标签.
                    - 训练阶段推荐值: {train_dates -> "train", val_dates -> "val",
                                       test_dates -> "test", 其余 -> "未使用"}
                    - 推理阶段推荐值: {推理集日 -> "used", 被 exclude 的日 -> "excluded"}
                    - 未在 dict 中的日期显示为空字符串
                    向后兼容: 不传 = 不加 dataset 列

    返回 DataFrame, 列:
        being_time, end_time, <target_col>=1,
        duration_min, mean_w, peak_w, energy_kwh
        [+ dataset (可选)]
    """
    if "time" not in df.columns:
        raise KeyError(f"CSV 缺少 'time' 列, 实际列: {df.columns.tolist()}")

    # [v13.16] 若 target_col 是复合 (如 "p1+p2"), 先按需物化
    if target_col and "+" in target_col and target_col not in df.columns:
        df = _materialize_composite_target(df, target_col)

    if target_col not in df.columns:
        raise KeyError(f"CSV 缺少目标列 {target_col!r}, 实际列: {df.columns.tolist()}")

    d = df[["time", target_col]].copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    n_bad = int(d["time"].isna().sum() + d[target_col].isna().sum())
    if n_bad > 0:
        print(f"[WARN] 剔除 {n_bad} 行 time/{target_col} 解析失败的记录")
    d = d.dropna(subset=["time", target_col]).sort_values("time").reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=[
            "being_time", "end_time", target_col,
            "duration_min", "min_w", "mean_w", "peak_w", "energy_kwh",  # [v13.16]
        ])

    # 估算采样步长 (中位数); 用于末点持续时间 & 能量
    dt_sec = d["time"].diff().dt.total_seconds().dropna()
    if len(dt_sec) == 0:
        step_sec = 900.0  # 兜底 15 min
    else:
        step_sec = float(dt_sec.median())
        if step_sec <= 0:
            step_sec = 900.0
    step_h = step_sec / 3600.0

    # [v13.9] 采样均匀性 + 时间断裂预检 (审计发现的 BUG #1 / #2 前置告警)
    # 主流场景 (03_train.py 之后的 15min 严格 resample 数据) 100% 通过, 无警告.
    # 边界场景 (原始不规则分路 CSV / 事件驱动采样 / NaN 密集) 会命中警告,
    # 提示用户: duration_min 与 energy_kwh 可能存在口径不一致 (最大 25%~100% 偏差).
    if len(dt_sec) >= 2:
        _dt_arr = dt_sec.values
        _mean_dt = float(_dt_arr.mean())
        if _mean_dt > 0:
            _cv = float(_dt_arr.std() / _mean_dt)  # 变异系数
            if _cv > 0.10:
                print(f"[WARN v13.9 采样均匀性] 采样步长非严格均匀 "
                      f"(CV={_cv:.3f} > 0.10, 中位数 {step_sec:.0f}s, "
                      f"最小 {_dt_arr.min():.0f}s, 最大 {_dt_arr.max():.0f}s). "
                      f"duration_min 用 t_end-t_start+step_median, "
                      f"energy_kwh 用 N×mean×step_median; 二者口径不一致时最大偏差可达 25%.")
        # 时间断裂检测: dt > 2 × step_median 视为断裂
        _gap_thr = 2.0 * step_sec
        _n_gaps = int((_dt_arr > _gap_thr).sum())
        if _n_gaps > 0:
            _max_gap = float(_dt_arr.max())
            print(f"[WARN v13.9 时间断裂] 检测到 {_n_gaps} 处相邻采样间隔 > "
                  f"{_gap_thr:.0f}s (最大 {_max_gap:.0f}s = {_max_gap/60:.1f}min). "
                  f"若断裂位于连续 ON 段中间, duration_min 会严重高估 "
                  f"(状态'连续'但物理断开); 建议先 resample 或按断裂手动切段.")

    # 二值化: >= 阈值 = ON
    state = (d[target_col].values >= on_thr_w).astype(int)

    # 找连续 ON 段的起止 index (半开半闭)
    on_segments = []  # (start_idx, end_idx_inclusive)
    i, n = 0, len(state)
    while i < n:
        if state[i] == 1:
            j = i
            while j + 1 < n and state[j + 1] == 1:
                j += 1
            on_segments.append((i, j))
            i = j + 1
        else:
            i += 1

    # [v13.11] 记录数据中所有出现的自然日 (用于后续找 "全天 OFF" 的日子)
    all_dates_in_data = sorted(set(pd.to_datetime(d["time"]).dt.date))
    # 收集所有落入 ON 段的日期 (拆日时会跨天, 需展开; 简单起见先记录起始日)
    on_day_set = set()

    rows = []
    for si, ej in on_segments:
        seg = d.iloc[si:ej + 1]
        t_start = seg["time"].iloc[0]
        t_end = seg["time"].iloc[-1]  # 该段最后一个采样点的时间戳
        w_vals = seg[target_col].values

        # 段统计 (以采样点为准, 而非拆日切片)
        # 时长 = (末点 - 首点) + 一个步长 (代表末点的持续区间)
        duration_sec = (t_end - t_start).total_seconds() + step_sec
        duration_min = round(duration_sec / 60.0, 2)
        mean_w = round(float(w_vals.mean()), 3)
        # [v13.16] 新增: 段内最小功率 (与 max 对称输出, 便于业务方判断"是否有短暂低谷"
        # 或"变频空调是否真的一直保持高档")
        min_w = round(float(w_vals.min()), 3)
        peak_w = round(float(w_vals.max()), 3)
        energy_kwh = round(float(w_vals.sum()) * step_h / 1000.0, 6)

        # 段末的展示 end_time (与用户示例对齐: 最后一个采样点的时间戳)
        seg_end_display = t_end

        if not split_by_day or t_start.normalize() == seg_end_display.normalize():
            rows.append({
                "being_time": _fmt_ts(t_start),
                "end_time": _fmt_ts(seg_end_display),
                target_col: 1,
                "duration_min": duration_min,
                "min_w": min_w,
                "mean_w": mean_w,
                "peak_w": peak_w,
                "energy_kwh": energy_kwh,
            })
            on_day_set.add(t_start.date())  # [v13.11] 记录该 ON 段占用的日期
        else:
            # 跨自然日: 按 00:00 拆分. 段级统计 (min_w/mean_w/peak_w) 保留全段值,
            # duration_min / energy_kwh 按拆片重算 (基于采样点分布).
            day_cursor = t_start.normalize()
            while day_cursor <= seg_end_display.normalize():
                day_start = day_cursor
                day_end = day_cursor + pd.Timedelta(days=1)
                # 本片采样点
                mask = (seg["time"] >= day_start) & (seg["time"] < day_end)
                sub = seg[mask]
                if not sub.empty:
                    piece_start = sub["time"].iloc[0]
                    piece_end = sub["time"].iloc[-1]
                    piece_w = sub[target_col].values
                    piece_dur_sec = (piece_end - piece_start).total_seconds() + step_sec
                    piece_dur_min = round(piece_dur_sec / 60.0, 2)
                    piece_energy = round(float(piece_w.sum()) * step_h / 1000.0, 6)
                    rows.append({
                        "being_time": _fmt_ts(piece_start),
                        "end_time": _fmt_ts(piece_end),
                        target_col: 1,
                        "duration_min": piece_dur_min,
                        "min_w": round(float(piece_w.min()), 3),   # [v13.16]
                        "mean_w": round(float(piece_w.mean()), 3),
                        "peak_w": round(float(piece_w.max()), 3),
                        "energy_kwh": piece_energy,
                    })
                    on_day_set.add(day_cursor.date())  # [v13.11] 记录拆片占用日期
                day_cursor = day_end

    # [v13.11] 追加"全天 OFF"日的行
    # 需求: 分路数据里出现但全天没有 ON 采样点的日子, 也要输出到 CSV, target_col=0
    # 其它列内容按"全天所有采样点"统计 (与 ON 段的"段内所有采样点"风格一致)
    off_day_rows = []
    for date in all_dates_in_data:
        if date in on_day_set:
            continue  # 该天有 ON 段, 已在上面 rows 中
        # 提取该天全部采样点
        day_start_ts = pd.Timestamp(date)
        day_end_ts = day_start_ts + pd.Timedelta(days=1)
        day_mask = (d["time"] >= day_start_ts) & (d["time"] < day_end_ts)
        day_seg = d[day_mask]
        if len(day_seg) == 0:
            continue  # 日期在集合里但无采样点 (理论上不应发生), 跳过
        w = day_seg[target_col].values
        t_first = day_seg["time"].iloc[0]
        t_last = day_seg["time"].iloc[-1]
        # 时长: 全天覆盖的采样时长 (与 ON 段公式对齐: (t_last - t_first) + step_sec)
        off_dur_sec = (t_last - t_first).total_seconds() + step_sec
        off_dur_min = round(off_dur_sec / 60.0, 2)
        off_min = round(float(w.min()), 3)     # [v13.16] 全天最小 (待机功率下限)
        off_mean = round(float(w.mean()), 3)
        off_peak = round(float(w.max()), 3)
        off_energy = round(float(w.sum()) * step_h / 1000.0, 6)
        off_day_rows.append({
            "being_time": _fmt_ts(t_first),
            "end_time": _fmt_ts(t_last),
            target_col: 0,          # ← 关键: OFF 天状态为 0
            "duration_min": off_dur_min,
            "min_w": off_min,       # [v13.16]
            "mean_w": off_mean,
            "peak_w": off_peak,
            "energy_kwh": off_energy,
        })
    rows.extend(off_day_rows)
    # 按 being_time 排序保持自然日顺序
    rows.sort(key=lambda r: pd.to_datetime(r["being_time"], format="%Y/%m/%d %H:%M:%S"))

    out = pd.DataFrame(rows, columns=[
        "being_time", "end_time", target_col,
        "duration_min", "min_w", "mean_w", "peak_w", "energy_kwh",   # [v13.16] min_w
    ])

    # [v13.10] 追加 dataset 列 (数据集归属)
    #   训练阶段: 从 bundle.train_dates/val_dates/test_dates 映射; 其余日期标 "未使用"
    #   推理阶段: 从 infer.include/exclude 计算的实际推理日集合映射; 其余标 "excluded"
    if date_labels is not None and len(out) > 0:
        # 从 being_time 提取日期 (格式 "YYYY/M/D H:MM:SS")
        def _date_from_being(t_str):
            try:
                # 提取 "YYYY/M/D" 部分, 归一化为 ISO
                d = pd.to_datetime(t_str.split(" ")[0]).strftime("%Y-%m-%d")
                return date_labels.get(d, "")
            except Exception:
                return ""
        out["dataset"] = out["being_time"].apply(_date_from_being)

    return out


def compute_daily_summary(periods: pd.DataFrame, target_col: str,
                          date_labels: dict = None,
                          bus_daily_counts: dict = None,
                          branch_daily_counts: dict = None) -> pd.DataFrame:
    """[v13.6] 从段级明细聚合为每日汇总.

    输入 periods 必须来自 compute_on_periods(split_by_day=True),
    确保没有段跨自然日 (否则 date 归属会混乱).

    输出列:
        date, n_segments, total_on_min, total_on_hours,
        first_on_time, last_off_time,
        [n_bus_raw, n_branch_raw],   ← v13.17 新增 (仅在传入 counts 时输出)
        min_w, mean_w, peak_w, energy_kwh
        [v13.16] min_w: ON 天=当日所有 ON 段 min_w 的最小值 (=开机期间最低瞬时功率);
                        OFF 天=全天最小 (=待机功率下限, 与 mean_w/peak_w 语义对齐)

    [v13.17] 新增两列, 与 metrics_utils.build_daily_metrics_rows 同名同义:
        bus_daily_counts:    {"yyyy-mm-dd": int} 当天总线 CSV 原始采样点数
                             (5min 采样满 288); None 时不输出该列, 向后兼容
        branch_daily_counts: {"yyyy-mm-dd": int} 当天分路 CSV 原始采样点数
                             (15min 采样满 96); None 时不输出该列
        缺失日期 = 0 (与 daily_metrics 语义一致)
    """
    # [v13.17] 只要传入任一 counts, 就输出两列 (统一开关, 避免半配置状态)
    _emit_raw = (bus_daily_counts is not None) or (branch_daily_counts is not None)

    def _base_cols():
        base = ["date", "n_segments", "total_on_min", "total_on_hours",
                "first_on_time", "last_off_time"]
        if _emit_raw:
            base += ["n_bus_raw", "n_branch_raw"]   # [v13.17]
        base += ["min_w", "mean_w", "peak_w", "energy_kwh"]   # [v13.16]
        return base

    if periods.empty:
        return pd.DataFrame(columns=_base_cols())

    d = periods.copy()
    # 解析 being_time / end_time 为 Timestamp
    d["_being"] = pd.to_datetime(d["being_time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    d["_end"]   = pd.to_datetime(d["end_time"],   format="%Y/%m/%d %H:%M:%S", errors="coerce")
    d["_date"]  = d["_being"].dt.date
    # [v13.11] 区分 ON 段行 (target_col=1) 与 OFF 天行 (target_col=0)
    # OFF 天在 daily 汇总时: n_segments=0, total_on_min=0, first/last=空, 统计列走全天值
    d["_is_on"] = d[target_col].astype(int) == 1

    rows = []
    # 段的 mean_w 按 duration_min 加权更公平; 但采样等间隔情形下与算术平均差别小
    for date, g in d.groupby("_date", sort=True):
        # [v13.11] 只用 ON 段计算 n_segments 和 total_on_min
        g_on = g[g["_is_on"]]
        n_segments = int(len(g_on))
        # [v13.16] min_w 支持: 兼容旧段级 CSV 无该列时用 NaN
        _has_min_col = "min_w" in g.columns
        if n_segments > 0:
            # 至少 1 个 ON 段
            total_min = float(g_on["duration_min"].sum())
            if total_min > 0:
                wmean = float((g_on["mean_w"] * g_on["duration_min"]).sum() / total_min)
            else:
                wmean = float(g_on["mean_w"].mean())
            first_on = g_on["_being"].min().strftime("%H:%M:%S")
            last_off = g_on["_end"].max().strftime("%H:%M:%S")
            peak_val = float(g_on["peak_w"].max())
            energy_val = float(g_on["energy_kwh"].sum())
            # [v13.16] 当日 min_w = 各 ON 段 min_w 的最小 (=开机期间最低瞬时功率)
            if _has_min_col:
                min_val = float(g_on["min_w"].min())
            else:
                min_val = float("nan")
        else:
            # [v13.11] 全天 OFF: 唯一一行 target_col=0, 统计走它的全天值
            g_off = g[~g["_is_on"]]
            total_min = 0.0
            # OFF 天的 min_w/mean_w/peak_w/energy_kwh 已按全天算, 直接取
            wmean = float(g_off["mean_w"].iloc[0]) if len(g_off) > 0 else 0.0
            peak_val = float(g_off["peak_w"].iloc[0]) if len(g_off) > 0 else 0.0
            energy_val = float(g_off["energy_kwh"].iloc[0]) if len(g_off) > 0 else 0.0
            # [v13.16] OFF 天 min_w 就是段行 min_w (全天最低=待机功率下限)
            if _has_min_col and len(g_off) > 0:
                min_val = float(g_off["min_w"].iloc[0])
            else:
                min_val = float("nan")
            first_on = ""  # 全天无 ON, first_on 留空
            last_off = ""
        _date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        row = {
            "date": _date_str,
            "n_segments": n_segments,
            "total_on_min": round(total_min, 2),
            "total_on_hours": round(total_min / 60.0, 3),
            "first_on_time": first_on,
            "last_off_time": last_off,
        }
        # [v13.17] 原始采集点数 2 列 (仅当传入 counts 才输出, 缺失日期=0)
        if _emit_raw:
            row["n_bus_raw"]    = int(bus_daily_counts.get(_date_str, 0)) \
                                  if bus_daily_counts    is not None else ""
            row["n_branch_raw"] = int(branch_daily_counts.get(_date_str, 0)) \
                                  if branch_daily_counts is not None else ""
        # [v13.16] min_w + 现有 mean/peak/energy
        row["min_w"]      = round(min_val, 3) if pd.notna(min_val) else ""
        row["mean_w"]     = round(wmean, 3)
        row["peak_w"]     = round(peak_val, 3)
        row["energy_kwh"] = round(energy_val, 6)
        # [v13.10] dataset 列
        if date_labels is not None:
            row["dataset"] = date_labels.get(_date_str, "")
        rows.append(row)

    cols = _base_cols()
    if date_labels is not None:
        cols.append("dataset")
    return pd.DataFrame(rows, columns=cols)


# ---------- CLI ----------

def _build_parser():
    p = argparse.ArgumentParser(
        description="分路数据开机时段分析 (根据 ON_THR_W 阈值)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 模式 A: 用户 ID
    p.add_argument("--user", type=str, default=None,
                   help="[模式A] 用户 folder_name (如 800080270708_4206602981958)")
    p.add_argument("--stage", type=str, default="train", choices=["train", "infer"],
                   help="[模式A] 阶段, 决定读 data/trains|infers 目录 (默认 train)")
    p.add_argument("--config", type=str, default=None,
                   help="[模式A] time_filter_config JSON 路径 (可选; 用于读 target_col/on_thr_w)")

    # 模式 B: 显式 CSV
    p.add_argument("--br-csv", type=str, default=None,
                   help="[模式B] 分路 CSV 路径 (与 --user 二选一)")
    p.add_argument("--bus-csv", type=str, default=None,
                   help="[v13.17] 总线 CSV 路径 (可选). 提供后 daily CSV 会追加 "
                        "n_bus_raw 列 (当天总线 5min 原始采样点数, 满 288). "
                        "n_branch_raw 列总是输出 (分路必存在)")

    # 通用覆盖 (可覆盖配置)
    p.add_argument("--target-col", type=str, default=None,
                   help="覆盖目标列名 (格式 pN, 优先级最高)")
    p.add_argument("--on-thr-w", type=float, default=None,
                   help="覆盖 ON 阈值 (W, 优先级最高)")
    p.add_argument("--no-split-by-day", action="store_true",
                   help="跨日 ON 段不拆分 (默认拆分). 注意: 关闭后 --daily-out 会失效.")
    p.add_argument("--out", type=str, default=None,
                   help="段级明细 CSV 路径 (默认 artifacts/<user>_<stage>_on_periods.csv)")
    p.add_argument("--daily-out", type=str, default=None,
                   help="[v13.6] 每日汇总 CSV 路径 (默认与 --out 同目录, 文件名 _daily 后缀). "
                        "传 'none' (字面量) 可禁用每日汇总输出.")
    return p


def main():
    args = _build_parser().parse_args()

    if not args.user and not args.br_csv:
        print("[FAIL] 必须提供 --user 或 --br-csv 之一", file=sys.stderr)
        sys.exit(2)
    if args.user and args.br_csv:
        print("[WARN] 同时提供 --user 和 --br-csv, 以 --br-csv 为准")

    # 加载配置 (可选)
    config = None
    if args.config:
        try:
            from time_filter_utils import load_time_filter_config
            config = load_time_filter_config(args.config)
            print(f"[OK] 已加载配置文件: {args.config} (含 {len(config)} 项 user 键)")
        except Exception as e:
            print(f"[WARN] 加载配置失败, 忽略: {e}")

    # 定位 CSV
    if args.br_csv:
        br_csv = Path(args.br_csv).expanduser().resolve()
        if not br_csv.exists():
            print(f"[FAIL] 分路 CSV 不存在: {br_csv}", file=sys.stderr)
            sys.exit(2)
        user_id = args.user  # 可能为 None
    else:
        user_id = args.user
        br_csv = _locate_br_csv(user_id, args.stage)
    print(f"[OK] 分路 CSV: {br_csv}")

    # 解析参数
    target_col = _resolve_target_col(br_csv, args.target_col, config, user_id)
    on_thr_w = _resolve_on_thr_w(args.on_thr_w, config, user_id)
    print(f"[OK] target_col = {target_col}")
    print(f"[OK] on_thr_w  = {on_thr_w:.2f} W")

    # 加载数据
    df = pd.read_csv(br_csv)
    print(f"[OK] 读取 {len(df)} 行, 列: {df.columns.tolist()}")

    # 计算 ON 段
    result = compute_on_periods(
        df, target_col, on_thr_w,
        split_by_day=(not args.no_split_by_day),
    )

    # 输出 - 段级明细
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        tag = f"{user_id or br_csv.stem}_{args.stage if args.user else 'custom'}"
        out_path = PROJECT_ROOT / "artifacts" / f"{tag}_on_periods.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 汇总
    n_seg = len(result)
    total_dur_h = float(result["duration_min"].sum()) / 60.0 if n_seg else 0.0
    total_energy = float(result["energy_kwh"].sum()) if n_seg else 0.0
    n_days = result["being_time"].str.split(" ").str[0].nunique() if n_seg else 0
    print("")
    print("=" * 60)
    print(f"[OK] 共 {n_seg} 段 ON, 覆盖 {n_days} 天, 累计 {total_dur_h:.2f} 小时, 用电 {total_energy:.3f} kWh")
    print(f"[OK] 段级明细 -> {out_path}")
    if n_seg > 0:
        print("\n前 5 段预览:")
        print(result.head(5).to_string(index=False))

    # [v13.6] 每日汇总
    daily_arg = (args.daily_out or "").strip().lower()
    if daily_arg == "none":
        print("\n[SKIP] --daily-out=none, 跳过每日汇总")
    elif args.no_split_by_day:
        print("\n[SKIP] --no-split-by-day 已启用, 跨日段未拆, 每日汇总失效, 跳过")
    else:
        if args.daily_out:
            daily_path = Path(args.daily_out).expanduser().resolve()
        else:
            daily_path = out_path.with_name(out_path.stem + "_daily" + out_path.suffix)

        # [v13.17] 计算 daily raw counts (与 pipeline 集成路径同源)
        _bus_counts = None
        _br_counts = None
        try:
            from metrics_utils import compute_raw_daily_counts
            _br_counts = compute_raw_daily_counts(str(br_csv), "time")
            if args.bus_csv:
                bus_p = Path(args.bus_csv).expanduser().resolve()
                if bus_p.exists():
                    _bus_counts = compute_raw_daily_counts(str(bus_p), "event_time")
                    print(f"[v13.17] daily raw counts: 总线 {len(_bus_counts)} 天 + "
                          f"分路 {len(_br_counts)} 天")
                else:
                    print(f"[v13.17 WARN] --bus-csv 路径不存在, n_bus_raw 列不输出")
            else:
                print(f"[v13.17] daily raw counts: 分路 {len(_br_counts)} 天 (未提供 --bus-csv)")
        except Exception as _e:
            print(f"[v13.17 WARN] compute_raw_daily_counts 失败: {_e}")

        daily_df = compute_daily_summary(result, target_col,
                                          bus_daily_counts=_bus_counts,
                                          branch_daily_counts=_br_counts)
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 每日汇总 -> {daily_path} ({len(daily_df)} 天)")
        if len(daily_df) > 0:
            print("\n每日汇总预览 (前 5 天):")
            print(daily_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
