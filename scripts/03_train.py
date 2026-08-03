# -*- coding: utf-8 -*-
"""
Step 3: 模型训练 v5 (温度特征 + 温度驱动季节路由)

- 阶段一: 开/关二分类  GradientBoostingClassifier (共享)
- 阶段二: 季节分层条件功率回归 (MoE)
    Summer / Transition / Winter Expert + Fallback 全局兜底
    每个 expert 含 P10/P50/P90 三个分位
- 基线:   单阶段     RandomForestRegressor

- 历代优化保留:
    [B] 阈值优化: F_beta=1.0 (v6.11 后处理优化, 旧 0.5) + 最小持续时长后处理
    [C] 时序特征 (60 维)
    [1] 样本逆密度加权
    [2] 分位回归 (Quantile Regression)
    [A] 季节分层 MoE
    [stratified] 按月分层时序切分

- v5 新增:
    [W] 12 维温度衍生特征 (Open-Meteo Archive API)
    [S] 温度驱动的季节路由 (取代按月硬路由)
    [DUAL] 同时训练 v4.2 (无温度) 与 v5 (含温度), 便于对照评估
"""
import json
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.preprocessing import StandardScaler

from common import (ARTIFACT_DIR, MODEL_DIR, PRED_DIR, METRIC_DIR,
                    MODEL_PKL, MODEL_V42_PKL,
                    ON_THR_W,             # v6.12.6 单一阈值 (训练 + 评估同口径)
                    ON_THR_TRAIN_W,       # 别名 = ON_THR_W (保留导入兼容)
                    ON_THR_BUSINESS_W,    # 别名 = ON_THR_W (保留导入兼容)
                    RANDOM_SEED,
                    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_CACHE_DIR,
                    SUMMER_TEMP_THRESHOLD, WINTER_TEMP_THRESHOLD,
                    USE_WEATHER_FEATURES, USE_TEMP_BASED_SEASON,
                    SPLIT_STRATEGY, SPLIT_RATIOS, validate_split_ratios,
                    PROJECT_VERSION,
                    BUS_CSV, BR_CSV, TARGET_COL,
                    D87_ADAPTIVE_GUARD_ENABLED,   # [v11] d87 守卫总开关
                    get_logger, Timer)


class _SkipD87Guard(Exception):
    """[v11] 内部信号: D87_ADAPTIVE_GUARD_ENABLED=False 时用它优雅跳过 d87 元数据 try 块"""
    pass
from feature_utils import (build_features, assert_no_nan_features,
                           parse_exclude_features_env)
from postprocess import search_best_threshold, apply_postprocess
from sample_weight_utils import (compute_inverse_density_weights,
                                 summarize_weights)
from expert_utils import (SeasonalRegressorBundle, assign_season,
                          diagnose_seasonal_distribution, SEASON_LABELS)
from split_utils import make_splits
from weather_utils import get_weather_for_period
from drift_features import (build_temp_power_lut,
                            export_temp_power_lut_csv)  # v13.15 CSV 导出
from residual_calibrator import ResidualCalibrator
from metrics_utils import (compute_classification_metrics,
                           compute_regression_metrics,
                           save_predictions_csv,
                           flatten_metrics_to_rows,
                           save_metrics_csv)

log = get_logger("train")

# ===== v2 后处理参数 (v6.11 优化后) =====
# 阈值优化目标 (FBETA) 调优记录:
#   FBETA = 0.5 -> 偏 Precision (旧默认): 阈值会被推高到 0.79+, 保 P=1.0 但 Recall 偏低
#       问题: F0.5 在 ON 段较多时过度保守, 边缘样本 (p_on=0.5~0.8) 被压成 FN
#   FBETA = 1.0 -> 平衡 P/R (新值): 阈值自动降到 0.5~0.7, FN 减少, 但 FP 可能增加
#       预期: Val FN 进一步降低, F1 略升 (因为 F1 本身就是 FBETA=1.0 的优化目标)
#
# 后处理调优记录:
#   POST_MIN_ON=2 + POST_FILL_SHORT_OFF=1  -> 旧默认 (v6.0~6.10), 过度抑制单步 ON
#       问题: 7/12 早晨 09:30 单步 ON (p_on=0.83 > 阈值) 被强制压回 OFF, 造成 6 步连续 FN
#   POST_MIN_ON=1 + POST_FILL_SHORT_OFF=3  -> 新优化 (v6.11 后处理优化)
#       策略: 不要求 ON 段连续 ≥2 步 (允许孤立 ON 通过) + 填充 ≤3 步的 OFF 间隙
#       预期: 修复事件 1 (7/12 早晨) 6 个 FN + Val FN 进一步降低
FBETA = 1.0                # 旧值 0.5 (偏 P); 改 1.0 平衡 P/R, 让阈值搜索倾向于减少 FN
POST_MIN_ON = 1            # 旧值 2; 改 1 让单步 ON 也保留
POST_FILL_SHORT_OFF = 3    # 旧值 1; 改 3 让"开-关-开-关"震荡序列被填回连续 ON

# ===== v3 参数 =====
USE_SAMPLE_WEIGHT = True
WEIGHT_N_BINS     = 20
QUANTILE_ALPHA    = 0.5
QUANTILE_LOW      = 0.10
QUANTILE_HIGH     = 0.90

    # ===== v4 参数 =====
USE_SEASONAL_MOE = True       # 是否启用季节分层 MoE
# 注: SPLIT_STRATEGY 和 SPLIT_RATIOS 已统一到 common.py 集中配置

# ===== v5 参数: 默认从 common.py 读取, 但 NILM_BASELINE_MODE=1 时强制关闭 =====
# 这样 03b_train_v42_baseline.py 通过环境变量切换为 v4.2 基线训练模式
import os as _os
if _os.environ.get("NILM_BASELINE_MODE") == "1":
    USE_WEATHER_FEATURES = False
    USE_TEMP_BASED_SEASON = False
    MODEL_PKL = MODEL_V42_PKL    # 输出到对照模型路径, 不覆盖主模型
    print("[INFO] NILM_BASELINE_MODE=1 检测到, 切换为 v4.2 基线训练模式")

# ===== v6 参数 =====
USE_DRIFT_FEATURES = True       # L1: 5 维漂移感知特征
USE_RESIDUAL_CALIB = True       # L4: 残差校正层 (在 val 集上学, 推理时应用)
if _os.environ.get("NILM_BASELINE_MODE") == "1":
    USE_DRIFT_FEATURES = False  # 基线模式不带漂移特征
    USE_RESIDUAL_CALIB = False  # 基线模式不带残差校正


def main():
    # ============================================================
    # [v13.5] 用户级 common 常量覆盖 (env vars 由 run_user_pipeline.py 注入)
    # ------------------------------------------------------------
    # 若配置文件里指定了对应字段, run_user_pipeline.py 会通过下面这些 env vars 注入:
    #   NILM_USER_ON_THR_W           -> 覆盖 ON_THR_W
    #   NILM_USER_SPLIT_RATIOS       -> 覆盖 SPLIT_RATIOS (JSON list str)
    #   NILM_USER_SPLIT_STRATEGY     -> 覆盖 SPLIT_STRATEGY
    #   NILM_USER_POST_MIN_ON        -> 覆盖 POST_MIN_ON
    #   NILM_USER_POST_FILL_SHORT_OFF-> 覆盖 POST_FILL_SHORT_OFF
    #   NILM_USER_WEATHER_LATITUDE   -> 覆盖 WEATHER_LATITUDE
    #   NILM_USER_WEATHER_LONGITUDE  -> 覆盖 WEATHER_LONGITUDE
    #   NILM_USER_USE_WEATHER_FEATURES  -> 覆盖 USE_WEATHER_FEATURES ("1"/"0")
    #   NILM_USER_USE_TEMP_BASED_SEASON -> 覆盖 USE_TEMP_BASED_SEASON
    # 未设置 env var 时使用文件顶部/common.py 默认.
    # 所有覆盖后的值会通过下面既有的 bundle 保存机制传给 04/05 保证训推一致.
    # ============================================================
    global ON_THR_W, ON_THR_TRAIN_W, ON_THR_BUSINESS_W
    global POST_MIN_ON, POST_FILL_SHORT_OFF
    global SPLIT_STRATEGY, SPLIT_RATIOS
    global WEATHER_LATITUDE, WEATHER_LONGITUDE
    global USE_WEATHER_FEATURES, USE_TEMP_BASED_SEASON

    def _env_or(name, default, conv):
        raw = _os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return conv(raw)
        except Exception as e:
            log.warning(f"  [v13.5] env {name}={raw!r} 解析失败 ({e}), 用默认 {default}")
            return default

    _override_log = []

    v = _env_or("NILM_USER_ON_THR_W", None, float)
    if v is not None and v != ON_THR_W:
        _override_log.append(f"ON_THR_W: {ON_THR_W} -> {v}")
        ON_THR_W = v
        ON_THR_TRAIN_W = v
        ON_THR_BUSINESS_W = v

    v = _env_or("NILM_USER_POST_MIN_ON", None, int)
    if v is not None and v != POST_MIN_ON:
        _override_log.append(f"POST_MIN_ON: {POST_MIN_ON} -> {v}")
        POST_MIN_ON = v

    v = _env_or("NILM_USER_POST_FILL_SHORT_OFF", None, int)
    if v is not None and v != POST_FILL_SHORT_OFF:
        _override_log.append(f"POST_FILL_SHORT_OFF: {POST_FILL_SHORT_OFF} -> {v}")
        POST_FILL_SHORT_OFF = v

    v = _env_or("NILM_USER_SPLIT_STRATEGY", None, str)
    if v is not None and v != SPLIT_STRATEGY:
        _override_log.append(f"SPLIT_STRATEGY: {SPLIT_STRATEGY} -> {v}")
        SPLIT_STRATEGY = v

    v = _env_or("NILM_USER_SPLIT_RATIOS", None,
                lambda s: tuple(__import__('json').loads(s)))
    if v is not None and tuple(v) != tuple(SPLIT_RATIOS):
        _override_log.append(f"SPLIT_RATIOS: {tuple(SPLIT_RATIOS)} -> {tuple(v)}")
        SPLIT_RATIOS = tuple(v)

    v = _env_or("NILM_USER_WEATHER_LATITUDE", None, float)
    if v is not None and v != WEATHER_LATITUDE:
        _override_log.append(f"WEATHER_LATITUDE: {WEATHER_LATITUDE} -> {v}")
        WEATHER_LATITUDE = v

    v = _env_or("NILM_USER_WEATHER_LONGITUDE", None, float)
    if v is not None and v != WEATHER_LONGITUDE:
        _override_log.append(f"WEATHER_LONGITUDE: {WEATHER_LONGITUDE} -> {v}")
        WEATHER_LONGITUDE = v

    v = _env_or("NILM_USER_USE_WEATHER_FEATURES", None,
                lambda s: s.strip().lower() in ("1", "true", "yes"))
    if v is not None and v != USE_WEATHER_FEATURES:
        _override_log.append(f"USE_WEATHER_FEATURES: {USE_WEATHER_FEATURES} -> {v}")
        USE_WEATHER_FEATURES = v

    v = _env_or("NILM_USER_USE_TEMP_BASED_SEASON", None,
                lambda s: s.strip().lower() in ("1", "true", "yes"))
    if v is not None and v != USE_TEMP_BASED_SEASON:
        _override_log.append(f"USE_TEMP_BASED_SEASON: {USE_TEMP_BASED_SEASON} -> {v}")
        USE_TEMP_BASED_SEASON = v

    if _override_log:
        for line in _override_log:
            log.info(f"  [v13.5 用户级覆盖] {line}")

    log.info("=" * 70)
    log.info("Step 3: 模型训练 v5 (温度特征 + 温度驱动季节路由)")
    log.info(f"  [B] 阈值优化: F_beta = {FBETA}, 后处理 min_on={POST_MIN_ON}")
    log.info(f"  [C] 时序特征 (60 维基础)")
    log.info(f"  [1] 样本逆密度加权: {USE_SAMPLE_WEIGHT}")
    log.info(f"  [2] 分位回归 P{int(QUANTILE_LOW*100)}/P50/P{int(QUANTILE_HIGH*100)}")
    log.info(f"  [A] 季节分层 MoE: {USE_SEASONAL_MOE}")
    log.info(f"  [W] 温度特征 (v5): {USE_WEATHER_FEATURES}   "
             f"位置=({WEATHER_LATITUDE}, {WEATHER_LONGITUDE})")
    log.info(f"  [S] 温度驱动季节路由 (v5): {USE_TEMP_BASED_SEASON}   "
             f"阈值 summer>={SUMMER_TEMP_THRESHOLD}°C / winter<={WINTER_TEMP_THRESHOLD}°C")
    log.info(f"  [D] 漂移感知特征 (v6 L1): {USE_DRIFT_FEATURES}  (5 维用户行为基线)")
    log.info(f"  [R] 残差校正层 (v6 L4): {USE_RESIDUAL_CALIB}  (val 集学习+ON 段加性校正)")
    log.info("=" * 70)

    # ---------- 1. 加载对齐数据 ----------
    aligned_csv = ARTIFACT_DIR / "aligned_15min.csv"
    with Timer(f"加载对齐数据 {aligned_csv.name}", log):
        df = pd.read_csv(aligned_csv, index_col=0, parse_dates=True,
                         encoding="utf-8-sig")
        log.info(f"  数据 shape={df.shape}")

    # ---------- [v6.12.6+v6.15.0-graceful-v5] 数据质量门 1: 对齐样本量 ----------
    # 触发条件: 对齐后样本数 < 96 (=1 天 15min 数据), 训练无意义
    # 退出码 11 -> run_user_pipeline.py 识别为软跳过, 不抛 RuntimeError
    MIN_ALIGNED_SAMPLES = 96
    if len(df) < MIN_ALIGNED_SAMPLES:
        log.warning("=" * 70)
        log.warning(f"[SKIP] 数据质量门 1 触发: 对齐样本数 {len(df)} < {MIN_ALIGNED_SAMPLES} (1天)")
        log.warning(f"[SKIP] 跳过原因: aligned_too_few")
        log.warning(f"[SKIP] 详情: aligned_n={len(df)}, 总线和分路的时间窗口可能错位")
        log.warning(f"[SKIP] 该用户的训练流水线已提前终止 (退出码 11)")
        log.warning("=" * 70)
        # 写入跳过原因文件供批量层收集
        skip_info = {"skip_reason": "aligned_too_few",
                     "detail": f"aligned_n={len(df)}",
                     "aligned_n": int(len(df))}
        (ARTIFACT_DIR / "skip_reason.json").write_text(
            json.dumps(skip_info, ensure_ascii=False), encoding="utf-8")
        sys.exit(11)

    feat_cols_all = [c for c in df.columns if c.startswith("load_iden_data")]

    # ---------- 2a. 拉取气象数据 (v5 新增) ----------
    weather_df = None
    if USE_WEATHER_FEATURES:
        with Timer("[v5] 拉取气象数据 (Open-Meteo, 自动缓存)", log):
            weather_df = get_weather_for_period(
                latitude=WEATHER_LATITUDE, longitude=WEATHER_LONGITUDE,
                start_ts=df.index.min(), end_ts=df.index.max(),
                cache_dir=WEATHER_CACHE_DIR, logger=log,
            )
            log.info(f"  气象数据 shape={weather_df.shape}, "
                     f"温度范围 [{weather_df['temperature_2m'].min():.1f}, "
                     f"{weather_df['temperature_2m'].max():.1f}] °C")

    # ---------- 2b. 特征工程 (v6: 含漂移感知) ----------
    with Timer("特征工程 v6 (相关性+差分+滚动+滞后+时间+温度+漂移)", log):
        corr = df[feat_cols_all].corrwith(df["y_ac"]).abs() \
                                .sort_values(ascending=False)
        top_cols = corr.head(25).index.tolist()
        log.info(f"  Top-25 相关性列, 例: {top_cols[:5]} ...")
        
        # v6: 构造温度-功率 LUT (训练阶段保存, 推理阶段复用)
        # v13.15: 同步导出 temp_power_lut.csv, 便于事后审计与漂移对比
        temp_power_lut = None
        temp_power_lut_meta = None
        if USE_DRIFT_FEATURES and weather_df is not None:
            temp_power_lut, temp_power_lut_meta = build_temp_power_lut(
                df, weather_df, top_cols, return_meta=True)
            n_bins = len([k for k in temp_power_lut if isinstance(k, tuple)])
            log.info(f"  [v6] 温度-功率 LUT 构造完成: {n_bins} 个桶, "
                     f"全局中位={temp_power_lut.get('__global_median__', 0):.1f}")
            # v13.15: 训练侧 CSV 导出
            _lut_csv_path = METRIC_DIR / "temp_power_lut.csv"
            export_temp_power_lut_csv(temp_power_lut, _lut_csv_path,
                                      meta=temp_power_lut_meta, logger=log)
        
        X_df = build_features(df, top_cols, weather_df=weather_df,
                              temp_power_lut=temp_power_lut)
        # [v14.3] 特征黑名单 (逐用户): 训练侧证据驱动的反"记忆捷径"开关.
        #   物理依据: 单变量日历特征 (如 dow) 在小样本冷却季可当"捷径"死记
        #   "周一=OFF" (U842 训练窗冷却季近 4 个周一全 OFF, 仅 ~6 个样本),
        #   无物理因果, 推理期泛化到 ON 的周一则整日崩塌.
        #   env NILM_EXCLUDE_FEATURES="dow,is_weekend" (逗号分隔) 生效.
        #   训推一致性: bundle["feat_names"] 自动剔除 -> 推理 reindex 同口径.
        _excl = parse_exclude_features_env(_os.environ.get("NILM_EXCLUDE_FEATURES", ""))
        if _excl:
            _hit = [c for c in _excl if c in X_df.columns]
            _miss = [c for c in _excl if c not in X_df.columns]
            if _hit:
                X_df = X_df.drop(columns=_hit)
            log.info(f"  [v14.3] NILM_EXCLUDE_FEATURES 生效: 剔除 {_hit}"
                     + (f" | 未命中(忽略): {_miss}" if _miss else ""))
        feat_names = X_df.columns.tolist()
        log.info(f"  最终特征维度: {len(feat_names)}")
        log.info(f"    - 原始电参量: 25")
        log.info(f"    - 一阶差分  : 10")
        log.info(f"    - 滚动统计  : 10 (Top-5 × mean/std, 窗=4)")
        log.info(f"    - 滞后项    :  6 (Top-3 × lag1/lag2)")
        log.info(f"    - 时间特征  :  9 (含 sin/cos 周期编码 + 季节)")
        if USE_WEATHER_FEATURES:
            log.info(f"    - 温度特征  : 12 (即时+统计+趋势+物理派生, v5)")
        if USE_DRIFT_FEATURES and temp_power_lut is not None:
            log.info(f"    - 漂移特征  :  5 (用户行为基线+温度残差, v6)")

    # [v13.7] NaN 硬检测: 训练特征若含 NaN, sklearn 会崩; 尽早暴露避免坏模型入 bundle
    assert_no_nan_features(X_df, stage_name="train", logger=log,
                           raise_on_nan=True)
    X = X_df.values.astype(np.float32)
    y = df["y_ac"].values.astype(np.float32)
    # v6.12.6 单口径: 训练标签用 ON_THR_W=10W (评估同口径)
    state = (y >= ON_THR_W).astype(int)
    on_pct = float(state.mean() * 100)
    log.info(f"  样本 {len(y)} | ON 占比 {on_pct:.2f}% "
             f"(单口径阈值 ON_THR_W={ON_THR_W} W)")

    # ---------- [v6.12.6+v6.15.0-graceful-v5] 数据质量门 2: 单类标签 ----------
    # 触发条件: ON 占比 = 0% (全 OFF) 或 100% (全 ON), GradientBoostingClassifier 拒绝单类
    # 退出码 12 -> 软跳过
    n_on  = int(state.sum())
    n_off = int(len(state) - n_on)
    if n_on == 0 or n_off == 0:
        kind = "all_off" if n_on == 0 else "all_on"
        log.warning("=" * 70)
        log.warning(f"[SKIP] 数据质量门 2 触发: ON 占比 = {on_pct:.2f}% (单类标签)")
        log.warning(f"[SKIP] 跳过原因: single_class_label ({kind})")
        if kind == "all_off":
            log.warning(f"[SKIP] 详情: 全部 {len(state)} 样本均 < ON_THR_W={ON_THR_W}W, "
                        f"该用户训练期空调可能未开机")
        else:
            peak_w = float(y.max())
            log.warning(f"[SKIP] 详情: 全部 {len(state)} 样本均 >= ON_THR_W={ON_THR_W}W "
                        f"(峰值 {peak_w:.1f}W), 该分路可能不是空调而是持续低功耗设备")
        log.warning(f"[SKIP] 该用户的训练流水线已提前终止 (退出码 12)")
        log.warning("=" * 70)
        skip_info = {"skip_reason": "single_class_label",
                     "detail": f"on_pct={on_pct:.2f}%, n_on={n_on}, n_off={n_off}, kind={kind}",
                     "on_pct": on_pct, "n_on": n_on, "n_off": n_off,
                     "peak_w": float(y.max()), "kind": kind}
        (ARTIFACT_DIR / "skip_reason.json").write_text(
            json.dumps(skip_info, ensure_ascii=False), encoding="utf-8")
        sys.exit(12)

    # ---------- 2c. 计算季节标签 (v5: 温度驱动) ----------
    if USE_TEMP_BASED_SEASON and weather_df is not None:
        # 每个时间戳对应的"当日平均温度"
        daily_t = weather_df["temperature_2m"].resample("D").mean()
        ts_dates = pd.DatetimeIndex(df.index).normalize()
        daily_avg_temp = daily_t.reindex(ts_dates, method="ffill").values
        season_labels_all = assign_season(
            df.index,
            daily_avg_temp=daily_avg_temp,
            use_temperature=True,
            summer_th=SUMMER_TEMP_THRESHOLD,
            winter_th=WINTER_TEMP_THRESHOLD,
        )
        log.info(f"  [v5] 温度驱动季节路由已启用 "
                 f"(summer>={SUMMER_TEMP_THRESHOLD}°C / winter<={WINTER_TEMP_THRESHOLD}°C)")
    else:
        season_labels_all = assign_season(df.index, use_temperature=False)
        log.info(f"  [v4.2 兼容] 按月份硬路由")
    # 各季节计数
    for sea in SEASON_LABELS:
        n = int((season_labels_all == sea).sum())
        log.info(f"    全量样本中 {sea:<11}: {n:>5} 条 ({n/len(df)*100:.1f}%)")

    # ---------- 3. 数据集切分 ----------
    # 数据集切分 (从 common.py 集中读取配置)
    split_ratios = validate_split_ratios(SPLIT_RATIOS)
    log.info(f"切分策略: {SPLIT_STRATEGY}, "
             f"比例: train={split_ratios[0]:.0%} / "
             f"val={split_ratios[1]:.0%} / "
             f"test={split_ratios[2]:.0%}")
    sp = make_splits(df.index, strategy=SPLIT_STRATEGY, ratios=split_ratios)
    idx_tr, idx_va, idx_te = sp["train"], sp["val"], sp["test"]
    # v6.10: stratified_day 会附带切分元数据 (完整天数/碎片天数/seed)
    if "_meta" in sp:
        meta = sp["_meta"]
        log.info(f"  [v6.10] 切分元数据: 完整天={meta['n_full_days']} "
                 f"碎片天={meta['n_partial_days']} (阈值={meta['full_day_threshold']} 条/天)  "
                 f"seed={meta['seed']}")

    # [v13 per-split time_filter] 应用 train/val/test 独立 include/exclude
    # 环境变量 NILM_SPLITS_FILTER_SPEC 由 run_user_pipeline.py 从批量配置注入
    _splits_filter_spec_str = _os.environ.get("NILM_SPLITS_FILTER_SPEC", "").strip()
    if _splits_filter_spec_str:
        try:
            from time_filter_utils import cli_arg_to_splits_spec, apply_per_split_filter, splits_spec_summary
            _splits_spec = cli_arg_to_splits_spec(_splits_filter_spec_str)
            if _splits_spec is not None:
                log.info(f"  [v13 per-split filter] 规格: {splits_spec_summary(_splits_spec)}")
                idx_tr, idx_va, idx_te = apply_per_split_filter(
                    df.index, idx_tr, idx_va, idx_te,
                    _splits_spec, logger=log
                )
        except Exception as _e:
            log.warning(f"  [v13 per-split filter] 应用失败 ({_e}), 使用原切分")
    log.info(f"数据集切分结果 (策略={SPLIT_STRATEGY}):")
    # [v13.8] 收集 train/val/test 实际使用的自然日集合, 用于 05 推理时检测数据泄漏
    # 存 ISO 日期字符串 (yyyy-mm-dd), 便于 JSON/joblib 序列化和跨环境比对
    _split_dates = {}
    for name, idx in zip(["train", "val", "test"], [idx_tr, idx_va, idx_te]):
        if len(idx) == 0:
            log.info(f"  {name:<5}: 空")
            _split_dates[name] = []
            continue
        ts = df.index[idx]
        _split_dates[name] = sorted({str(d) for d in pd.to_datetime(ts).normalize().date})
        log.info(f"  {name:<5}: {len(idx):4d} 条  "
                 f"({ts.min()} ~ {ts.max()})  "
                 f"ON 占比 {state[idx].mean()*100:.2f}%  "
                 f"[v13.8] 覆盖 {len(_split_dates[name])} 天")
        # 月份分布
        mc = pd.to_datetime(ts).to_period("M").value_counts().sort_index()
        log.info(f"        月份分布: " +
                 ", ".join([f"{str(k)}={v}" for k, v in mc.items()]))

    # ---------- [v6.12.6+v6.15.0-graceful-v5] 数据质量门 3: 切分后 val/test 为空 ----------
    # 触发条件: 对齐样本太少时 stratified_day 全塞 train -> val/test 空 -> scaler.transform 崩
    # 退出码 13 -> 软跳过
    if len(idx_va) == 0 or len(idx_te) == 0:
        log.warning("=" * 70)
        log.warning(f"[SKIP] 数据质量门 3 触发: 切分后验证集或测试集为空")
        log.warning(f"[SKIP] 跳过原因: split_empty_val_test")
        log.warning(f"[SKIP] 详情: train={len(idx_tr)} / val={len(idx_va)} / test={len(idx_te)}, "
                    f"对齐 {len(df)} 条 (~{len(df)/96:.1f} 天) 数据不足以做时序切分")
        log.warning(f"[SKIP] 该用户的训练流水线已提前终止 (退出码 13)")
        log.warning("=" * 70)
        skip_info = {"skip_reason": "split_empty_val_test",
                     "detail": (f"aligned_n={len(df)}, train_n={len(idx_tr)}, "
                                f"val_n={len(idx_va)}, test_n={len(idx_te)}"),
                     "aligned_n": int(len(df)),
                     "train_n": int(len(idx_tr)),
                     "val_n":   int(len(idx_va)),
                     "test_n":  int(len(idx_te))}
        (ARTIFACT_DIR / "skip_reason.json").write_text(
            json.dumps(skip_info, ensure_ascii=False), encoding="utf-8")
        sys.exit(13)

    X_tr, X_va = X[idx_tr], X[idx_va]
    y_tr, y_va = y[idx_tr], y[idx_va]
    s_tr, s_va = state[idx_tr], state[idx_va]
    t_tr, t_va = df.index[idx_tr], df.index[idx_va]
    # 季节标签 (v5 温度驱动 / v4.2 月份)
    sea_tr, sea_va = season_labels_all[idx_tr], season_labels_all[idx_va]

    # ---------- 4. 标准化 ----------
    with Timer("StandardScaler 拟合 (仅 train)", log):
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

    # ---------- 5. Stage-1 分类器 ----------
    log.info("-" * 70)
    log.info("[Stage-1] 训练 开/关分类器 GradientBoostingClassifier")
    log.info(f"  超参: n_estimators=300, max_depth=3, lr=0.05, subsample=0.8")
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED,
    )
    with Timer("Stage-1 训练", log):
        clf.fit(X_tr_s, s_tr)

    # ---------- 5.1 [B] 阈值优化 (F_beta + 后处理) ----------
    with Timer(f"[B] Stage-1 阈值搜索 (F{FBETA} + 后处理 min_on={POST_MIN_ON})", log):
        p_va = clf.predict_proba(X_va_s)[:, 1]
        result = search_best_threshold(
            p_va, s_va, beta=FBETA,
            min_on=POST_MIN_ON,
            fill_short_off=POST_FILL_SHORT_OFF,
        )
        best_thr = result["best_thr"]
        log.info(f"  最佳阈值 = {best_thr:.3f}   "
                 f"Val F{FBETA} = {result['best_fbeta']:.4f}")

    # 保存阈值-指标曲线
    curve_df = pd.DataFrame(result["curve"])
    curve_path = METRIC_DIR / "threshold_curve_val.csv"
    curve_df.to_csv(curve_path, index=False, encoding="utf-8-sig")
    log.info(f"  阈值曲线 -> {curve_path}  ({len(curve_df)} 行)")

    # Train / Val 预测 (含后处理)
    p_tr = clf.predict_proba(X_tr_s)[:, 1]
    pred_state_raw_tr = (p_tr >= best_thr).astype(int)
    pred_state_raw_va = (p_va >= best_thr).astype(int)
    # 后处理
    pred_state_tr, _ = apply_postprocess(pred_state_raw_tr,
                                         np.zeros_like(pred_state_raw_tr),
                                         POST_MIN_ON, POST_FILL_SHORT_OFF)
    pred_state_va, _ = apply_postprocess(pred_state_raw_va,
                                         np.zeros_like(pred_state_raw_va),
                                         POST_MIN_ON, POST_FILL_SHORT_OFF)
    log.info(f"  Train: raw_ON={pred_state_raw_tr.sum()}, "
             f"postproc_ON={pred_state_tr.sum()}, "
             f"真实_ON={s_tr.sum()}")
    log.info(f"  Val  : raw_ON={pred_state_raw_va.sum()}, "
             f"postproc_ON={pred_state_va.sum()}, "
             f"真实_ON={s_va.sum()}")

    # v6.12.6 单口径分类指标 (用训练标签 ON_THR_W=10W 评估)
    cls_tr = compute_classification_metrics(s_tr, pred_state_tr, p_tr)
    cls_va = compute_classification_metrics(s_va, pred_state_va, p_va)
    log.info(f"  Train cls: {cls_tr}")
    log.info(f"  Val   cls: {cls_va}")

    # ---------- 6. Stage-2 季节分层 MoE 条件功率回归 (v4) ----------
    log.info("-" * 70)
    log.info("[Stage-2] 训练 季节分层 MoE 条件功率回归器")
    mask_on = s_tr == 1
    n_on = int(mask_on.sum())
    log.info(f"  ON 训练样本数: {n_on}")

    # 训练前: 季节分布诊断 (v5 用温度驱动的 season 标签)
    log.info("  ---- 季节分布诊断 (Train ON 样本) ----")
    diag = diagnose_seasonal_distribution(sea_tr, y_tr, s_tr, logger=log)

    # [1] 样本权重 (整个 train 集计算, 后续按 mask 取子集)
    sw_full = None
    if USE_SAMPLE_WEIGHT and n_on > 0:
        with Timer("[1] 计算逆密度样本权重 (全局)", log):
            sw_on = compute_inverse_density_weights(
                y_tr[mask_on], n_bins=WEIGHT_N_BINS,
            )
            # 扩展回全 train 长度 (OFF 样本权重 0, 但本来就不参与回归训练)
            sw_full = np.zeros_like(y_tr, dtype=float)
            sw_full[mask_on] = sw_on
            stats = summarize_weights(y_tr[mask_on], sw_on, n_bins=5)
            log.info(f"  权重 范围 [{stats['weight_min']:.3f}, "
                     f"{stats['weight_max']:.3f}], "
                     f"mean={stats['weight_mean']:.3f}")

    # [2] GBR 工厂函数
    def make_quantile_reg(alpha):
        return GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED,
            loss="quantile", alpha=alpha,
        )

    # [A] 季节分层 MoE
    log.info("  ---- 训练季节专家 ----")
    moe = SeasonalRegressorBundle(
        gbr_factory=make_quantile_reg,
        quantiles=(QUANTILE_LOW, QUANTILE_ALPHA, QUANTILE_HIGH),
        logger=log,
    )
    with Timer("[A] 季节分层 MoE 训练 (3 expert × 3 quantile + 3 fallback)", log):
        # v5: 用 sea_tr (温度路由) 取代原来的 t_tr (月份路由)
        moe.fit(X_tr_s, y_tr, s_tr, sea_tr, sample_weight=sw_full)

    # 兼容旧 bundle 的别名 (供下游脚本无缝读取)
    reg      = moe   # 注意: 这里 reg 已是 MoE 对象, 推理时需传 timestamps
    # 保留单一 fallback 全局 GBR (向下兼容老推理脚本)
    reg_global_p50  = moe.fallback[QUANTILE_ALPHA]
    reg_global_low  = moe.fallback[QUANTILE_LOW]
    reg_global_high = moe.fallback[QUANTILE_HIGH]

    def predict_combined_moe(X_s, p_state, season_labels):
        raw_state = (p_state >= best_thr).astype(int)
        p_med  = np.clip(moe.predict(X_s, season_labels, alpha=QUANTILE_ALPHA), 0, None)
        p_low  = np.clip(moe.predict(X_s, season_labels, alpha=QUANTILE_LOW),  0, None)
        p_high = np.clip(moe.predict(X_s, season_labels, alpha=QUANTILE_HIGH), 0, None)
        state_filt, y_pred_filt = apply_postprocess(
            raw_state, p_med, POST_MIN_ON, POST_FILL_SHORT_OFF,
        )
        return y_pred_filt, state_filt, p_med, p_low, p_high

    y_pred_tr, _, p_med_tr, p_low_tr, p_high_tr = \
        predict_combined_moe(X_tr_s, p_tr, sea_tr)
    y_pred_va, _, p_med_va, p_low_va, p_high_va = \
        predict_combined_moe(X_va_s, p_va, sea_va)

    reg_tr = compute_regression_metrics(y_tr, y_pred_tr)
    reg_va = compute_regression_metrics(y_va, y_pred_va)
    log.info(f"  Train reg: {reg_tr}")
    log.info(f"  Val   reg: {reg_va}")

    # 残差诊断 (按季节细分, 关键!)
    def log_residual_bias_by_season(y_true, y_pred, s_true, sea, tag):
        m = s_true == 1
        if m.sum() == 0:
            return
        res_all = y_pred[m] - y_true[m]
        log.info(f"  [{tag}] ON 整体  残差: 中位={np.median(res_all):+.1f}W "
                 f"均值={res_all.mean():+.1f}W std={res_all.std():.1f}W "
                 f"(n={int(m.sum())})")
        for season in SEASON_LABELS:
            mm = m & (sea == season)
            if mm.sum() == 0:
                continue
            r = y_pred[mm] - y_true[mm]
            log.info(f"      + {season:<11} 残差: 中位={np.median(r):+.1f}W "
                     f"均值={r.mean():+.1f}W std={r.std():.1f}W "
                     f"(n={int(mm.sum())})")

    log_residual_bias_by_season(y_tr, y_pred_tr, s_tr, sea_tr, "Train")
    log_residual_bias_by_season(y_va, y_pred_va, s_va, sea_va, "Val  ")

    # ---------- 7. [L4] 训练残差校正层 (在 val 集上学) ----------
    residual_calib = None
    if USE_RESIDUAL_CALIB:
        log.info("-" * 70)
        log.info("[L4] 残差校正层训练 (在 val 集上学习, 不污染主模型)")

        # 计算 val 集的 recent_signal (Top-1 列近 24h 滚动均值)
        if top_cols and top_cols[0] in df.columns:
            top1 = df[top_cols[0]].astype(float)
            recent_24h = top1.rolling(window=96, min_periods=4,
                                      closed="left").mean().bfill().values
        else:
            recent_24h = np.zeros(len(df))

        recent_va = recent_24h[idx_va]
        # val 的 state_pred (用主模型推理结果)
        state_pred_va, _ = apply_postprocess(
            pred_state_raw_va, np.zeros_like(pred_state_raw_va),
            POST_MIN_ON, POST_FILL_SHORT_OFF,
        )

        with Timer("[L4] 训练 ResidualCalibrator (50 trees, depth=3)", log):
            residual_calib = ResidualCalibrator(n_estimators=50, max_depth=3,
                                                learning_rate=0.05)
            residual_calib.fit(
                y_pred_raw=y_pred_va, y_true=y_va,
                state_pred=state_pred_va,
                timestamps=t_va, weather_df=weather_df,
                recent_signal=recent_va, season_labels=sea_va,
                logger=log,
            )

        # 验证: val 集校正前后 MAE 对比
        y_tr_calib = y_pred_tr.copy()   # 默认无变化
        y_va_calib = y_pred_va.copy()
        if residual_calib._trained:
            # train 也需要近期信号 (与 val 同口径)
            recent_tr = recent_24h[idx_tr]
            state_pred_tr_filt, _ = apply_postprocess(
                pred_state_raw_tr, np.zeros_like(pred_state_raw_tr),
                POST_MIN_ON, POST_FILL_SHORT_OFF,
            )
            y_tr_calib = residual_calib.apply(
                y_pred_tr, t_tr, weather_df, recent_tr, sea_tr,
                state_pred=state_pred_tr_filt,
            )
            y_va_calib = residual_calib.apply(
                y_pred_va, t_va, weather_df, recent_va, sea_va,
                state_pred=state_pred_va,
            )
            mae_before = np.abs(y_pred_va - y_va).mean()
            mae_after  = np.abs(y_va_calib - y_va).mean()
            log.info(f"  [L4] Val MAE: 校正前 {mae_before:.2f}W -> "
                     f"校正后 {mae_after:.2f}W  (变化 {mae_after-mae_before:+.2f}W)")

    # ---------- 6c. [新增] Fallback 全局回归器指标计算 ----------
    # MoE 内的 fallback 是无季节路由的全局 GBR, 作为对照基线
    log.info("-" * 70)
    log.info("[Fallback 全局回归器] 计算指标 (无季节路由对照)")
    p_fb_tr = np.clip(reg_global_p50.predict(X_tr_s), 0, None)
    p_fb_va = np.clip(reg_global_p50.predict(X_va_s), 0, None)
    state_pred_tr_filt_fb, y_pred_fb_tr = apply_postprocess(
        pred_state_raw_tr, p_fb_tr, POST_MIN_ON, POST_FILL_SHORT_OFF,
    )
    state_pred_va_filt_fb, y_pred_fb_va = apply_postprocess(
        pred_state_raw_va, p_fb_va, POST_MIN_ON, POST_FILL_SHORT_OFF,
    )
    cls_fb_tr = compute_classification_metrics(s_tr, state_pred_tr_filt_fb, p_tr)
    cls_fb_va = compute_classification_metrics(s_va, state_pred_va_filt_fb, p_va)
    reg_fb_tr = compute_regression_metrics(y_tr, y_pred_fb_tr)
    reg_fb_va = compute_regression_metrics(y_va, y_pred_fb_va)
    log.info(f"  Train cls: F1={cls_fb_tr['F1']:.4f}, "
             f"reg: MAE={reg_fb_tr['MAE_W']:.2f}W SAE={reg_fb_tr['SAE']*100:.2f}%")
    log.info(f"  Val   cls: F1={cls_fb_va['F1']:.4f}, "
             f"reg: MAE={reg_fb_va['MAE_W']:.2f}W SAE={reg_fb_va['SAE']*100:.2f}%")

    # ---------- 6d. [新增] L4 校正后主模型指标 ----------
    cls_calib_tr = cls_calib_va = None
    reg_calib_tr = reg_calib_va = None
    if residual_calib is not None and residual_calib._trained:
        log.info("-" * 70)
        log.info("[main_L4_calib] 计算 L4 校正后主模型指标")
        # 分类指标与主模型一致 (L4 只校正功率, 不改变状态)
        cls_calib_tr = cls_tr.copy()
        cls_calib_va = cls_va.copy()
        reg_calib_tr = compute_regression_metrics(y_tr, y_tr_calib)
        reg_calib_va = compute_regression_metrics(y_va, y_va_calib)
        log.info(f"  Train reg (校正后): MAE={reg_calib_tr['MAE_W']:.2f}W "
                 f"SAE={reg_calib_tr['SAE']*100:.2f}%")
        log.info(f"  Val   reg (校正后): MAE={reg_calib_va['MAE_W']:.2f}W "
                 f"SAE={reg_calib_va['SAE']*100:.2f}%")

    # 保存 expert 训练摘要
    expert_summary_df = pd.DataFrame(moe.expert_summary())
    es_path = METRIC_DIR / "expert_summary.csv"
    expert_summary_df.to_csv(es_path, index=False, encoding="utf-8-sig")
    log.info(f"  ✓ expert 训练摘要 -> {es_path}")

    # ---------- 7. 基线: 单阶段 RF ----------
    log.info("-" * 70)
    log.info("[Baseline] 训练 RandomForestRegressor (单阶段)")
    rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                               random_state=RANDOM_SEED, n_jobs=-1)
    with Timer("RF 训练", log):
        rf.fit(X_tr_s, y_tr)
    y_rf_tr = np.clip(rf.predict(X_tr_s), 0, None)
    y_rf_va = np.clip(rf.predict(X_va_s), 0, None)
    # v6.12.6 RF baseline 用统一 ON_THR_W=10W (与训练标签同口径)
    s_rf_tr = (y_rf_tr >= ON_THR_W).astype(int)
    s_rf_va = (y_rf_va >= ON_THR_W).astype(int)
    cls_rf_tr = compute_classification_metrics(s_tr, s_rf_tr, y_rf_tr)
    cls_rf_va = compute_classification_metrics(s_va, s_rf_va, y_rf_va)
    reg_rf_tr = compute_regression_metrics(y_tr, y_rf_tr)
    reg_rf_va = compute_regression_metrics(y_va, y_rf_va)
    log.info(f"  RF Train: cls F1={cls_rf_tr['F1']:.4f}, reg {reg_rf_tr}")
    log.info(f"  RF Val  : cls F1={cls_rf_va['F1']:.4f}, reg {reg_rf_va}")

    # ---------- 8. 保存模型 ----------
    log.info("-" * 70)
    log.info("保存模型")
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- v6.12.1: 计算 d87 启动签名守卫的自适应阈值元数据 ----------
    # 目的: 让推理侧 d87 守卫不再依赖硬编码阈值 100, 而是从训练集统计学习
    # 物理依据:
    #   启动 d87 = 单步负向尖峰 (842: -150~-224, 270848: -65~-106)
    #   OFF 段 d87 = 噪声范围 (P1=-32, P50=-16) - 不分用户
    #   合理阈值 = train_off_P1 (-32) * 安全系数 2 = -64
    # 推理时: 一日 (或一窗口) 内 d87.min() < 阈值 -> 认为有空调启动迹象, 放行
    d87_guard_meta = {"enabled": False}
    try:
        # [v13] 用户级守卫决策 (优先级最高) - 从环境变量读 (由 run_user_pipeline.py 注入):
        #   NILM_USER_GUARD_ENABLED = "1" / "0" / "" (空表示未指定)
        #   若显式指定, 覆盖 common.D87_ADAPTIVE_GUARD_ENABLED 全局开关
        _env_guard = _os.environ.get("NILM_USER_GUARD_ENABLED", "").strip()
        if _env_guard == "0":
            d87_guard_meta["disabled_by_switch"] = True
            d87_guard_meta["disabled_by_user_config"] = True
            log.info("  [v13] NILM_USER_GUARD_ENABLED=0 (用户级配置显式关闭), 跳过 d87 元数据")
            log.info("        (bundle['d87_guard_meta']['enabled']=False, 推理侧将不做步级压制)")
            raise _SkipD87Guard()
        elif _env_guard == "1":
            log.info("  [v13] NILM_USER_GUARD_ENABLED=1 (用户级配置显式开启), 覆盖全局开关继续计算 d87 元数据")
            # 继续正常流程 (不跳过)
        else:
            # 未指定用户级配置, 回退到全局开关 [v11] 语义
            if not D87_ADAPTIVE_GUARD_ENABLED:
                d87_guard_meta["disabled_by_switch"] = True
                log.info("  [v11] D87_ADAPTIVE_GUARD_ENABLED=False (全局开关关闭), 跳过 d87 自适应守卫元数据计算")
                log.info("        (bundle['d87_guard_meta']['enabled']=False, 推理侧将不做步级压制)")
                raise _SkipD87Guard()
        log.info("  [v6.12.1] 计算 d87 守卫自适应阈值元数据 ...")
        # 重新加载 5min 原始 bus 与 branch (训练时只有 15min 对齐数据)
        from feature_utils import load_bus_csv, load_branch_csv
        _bus_raw, _ = load_bus_csv(BUS_CSV)
        _br_raw     = load_branch_csv(BR_CSV)
        _bus_raw["ts"] = pd.to_datetime(_bus_raw["event_time"])
        _bus_raw = _bus_raw.set_index("ts").sort_index()
        _br_raw["ts"]  = pd.to_datetime(_br_raw["time"])
        _br_raw = _br_raw.set_index("ts").sort_index()

        if "load_iden_data87" not in _bus_raw.columns:
            log.warning("  [v6.12.1] 训练 bus 缺 load_iden_data87, 跳过 d87 元数据")
        else:
            # v6.12.6: 守卫标定用业务功率阈值 (50W) 定义启动事件
            # 物理依据: 待机负荷 (路由器/常开设备) 功率 <50W, 真实空调启动一定 >50W
            # 用 50W 排除常态小负荷, 让 train_on_p10_abs 更纯净, 守卫阈值标定更准
            # 注: 这里 50W 是守卫局部硬编码, 不影响模型训练标签 (ON_THR_W=10W) 和指标评估
            _D87_CALIB_ON_THR_W = 50.0
            _br_raw["on"] = (_br_raw[TARGET_COL] > _D87_CALIB_ON_THR_W).astype(int)
            off_d87_min = []
            for i in range(0, len(_br_raw) - 4):
                if (_br_raw[TARGET_COL].iloc[i:i+4] < 10).all():
                    ts = _br_raw.index[i+2]
                    win = _bus_raw.loc[ts-pd.Timedelta("15min"):ts+pd.Timedelta("15min")]
                    if len(win) > 0:
                        off_d87_min.append(float(win["load_iden_data87"].min()))
            off_arr = np.array(off_d87_min) if off_d87_min else np.array([0.0])

            # 同时记录 ON 启动事件 d87.min, 仅用于诊断
            _br_raw["prev_on"] = _br_raw["on"].shift(1).fillna(0).astype(int)
            on_events_idx = _br_raw[(_br_raw["on"]==1)&(_br_raw["prev_on"]==0)].index
            on_d87_min = []
            for ts in on_events_idx:
                win = _bus_raw.loc[ts-pd.Timedelta("20min"):ts+pd.Timedelta("10min")]
                if len(win) > 0:
                    on_d87_min.append(float(win["load_iden_data87"].min()))
            on_arr = np.array(on_d87_min) if on_d87_min else np.array([0.0])

            # v6.12.4 关键统计 (用于阈值标定): 双向 |d87|.max (不分方向)
            # 因为 v6.12.3 已验证 4% 启动是正向尖峰
            off_d87_amax = []
            for i in range(0, len(_br_raw) - 4):
                if (_br_raw[TARGET_COL].iloc[i:i+4] < 10).all():
                    ts = _br_raw.index[i+2]
                    win = _bus_raw.loc[ts-pd.Timedelta("15min"):ts+pd.Timedelta("15min")]
                    if len(win) > 0:
                        off_d87_amax.append(float(max(
                            abs(win["load_iden_data87"].min()),
                            abs(win["load_iden_data87"].max())
                        )))
            off_amax_arr = np.array(off_d87_amax) if off_d87_amax else np.array([0.0])

            on_d87_amax = []
            for ts in on_events_idx:
                win = _bus_raw.loc[ts-pd.Timedelta("20min"):ts+pd.Timedelta("10min")]
                if len(win) > 0:
                    on_d87_amax.append(float(max(
                        abs(win["load_iden_data87"].min()),
                        abs(win["load_iden_data87"].max())
                    )))
            on_amax_arr = np.array(on_d87_amax) if on_d87_amax else np.array([0.0])

            # ============================================================
            # v6.12.4 阈值标定 (双源约束, 替代 v6.12.2 的单源 OFF×SF):
            # ============================================================
            #   ideal_thr = max(
            #       ON 启动 P10 × ALLOW_FACTOR (0.8),  # 让 90% 启动通过
            #       OFF 噪声 P99 × MARGIN_FACTOR (1.3),# 明显超过 OFF 噪声
            #   )
            # 设计动机:
            #   v6.12.2/3 用 OFF_P1 × 3.0 作为阈值, 在用户 OFF 噪声大时 (如用户2
            #   OFF_P1=-62, 阈值-186) 会推到比 ON 启动中位还严, 错压制启动事件 (5/27).
            #   双源约束让阈值兼顾两类不可分点: 弱启动 与 强噪声.
            # 物理依据 (89 个启动 + 4819 个 OFF 段验证):
            #   用户1: ON P10=75, OFF P99=50 -> 阈值 max(60, 65) = 65
            #   用户2: ON P10=129, OFF P99=69 -> 阈值 max(103, 90) = 103
            #          (旧 v6.12.3 用户2 阈值是 -185, 错压制 5/27 |d87|=166)
            # ============================================================
            # v6.12.5: ALLOW_FACTOR 从 0.8 -> 0.9
            # 实测 (用户2 OOD 14 天):
            #   AF=0.8 (阈值 96): FP=2 (5/30+6/02 边界噪声), Precision=78%
            #   AF=0.9 (阈值 109): FP=0, Precision=100%, Recall 仍 100%
            # 阈值 109 正好穿过 5/27 的 |d87|=166 (放行) 与 5/30 的 104 (压制) 之间
            # ============================================================
            # v6.15.0 自适应守卫阈值 (替代 v6.14.2 硬编码 ALLOW/MARGIN)
            # 设计三要素 (详见 common.py GUARD_* 配置):
            #   (A) 软最大平滑   -- gap 接近时不再硬翻转 max
            #   (B) 样本量自适应 -- n_on/n_off 越少, AF/MF 越保守
            #   (C) 元数据导出   -- 让推理侧 (05_inference.py) 做概率融合
            # ============================================================
            from common import (
                GUARD_AF_MIN, GUARD_AF_MAX, GUARD_AF_N_REF,
                GUARD_MF_MIN, GUARD_MF_MAX, GUARD_MF_N_REF,
                GUARD_SOFTMAX_TEMP_W, GUARD_SOFTMAX_PIVOT_W,
                GUARD_PROB_FUSION_ENABLED, GUARD_PROB_GAMMA, GUARD_PROB_MIN_RATIO,
            )

            n_on_amax  = int(len(on_amax_arr))
            n_off_amax = int(len(off_amax_arr))

            # --- (B) 样本量自适应 AF/MF ---
            # 物理意义: 样本量越少, 分位数 (P10/P99) 越易被极值污染,
            #          此时应取更保守的因子 (AF 更小=放行更多弱启动;
            #          MF 更小=不让 P99 噪声主导阈值)
            def _sqrt_interp(n, n_ref, vmin, vmax):
                ratio = max(0.0, min(1.0, n / max(1, n_ref)))
                return vmin + (vmax - vmin) * float(np.sqrt(ratio))

            ALLOW_FACTOR  = _sqrt_interp(n_on_amax,  GUARD_AF_N_REF, GUARD_AF_MIN, GUARD_AF_MAX)
            MARGIN_FACTOR = _sqrt_interp(n_off_amax, GUARD_MF_N_REF, GUARD_MF_MIN, GUARD_MF_MAX)

            on_p10_abs  = float(np.quantile(on_amax_arr,  0.10))
            off_p99_abs = float(np.quantile(off_amax_arr, 0.99)) if len(off_amax_arr) > 0 else 0.0
            on_constraint  = on_p10_abs  * ALLOW_FACTOR
            off_constraint = off_p99_abs * MARGIN_FACTOR

            # --- (A) 软最大平滑 (替代硬 max) ---
            # gap >> pivot -> weight≈1 -> 退化为 max (与旧行为一致, 安全)
            # gap << pivot -> weight≈0.5 -> 取均值 (避免 1W 差距决定生死)
            _gap = abs(on_constraint - off_constraint)
            _w = 1.0 / (1.0 + float(np.exp(-(_gap - GUARD_SOFTMAX_PIVOT_W) / GUARD_SOFTMAX_TEMP_W)))
            _hard_max = max(on_constraint, off_constraint)
            _mean_two = (on_constraint + off_constraint) / 2.0
            threshold_abs  = _w * _hard_max + (1.0 - _w) * _mean_two
            threshold_base = -threshold_abs   # 保留符号 (向下兼容旧代码假设)

            # 同时保留旧版统计 (兼容性 + 诊断)
            train_off_p1 = float(np.quantile(off_arr, 0.01))
            safety_factor_legacy = 3.0    # 仅记录, 不再用于决定 threshold

            # 训练用户的 d73_p95 (作为缩放锚点, 推理时按比例缩放)
            train_d73_p95 = float(_bus_raw["load_iden_data73"].quantile(0.95)) \
                            if "load_iden_data73" in _bus_raw.columns else None

            d87_guard_meta = {
                "enabled": True,
                "calibration": "v6.15.0_adaptive",       # v6.15: 自适应标定版本
                "threshold": threshold_base,             # 训练用户基准阈值
                "threshold_base": threshold_base,        # 同上
                "threshold_abs": threshold_abs,          # 绝对值阈值 (双向守卫直接用)
                # v6.12.4 双源约束 + v6.15 自适应
                "allow_factor": ALLOW_FACTOR,
                "margin_factor": MARGIN_FACTOR,
                "on_p10_abs": on_p10_abs,
                "off_p99_abs": off_p99_abs,
                "on_constraint": on_constraint,
                "off_constraint": off_constraint,
                "binding_constraint": ("ON" if on_constraint >= off_constraint else "OFF"),
                # v6.15 (A) 软最大平滑参数
                "softmax_weight": float(_w),             # 0~1, 1 为硬 max, 0.5 为均值
                "softmax_temp_w": float(GUARD_SOFTMAX_TEMP_W),
                "softmax_pivot_w": float(GUARD_SOFTMAX_PIVOT_W),
                # v6.15 (B) 样本量自适应参数
                "n_on_amax": n_on_amax,
                "n_off_amax": n_off_amax,
                "af_n_ref": GUARD_AF_N_REF,
                "mf_n_ref": GUARD_MF_N_REF,
                # v6.15 (C) 概率融合守卫 (推理侧使用)
                "prob_fusion_enabled": bool(GUARD_PROB_FUSION_ENABLED),
                "prob_gamma": float(GUARD_PROB_GAMMA),
                "prob_min_ratio": float(GUARD_PROB_MIN_RATIO),
                # v6.12.2 缩放支持
                "adaptive_scaling": True,
                "train_d73_p95": train_d73_p95,
                "scale_min": 0.05,
                "scale_max": 2.0,
                # v6.12.6 步级状态机守卫硬编码窗口 (12h, 典型空调一次运行 ≤ 12h)
                "max_on_hours": 12.0,
                # 历史字段 (兼容 + 诊断)
                "safety_factor": safety_factor_legacy,
                "train_off_n": int(len(off_arr)),
                "train_off_min": float(off_arr.min()),
                "train_off_p01": float(np.quantile(off_arr, 0.001)),
                "train_off_p1": train_off_p1,
                "train_off_p5": float(np.quantile(off_arr, 0.05)),
                "train_off_p50": float(np.quantile(off_arr, 0.50)),
                "train_off_amax_p99": off_p99_abs,
                "train_on_n": int(len(on_arr)),
                "train_on_p50": float(np.quantile(on_arr, 0.50)),
                "train_on_p90": float(np.quantile(on_arr, 0.90)),
                "train_on_amax_p10": on_p10_abs,
                "train_on_amax_p50": float(np.quantile(on_amax_arr, 0.50)),
            }
            log.info(f"  [v6.12.4] 训练集 OFF 段 |d87| (n={len(off_amax_arr)}): "
                     f"P50={np.quantile(off_amax_arr,0.5):.0f}, "
                     f"P95={np.quantile(off_amax_arr,0.95):.0f}, "
                     f"P99={off_p99_abs:.0f}")
            log.info(f"  [v6.12.4] 训练集 ON 启动 |d87| (n={len(on_amax_arr)}): "
                     f"P10={on_p10_abs:.0f}, P50={np.quantile(on_amax_arr,0.5):.0f}, "
                     f"P90={np.quantile(on_amax_arr,0.9):.0f}")
            log.info(f"  [v6.15.0] 双源约束 + 自适应:")
            log.info(f"  [v6.15.0]   样本量: n_on_amax={n_on_amax}, n_off_amax={n_off_amax}")
            log.info(f"  [v6.15.0]   自适应 AF={ALLOW_FACTOR:.3f} (范围 [{GUARD_AF_MIN}, {GUARD_AF_MAX}], n_ref={GUARD_AF_N_REF})")
            log.info(f"  [v6.15.0]   自适应 MF={MARGIN_FACTOR:.3f} (范围 [{GUARD_MF_MIN}, {GUARD_MF_MAX}], n_ref={GUARD_MF_N_REF})")
            log.info(f"  [v6.15.0]   ON 约束:  on_P10({on_p10_abs:.0f}) × {ALLOW_FACTOR:.3f} = {on_constraint:.1f}")
            log.info(f"  [v6.15.0]   OFF 约束: off_P99({off_p99_abs:.0f}) × {MARGIN_FACTOR:.3f} = {off_constraint:.1f}")
            log.info(f"  [v6.15.0]   gap={_gap:.1f}W -> 软最大权重={_w:.3f} (1=hard max, 0.5=均值)")
            log.info(f"  [v6.15.0]   绑定={d87_guard_meta['binding_constraint']} | "
                     f"|阈值| = {_w:.2f}×{_hard_max:.1f} + {1-_w:.2f}×{_mean_two:.1f} = {threshold_abs:.1f}")
            log.info(f"  [v6.15.0]   概率融合: enabled={GUARD_PROB_FUSION_ENABLED}, "
                     f"gamma={GUARD_PROB_GAMMA}, min_ratio={GUARD_PROB_MIN_RATIO}")
            log.info(f"  [v6.12.4] 训练用户 d73_p95 = {train_d73_p95:.0f}W (缩放锚点)")
            log.info(f"  [v6.12.4]   含义: 推理时按 (user_d73_p95 / {train_d73_p95:.0f}) "
                     f"× {threshold_abs:.0f} 自适应缩放")
    except _SkipD87Guard:
        # [v11] 开关关闭时的正常路径, 不打 warning
        pass
    except Exception as _e:
        log.warning(f"  [v6.12.4] d87 元数据计算失败 ({_e}), 守卫将退化到硬编码模式")

    # [v13] 兜底自动检测: 若用户配置未指定 guard_enabled 且守卫会导致严重误压,
    # 自动降级关闭守卫 (避免像 270708 case 那样大量真 ON 被守卫强制置零)
    #
    # 触发条件: 环境变量 NILM_USER_GUARD_ENABLED 未指定 且 d87_guard_meta.enabled=True
    #
    # 判据 (任一满足即触发降级):
    #   A. |d87|.max 绝对小 (< 50W)  -> 训练集 d87 特征本身弱
    #   B. 阈值覆盖率不足 (推理时会大幅误压):
    #      训练集 5min 原始 |d87| 中, 达到守卫阈值的天数占比 < 30%
    #      (意味着推理时大部分天没有触发点, 会被状态机全天强制 OFF)
    if (d87_guard_meta.get("enabled") is True and
        _os.environ.get("NILM_USER_GUARD_ENABLED", "").strip() == ""):

        _threshold_abs = d87_guard_meta.get("threshold_abs", 0)
        _off_max = d87_guard_meta.get("train_off_amax_p99", 0)
        _on_max  = d87_guard_meta.get("train_on_amax_p10", 0)
        _d87_effective_max = max(_off_max, _on_max, _threshold_abs)

        # 判据 A: 绝对值太小
        AUTO_DISABLE_THRESHOLD = 50.0
        trigger_A = _d87_effective_max < AUTO_DISABLE_THRESHOLD

        # 判据 B: 阈值天覆盖率不足 (基于 5min 原始 bus 逐日统计)
        # 用 _bus_raw 里的 d87 (前面已加载)
        trigger_B = False
        cover_ratio = None
        _n_days_total = 0
        _n_days_pass = 0
        try:
            # _bus_raw 在前面 d87 try 块里创建, 若守卫计算成功此变量存在
            if "load_iden_data87" in _bus_raw.columns:
                _d87_series = _bus_raw["load_iden_data87"].dropna()
                _daily_max = _d87_series.abs().resample("D").max()
                _n_days_total = len(_daily_max)
                _n_days_pass  = int((_daily_max >= _threshold_abs).sum())
                cover_ratio = _n_days_pass / _n_days_total if _n_days_total > 0 else 0
                COVER_MIN = 0.30
                trigger_B = cover_ratio < COVER_MIN
        except (NameError, Exception) as _e:
            log.debug(f"  [v13 auto_detect] 判据 B 计算失败: {_e}")

        if trigger_A or trigger_B:
            log.warning("=" * 70)
            log.warning(f"  [v13 auto_detect_guard] 训练侧检测到 d87 守卫会严重误压推理:")
            if trigger_A:
                log.warning(f"    判据 A: |d87|.max_effective={_d87_effective_max:.1f}W "
                            f"< 阈值 {AUTO_DISABLE_THRESHOLD}W")
            if trigger_B and cover_ratio is not None:
                log.warning(f"    判据 B: 训练集 {_n_days_total} 天中仅 {_n_days_pass} 天 "
                            f"|d87|.max ≥ 守卫阈值 {_threshold_abs:.1f}W "
                            f"(覆盖率 {cover_ratio*100:.1f}% < 30%)")
                log.warning(f"           -> 推理时约 {(1-cover_ratio)*100:.0f}% 的天会被守卫全天强制 OFF")
            log.warning(f"  [v13 auto_detect_guard] 自动关闭 d87 守卫 (避免推理灾难)")
            log.warning(f"  [v13 auto_detect_guard] 若需强制启用, "
                        f"在 time_filters.json 中该用户配置 guard_enabled=true")
            log.warning("=" * 70)
            d87_guard_meta = {
                "enabled": False,
                "disabled_by_auto_detect": True,
                "auto_detect_d87_max": float(_d87_effective_max),
                "auto_detect_threshold_abs": float(_threshold_abs),
                "auto_detect_cover_ratio": float(cover_ratio) if cover_ratio is not None else None,
                "auto_detect_trigger": ("A" if trigger_A else "") + ("B" if trigger_B else ""),
            }

    # 清除 MoE 中不可 pickle 的闭包函数 (gbr_factory) 和 logger
    moe.strip_for_save()

    # [v14.2] 训练侧先验统计 (供推理端可选标定/抑制; 统计只用训练窗数据, 零泄漏):
    #   - branch_temp_power_lut: 分路 ON 功率 × 温度桶分位 (修 P4/P5 推理低估)
    #   - hourly_on_prior:     逐小时 ON 率先验 (修 P3 边界 FP 毛刺)
    # 统计仅描述训练分布, 推理端是否启用由用户级 JSON 配置决定 (默认关, 零回归)
    branch_temp_power_lut = None
    hourly_on_prior = None
    try:
        from power_temp_calib import (build_branch_temp_power_lut,
                                      build_hourly_on_prior)
        # [v14.4] 解耦统计窗: env NILM_CALIB_STATS_SPEC (JSON {"include": [[a,b],...]})
        #   物理依据: 温桶 LUT / 时段先验只是训练侧分布统计量, 允许比分类器
        #   训练窗更宽 (仍严格 <评估起点, 零泄漏). 用途: 训练窗极短的用户
        #   (如 U0800 仅 6 天) LUT 温桶覆盖数 == 4, 推理期 43% ON 点无桶可抬;
        #   解耦后用全量 <评估起点 训练侧日期重建 -> 覆盖显著扩大.
        #   纪律: 窗口仍是配置显式声明的 include 区间, 日志审计实际覆盖日期.
        _calib_spec_str = _os.environ.get("NILM_CALIB_STATS_SPEC", "").strip()
        _stats_ts, _stats_y, _stats_wdf = None, None, None
        if _calib_spec_str:
            try:
                import json as _json_calib
                _cspec = _json_calib.loads(_calib_spec_str)
                _ranges = _cspec.get("include", [])
                from feature_utils import (load_bus_csv as _lbc,
                                           load_branch_csv as _lbr,
                                           resample_and_align as _rsa)
                _bus_c, _ = _lbc(BUS_CSV)
                _br_c = _lbr(BR_CSV, target_col=TARGET_COL
                             if isinstance(TARGET_COL, str) and "+" in TARGET_COL else None)
                _dfc = _rsa(_bus_c, _br_c, keep_cols=top_cols)
                if _ranges:
                    _m = pd.Series(False, index=_dfc.index)
                    for _a, _b2 in _ranges:
                        _m |= ((_dfc.index >= pd.Timestamp(_a)) &
                               (_dfc.index < pd.Timestamp(_b2) + pd.Timedelta(days=1)))
                    _dfc = _dfc[_m]
                if len(_dfc) > 0:
                    _stats_ts, _stats_y = _dfc.index, _dfc["y_ac"].values.astype(float)
                    if USE_WEATHER_FEATURES:
                        _stats_wdf = get_weather_for_period(
                            latitude=WEATHER_LATITUDE, longitude=WEATHER_LONGITUDE,
                            start_ts=_dfc.index.min(), end_ts=_dfc.index.max(),
                            cache_dir=WEATHER_CACHE_DIR, logger=log)
                    log.info(f"  [v14.4] 解耦统计窗生效: {len(_dfc)} 条 "
                             f"({_dfc.index.min().date()} ~ {_dfc.index.max().date()}), "
                             f"覆盖 {len(pd.unique(_dfc.index.normalize()))} 天")
                else:
                    log.warning(f"  [v14.4] 解耦统计窗过滤后为空, 回退训练窗统计")
            except Exception as _e_cs:
                log.warning(f"  [v14.4] 解耦统计窗构建失败 ({_e_cs}), 回退训练窗统计")
                _stats_ts, _stats_y, _stats_wdf = None, None, None
        if _stats_ts is None:
            _stats_ts = df.index
            _stats_y = df["y_ac"].values.astype(float)
            _stats_wdf = weather_df
        _s_true_all = (_stats_y >= ON_THR_W).astype(int)
        hourly_on_prior = build_hourly_on_prior(_stats_ts, _s_true_all)
        if USE_WEATHER_FEATURES and _stats_wdf is not None:
            branch_temp_power_lut = build_branch_temp_power_lut(
                _stats_ts, _stats_y, _stats_wdf,
                on_thr=ON_THR_W)
            log.info(f"  [v14.2] 功率温桶 LUT: "
                     f"{len(branch_temp_power_lut.get('bins', {}))} 桶 -> bundle")
    except Exception as _e_ptc:
        log.warning(f"  [v14.2] 训练侧先验统计失败 (不影响训练): {_e_ptc}")

    bundle = {
        "scaler": scaler, "clf": clf, "rf": rf,
        "moe": moe,
        "reg":      reg_global_p50,
        "reg_low":  reg_global_low,
        "reg_high": reg_global_high,
        "feat_cols": top_cols,
        "feat_names": feat_names,
        "best_thr": best_thr,
        "ON_THR": ON_THR_W,                       # 兼容旧推理代码 (= ON_THR_TRAIN_W)
        "ON_THR_TRAIN": ON_THR_TRAIN_W,           # v6.13: 训练标签阈值
        "ON_THR_BUSINESS": ON_THR_BUSINESS_W,     # v6.13: 业务评估阈值
        "trained_at": ts_tag,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
        # 版本与超参 (v6.10: version 字段从 common.PROJECT_VERSION 统一读取)
        "version": (f"v42_baseline_{PROJECT_VERSION}"
                    if _os.environ.get("NILM_BASELINE_MODE") == "1"
                    else f"{PROJECT_VERSION}_weather_aware_drift_defense"),
        "fbeta": FBETA,
        "post_min_on": POST_MIN_ON,
        "post_fill_short_off": POST_FILL_SHORT_OFF,
        "use_sample_weight": USE_SAMPLE_WEIGHT,
        "weight_n_bins": WEIGHT_N_BINS,
        "quantile_alpha": QUANTILE_ALPHA,
        "quantile_low":  QUANTILE_LOW,
        "quantile_high": QUANTILE_HIGH,
        "use_seasonal_moe": USE_SEASONAL_MOE,
        "expert_summary": moe.expert_summary(),
        "split_strategy": SPLIT_STRATEGY,
        "split_ratios":   list(split_ratios),   # v6.5: 保存到 bundle, 推理时复用
        # [v13.8] 三集实际使用的日期集合 (ISO yyyy-mm-dd 字符串列表, 已排序去重).
        # 用于 05_inference.py 检测"推理集 ∩ 训练日 > 0" 即数据泄漏, 并拆分指标.
        # 场景: 用户 infer 时段配置忘记排除训练区间, 导致模型对已见样本"过分自信",
        # 泄漏部分 F1/SAE 会异常好, 掩盖真实泛化能力 (270758 案例证据链完整).
        "train_dates":    _split_dates.get("train", []),
        "val_dates":      _split_dates.get("val",   []),
        "test_dates":     _split_dates.get("test",  []),
        # v5 新增
        "use_weather_features":  USE_WEATHER_FEATURES,
        "use_drift_features":    USE_DRIFT_FEATURES,
        "temp_power_lut":        temp_power_lut,   # v6 L1: 用于推理重建漂移特征
        "use_residual_calib":    USE_RESIDUAL_CALIB,
        "residual_calib":        residual_calib,   # v6 L4: 残差校正器
        # [v14.2] 推理端可选标定用训练侧统计 (None=该用户未启用/无气象)
        "branch_temp_power_lut": branch_temp_power_lut,
        "hourly_on_prior":       hourly_on_prior,
        "use_temp_based_season": USE_TEMP_BASED_SEASON,
        "weather_latitude":      WEATHER_LATITUDE,
        "weather_longitude":     WEATHER_LONGITUDE,
        "summer_temp_threshold": SUMMER_TEMP_THRESHOLD,
        "winter_temp_threshold": WINTER_TEMP_THRESHOLD,
        "n_features":            int(X.shape[1]),
        # v6.12.1: d87 守卫自适应阈值元数据
        "d87_guard_meta":        d87_guard_meta,
    }
    joblib.dump(bundle, MODEL_PKL)
    log.info(f"  ✓ 主模型 -> {MODEL_PKL}  ({MODEL_PKL.stat().st_size/1024:.1f} KB)")

    backup = MODEL_DIR / f"nilm_ac_two_stage_{ts_tag}.pkl"
    joblib.dump(bundle, backup)
    log.info(f"  ✓ 备份模型 -> {backup}")

    # v6.9 改进: 自动滚动清理, 仅保留最近 MAX_BACKUPS 份带时间戳的备份
    # (主/v42 各自一份, 避免每次训练后 models/ 目录膨胀)
    MAX_BACKUPS = 3
    bk_prefix = "nilm_ac_two_stage_"
    backups = sorted(
        [p for p in MODEL_DIR.glob(f"{bk_prefix}*.pkl")
         if p.name not in {"nilm_ac_two_stage.pkl", "nilm_ac_two_stage_v42.pkl"}],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if len(backups) > MAX_BACKUPS:
        for old in backups[MAX_BACKUPS:]:
            try:
                size_kb = old.stat().st_size / 1024
                old.unlink()
                log.info(f"  [清理] 删除旧备份 {old.name} ({size_kb:.1f} KB)")
            except Exception as e:
                log.warning(f"  [清理] 删除 {old.name} 失败: {e}")

    # 仅在主模型训练时拆分组件 + 写 meta JSON
    # (NILM_BASELINE_MODE=1 时 03b 调用此脚本, 仅出主 .pkl, 不污染主模型的组件文件)
    if _os.environ.get("NILM_BASELINE_MODE") != "1":
        joblib.dump(scaler,   MODEL_DIR / "scaler.pkl")
        joblib.dump(clf,      MODEL_DIR / "stage1_classifier.pkl")
        joblib.dump(moe,      MODEL_DIR / "stage2_moe_bundle.pkl")
        joblib.dump(reg_global_p50,  MODEL_DIR / "stage2_regressor.pkl")        # 全局 fallback
        joblib.dump(reg_global_low,  MODEL_DIR / "stage2_regressor_p10.pkl")
        joblib.dump(reg_global_high, MODEL_DIR / "stage2_regressor_p90.pkl")
        joblib.dump(rf,       MODEL_DIR / "baseline_rf.pkl")
        meta = {k: v for k, v in bundle.items()
                if k not in ("scaler", "clf", "rf",
                             "reg", "reg_low", "reg_high", "moe",
                             "residual_calib")}
        # v6: temp_power_lut 含 tuple 键, JSON 不支持, 单独序列化为字符串键再写
        if "temp_power_lut" in meta and meta["temp_power_lut"] is not None:
            lut_orig = meta["temp_power_lut"]
            meta["temp_power_lut"] = {
                (f"{k[0]:.2f}_{k[1]:.2f}" if isinstance(k, tuple) else str(k)): v
                for k, v in lut_orig.items()
            }
            meta["temp_power_lut_note"] = "tuple keys (lo, hi) 已序列化为 'lo_hi' 字符串"
        # residual_calib 是对象, JSON 不可表达, 仅记录是否存在
        meta["residual_calib_present"] = bool(bundle.get("residual_calib"))
        # [v14.6] 集成分类器形态显形: LightGBM 缺失时 EnsembleClf 静默退化为单
        #  GBDT (跨环境指标漂移实证根因, 2026-08-03), 写入 meta 便于事后排查
        meta["clf_class"] = type(clf).__name__
        meta["ensemble_lgb_active"] = bool(getattr(clf, "lgb", None) is not None)
        with open(MODEL_DIR / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"  ✓ 组件拆分 + meta JSON 已保存")
    else:
        log.info(f"  [基线模式] 跳过组件拆分和 meta 保存 (避免覆盖主模型组件)")

    # ---------- 9. 保存预测明细 ----------
    # v6.11 隔离修复: 基线模式 (NILM_BASELINE_MODE=1) 不覆盖主模型的预测明细
    # 否则 03b_v42 训练会覆盖 train_pred.csv/val_pred.csv, 让用户看到 v42 的结果
    # 而不是主模型 v6.x 的结果, 造成 FN 数等指标与训练日志报告不一致。
    log.info("-" * 70)
    log.info("保存预测明细")
    if _os.environ.get("NILM_BASELINE_MODE") != "1":
        save_predictions_csv(t_tr, y_tr, y_pred_tr, s_tr, pred_state_tr, p_tr,
                             y_pred_low=p_low_tr * pred_state_tr,
                             y_pred_high=p_high_tr * pred_state_tr,
                             out_path=PRED_DIR / "train_pred.csv")
        log.info(f"  ✓ {PRED_DIR / 'train_pred.csv'}")
        save_predictions_csv(t_va, y_va, y_pred_va, s_va, pred_state_va, p_va,
                             y_pred_low=p_low_va * pred_state_va,
                             y_pred_high=p_high_va * pred_state_va,
                             out_path=PRED_DIR / "val_pred.csv")
        log.info(f"  ✓ {PRED_DIR / 'val_pred.csv'}")
        save_predictions_csv(t_tr, y_tr, y_rf_tr,
                             out_path=PRED_DIR / "train_pred_rf.csv")
        save_predictions_csv(t_va, y_va, y_rf_va,
                             out_path=PRED_DIR / "val_pred_rf.csv")
        log.info(f"  ✓ RF 基线预测已保存")
    else:
        log.info(f"  [基线模式] 跳过 train_pred/val_pred 写入 (避免覆盖主模型预测)")

    # ---------- 10. 保存评估指标 ----------
    log.info("-" * 70)
    log.info("保存评估指标")
    extra = {"threshold": best_thr,
             "fbeta": FBETA,
             "post_min_on": POST_MIN_ON,
             "post_fill_short_off": POST_FILL_SHORT_OFF}
    # ---------- 完整保存所有模型的训练/验证指标 ----------
    # 命名统一规范 (与 04_evaluate / 05_inference 对齐, 便于 metrics_pivot 透视):
    #   main           : v6 主模型 (Stage-1 + Stage-2 MoE + 后处理, 无 L4 校正)
    #   main_L4_calib  : v6 主模型 + L4 残差校正 (v6 特有, 若启用)
    #   fallback       : MoE 全局兜底回归器 (无季节路由对照)
    #   rf             : 单阶段 RandomForest 基线
    #   v42_baseline   : v4.2 基线模型 (仅 NILM_BASELINE_MODE=1 时使用)
    is_baseline_mode = _os.environ.get("NILM_BASELINE_MODE") == "1"
    main_tag = "v42_baseline" if is_baseline_mode else "main"
    rows = []
    # 1. main 主模型 (train + val) - 单口径, 用训练标签 ON_THR_W=10W 评估
    rows += flatten_metrics_to_rows("train", main_tag,
                                    cls_metrics=cls_tr, reg_metrics=reg_tr,
                                    extra=extra)
    rows += flatten_metrics_to_rows("val", main_tag,
                                    cls_metrics=cls_va, reg_metrics=reg_va,
                                    extra=extra)
    # 2. main_L4_calib (若 v6 L4 启用)
    if cls_calib_tr is not None and reg_calib_tr is not None:
        rows += flatten_metrics_to_rows("train", "main_L4_calib",
                                        cls_metrics=cls_calib_tr,
                                        reg_metrics=reg_calib_tr,
                                        extra={**extra, "note": "L4 残差校正后"})
        rows += flatten_metrics_to_rows("val", "main_L4_calib",
                                        cls_metrics=cls_calib_va,
                                        reg_metrics=reg_calib_va,
                                        extra={**extra, "note": "L4 残差校正后"})
    # 3. fallback (MoE 全局回归器)
    rows += flatten_metrics_to_rows("train", "fallback",
                                    cls_metrics=cls_fb_tr,
                                    reg_metrics=reg_fb_tr,
                                    extra={**extra, "note": "MoE 全局兜底, 无季节路由"})
    rows += flatten_metrics_to_rows("val", "fallback",
                                    cls_metrics=cls_fb_va,
                                    reg_metrics=reg_fb_va,
                                    extra={**extra, "note": "MoE 全局兜底, 无季节路由"})
    # 4. rf (单阶段 RandomForest, 现在含分类指标)
    rows += flatten_metrics_to_rows("train", "rf",
                                    cls_metrics=cls_rf_tr,
                                    reg_metrics=reg_rf_tr,
                                    extra={"note": f"单阶段 RF, ON 阈值=ON_THR_W={ON_THR_W}W"})
    rows += flatten_metrics_to_rows("val", "rf",
                                    cls_metrics=cls_rf_va,
                                    reg_metrics=reg_rf_va,
                                    extra={"note": f"单阶段 RF, ON 阈值=ON_THR_W={ON_THR_W}W"})

    metric_path = METRIC_DIR / "train_val_metrics.csv"
    # v6.10: 统一使用 save_metrics_csv 的 append=True 机制
    #   - 历史记录完整保留 (timestamp + project_version + split + model 自然区分)
    #   - 基线模式 (NILM_BASELINE_MODE=1) 与主模式都追加, 不互相覆盖
    #   - 用户可通过 timestamp 追溯每次训练时间, 通过 project_version 追溯版本演进
    merged = save_metrics_csv(rows, metric_path, append=True)
    log.info(f"  ✓ {metric_path}  (本次追加 {len(rows)} 行, 文件累计 {len(merged)} 行)")

    # 概览: 各模型行数统计
    import collections
    cnt = collections.Counter((r["model"], r["split"], r["metric_type"]) for r in rows)
    log.info(f"  本次覆盖模型: {sorted(set(r['model'] for r in rows))}")
    log.info(f"  各模型×split×指标类型 行数:")
    for k, v in sorted(cnt.items()):
        log.info(f"    {k[0]:<16} {k[1]:<6} {k[2]:<15} {v} 项")

    log.info("=" * 70)
    log.info("Step 3 训练完成。下一步运行 04_evaluate.py。")


if __name__ == "__main__":
    main()
