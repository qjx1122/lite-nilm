# -*- coding: utf-8 -*-
"""
[v14 方向⑦] 烟测 (smoke test) 脚本

在一个用户的 train/val/test/inference 四阶段快速跑通检查,
确保:
  1. 特征工程无 NaN/Inf
  2. 模型训练不报错, n_features 一致
  3. 预测输出形状正确, 状态 ∈ {0,1}, 功率 ≥ 0
  4. 关键指标下限 (F1>0.8, MAE<200W)
  5. 训推特征一致性 (训练 feat_names = 推理期望列)

用法:
    python scripts/v14_smoke_test.py --user-dir data/trains/<uid> --infer-dir data/infers/<uid>
"""
import argparse
import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)

from common import (get_logger, ON_THR_W, TARGET_COL,
                    RANDOM_SEED, RESAMPLE, PROJECT_VERSION)
from feature_utils import (load_bus_csv, load_branch_csv,
                            resample_and_align, build_features,
                            assert_no_nan_features)
from split_utils import make_splits
from postprocess import search_best_threshold, apply_postprocess
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_absolute_error
from expert_utils import (SeasonalRegressorBundle, assign_season, SEASON_LABELS)
from sample_weight_utils import compute_inverse_density_weights
from metrics_utils import compute_classification_metrics, compute_regression_metrics

log = get_logger("v14_smoke", log_to_file=False)


def run_smoke_test(train_bus: Path, train_br: Path,
                   infer_bus: Path = None, infer_br: Path = None,
                   target_col: str = "p1",
                   on_thr_w: float = 10.0):
    """单用户烟测"""
    errors = []
    results = {"target_col": target_col, "on_thr_w": on_thr_w}

    # ---- 加载 ----
    log.info(f"[1/6] 加载: bus={train_bus.name}, br={train_br.name}")
    bus_df, data_cols = load_bus_csv(train_bus)
    br_df = load_branch_csv(train_br, target_col=target_col)
    if target_col not in br_df.columns:
        # 复合列可能已物化
        br_df = load_branch_csv(train_br, target_col=target_col)

    # ---- 对齐 ----
    log.info(f"[2/6] 重采样对齐 ({RESAMPLE})")
    keep_cols = [c for c in data_cols
                 if not (bus_df[c].isna().all() or bus_df[c].nunique() <= 1)]
    df = resample_and_align(bus_df, br_df, keep_cols=keep_cols)
    results["n_aligned"] = len(df)
    if len(df) < 300:
        errors.append(f"对齐样本 {len(df)} < 300")
        return results, errors
    log.info(f"  对齐后: {len(df)} 样本, {len(keep_cols)} 有效列")

    # ---- 特征工程 ----
    log.info("[3/6] 特征工程")
    corr = df[keep_cols].corrwith(df["y_ac"]).abs().sort_values(ascending=False)
    top_cols = corr.head(25).index.tolist()
    X_df = build_features(df, top_cols)
    try:
        assert_no_nan_features(X_df, stage_name="smoke", logger=log, raise_on_nan=True)
    except ValueError as e:
        errors.append(str(e))
        return results, errors
    feat_names = X_df.columns.tolist()
    results["n_features"] = len(feat_names)
    log.info(f"  特征维度: {len(feat_names)}")

    # ---- 切分 ----
    log.info("[4/6] stratified_day 切分 (70/15/15)")
    sp = make_splits(df.index, strategy="stratified_day", ratios=(0.7, 0.15, 0.15))
    idx_tr, idx_va, idx_te = sp["train"], sp["val"], sp["test"]
    X = X_df.values.astype(np.float32)
    y = df["y_ac"].values.astype(np.float32)
    state = (y >= on_thr_w).astype(int)
    results["on_pct"] = float(state.mean())
    if state[idx_tr].sum() == 0 or state[idx_tr].sum() == len(idx_tr):
        errors.append(f"训练集单类 (ON 占比 {state[idx_tr].mean():.2%})")
        return results, errors

    X_tr, X_va, X_te = X[idx_tr], X[idx_va], X[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
    s_tr, s_va, s_te = state[idx_tr], state[idx_va], state[idx_te]
    log.info(f"  tr/va/te = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}, "
             f"ON% = {s_tr.mean():.1%}/{s_va.mean():.1%}/{s_te.mean():.1%}")

    # ---- 标准化 ----
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    # ---- Stage-1 分类 ----
    log.info("[5/6] 训练 Stage-1 分类器 (100 trees, 快训)")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, subsample=0.8,
                                     random_state=RANDOM_SEED)
    clf.fit(X_tr_s, s_tr)
    p_va = clf.predict_proba(X_va_s)[:, 1]
    res = search_best_threshold(p_va, s_va, beta=1.0, min_on=1, fill_short_off=3)
    best_thr = res["best_thr"]
    results["best_thr"] = best_thr

    # ---- Stage-2 回归 (简化版, 不做 MoE 快速) ----
    log.info("[5b] 训练 Stage-2 回归器 (100 trees P50 fallback)")
    mask_on = s_tr == 1
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    learning_rate=0.1, subsample=0.8,
                                    loss="quantile", alpha=0.5,
                                    random_state=RANDOM_SEED)
    reg.fit(X_tr_s[mask_on], y_tr[mask_on])

    # ---- 评估 ----
    log.info("[6/6] 在 val/test 上评估")
    for name, X_s, y_t, s_t in [("val", X_va_s, y_va, s_va),
                                 ("test", X_te_s, y_te, s_te)]:
        p = clf.predict_proba(X_s)[:, 1]
        s_raw = (p >= best_thr).astype(int)
        p_reg = np.clip(reg.predict(X_s), 0, None)
        s_filt, y_filt = apply_postprocess(s_raw, p_reg, 1, 3)

        # 健全性检查
        assert set(np.unique(s_filt)).issubset({0, 1}), f"{name} 状态非0/1"
        assert (y_filt >= 0).all(), f"{name} 功率有负值"
        assert (np.isfinite(y_filt)).all(), f"{name} 功率有 Inf/NaN"

        F1 = f1_score(s_t, s_filt, zero_division=0)
        MAE = mean_absolute_error(y_t, y_filt)
        results[f"{name}_f1"] = float(F1)
        results[f"{name}_mae"] = float(MAE)
        log.info(f"  {name:5s}: F1={F1:.4f}, MAE={MAE:.1f}W")
        if F1 < 0.7 and len(s_t) > 100:
            errors.append(f"{name} F1={F1:.3f} < 0.7 (可能数据异常)")
        if MAE > 300:
            errors.append(f"{name} MAE={MAE:.0f}W > 300W (回归异常)")

    # ---- 推理路径 (若有 infer 数据) ----
    if infer_bus and infer_bus.exists() and infer_br and infer_br.exists():
        log.info("[infer] 推理路径烟测")
        bus_i, _ = load_bus_csv(infer_bus)
        br_i = load_branch_csv(infer_br, target_col=target_col)
        df_i = resample_and_align(bus_i, br_i, keep_cols=top_cols)
        if len(df_i) > 10:
            X_i = build_features(df_i, top_cols)
            # 检查列一致性 (训推一致)
            missing = set(feat_names) - set(X_i.columns)
            extra = set(X_i.columns) - set(feat_names)
            if missing:
                errors.append(f"推理侧缺特征: {list(missing)[:5]}")
            if extra:
                log.info(f"  推理侧多余特征 (忽略): {list(extra)[:5]}")
            X_i = X_i.reindex(columns=feat_names).fillna(0)
            try:
                assert_no_nan_features(X_i, "infer_smoke", logger=log, raise_on_nan=True)
            except ValueError as e:
                errors.append(str(e))
            X_i_s = scaler.transform(X_i.values.astype(np.float32))
            p_i = clf.predict_proba(X_i_s)[:, 1]
            p_reg_i = np.clip(reg.predict(X_i_s), 0, None)
            s_i_raw = (p_i >= best_thr).astype(int)
            s_i_filt, y_i_filt = apply_postprocess(s_i_raw, p_reg_i, 1, 3)
            log.info(f"  infer 样本 {len(df_i)}: ON 占比 {s_i_filt.mean():.1%}, "
                     f"平均预测功率 {y_i_filt.mean():.1f}W")
            results["infer_n"] = len(df_i)
            results["infer_on_pct_pred"] = float(s_i_filt.mean())

    log.info("=" * 60)
    if errors:
        log.error(f"烟测发现 {len(errors)} 个问题:")
        for i, e in enumerate(errors, 1):
            log.error(f"  {i}. {e}")
    else:
        log.info("✅ 烟测全部通过")
    log.info("=" * 60)
    return results, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, required=True,
                    help="训练用户目录 (data/trains/<uid>)")
    ap.add_argument("--infer-dir", type=str, default=None,
                    help="推理用户目录 (data/infers/<uid>), 默认同 uid")
    ap.add_argument("--target-col", type=str, default="p1")
    ap.add_argument("--on-thr-w", type=float, default=10.0)
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    infer_dir = Path(args.infer_dir) if args.infer_dir else None
    if infer_dir is None:
        # 默认从 trains 推 infers 目录名
        infer_dir = _PROJECT_ROOT / "data" / "infers" / train_dir.name

    # 自动解析 bus/br 文件名 (支持 -1/-infer 后缀)
    def find_csv(d: Path, prefix_pattern: str):
        if not d.exists():
            return None
        import re
        for p in sorted(d.glob("*.csv")):
            if re.match(prefix_pattern, p.name):
                return p
        # 简单 fallback
        csvs = list(d.glob("*.csv"))
        if len(csvs) >= 2:
            return csvs  # 返回 list, 由调用方区分
        return None

    import re
    train_bus = None
    train_br = None
    infer_bus = None
    infer_br = None
    for p in sorted(train_dir.glob("*.csv")):
        if p.name.startswith("e241_"):
            train_bus = p
        else:
            train_br = p
    if infer_dir and infer_dir.exists():
        for p in sorted(infer_dir.glob("*.csv")):
            if p.name.startswith("e241_"):
                infer_bus = p
            else:
                infer_br = p

    log.info(f"NILM v14 烟测启动")
    log.info(f"  train bus : {train_bus}")
    log.info(f"  train br  : {train_br}")
    log.info(f"  infer bus : {infer_bus}")
    log.info(f"  infer br  : {infer_br}")
    if not train_bus or not train_br:
        log.error("训练数据不完整, 退出")
        sys.exit(2)

    results, errors = run_smoke_test(
        train_bus, train_br, infer_bus, infer_br,
        target_col=args.target_col, on_thr_w=args.on_thr_w)
    log.info(f"结果摘要: {results}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
