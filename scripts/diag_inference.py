# -*- coding: utf-8 -*-
r"""
推理结果诊断脚本 - 定位 SAE 偏高的根因
用法:
    python scripts\diag_inference.py

会读取 artifacts\predictions\inference_result.csv 并输出:
    1. 时间跨度与季节分布
    2. 按月/按季节分组的 MAE/SAE
    3. 残差分布与系统性偏置
    4. 真实功率 vs 预测功率的相关性
    5. 漏报 (FN) 样本的功率分布
    6. 与训练分布的对比
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 复用项目模块
sys.path.insert(0, str(Path(__file__).parent))
from common import ARTIFACT_DIR, MODEL_DIR, ON_THR_W, ON_THR_BUSINESS_W, get_logger    # v6.13 解耦
from expert_utils import assign_season, SEASON_LABELS
import joblib

log = get_logger("diag", log_to_file=False)


def main():
    pred_csv = ARTIFACT_DIR / "predictions" / "inference_result.csv"
    if not pred_csv.exists():
        log.error(f"找不到 {pred_csv}, 请先运行 05_inference.py")
        return

    df = pd.read_csv(pred_csv, parse_dates=["time"], encoding="utf-8-sig")
    log.info("=" * 72)
    log.info(f"推理结果诊断: {pred_csv.name}, 共 {len(df)} 条")
    log.info("=" * 72)

    # ---------- 1. 时间跨度 ----------
    log.info("\n[1] 时间跨度与季节分布")
    log.info(f"  时间范围: {df['time'].min()} ~ {df['time'].max()}")
    log.info(f"  时长: {df['time'].max() - df['time'].min()}")

    # 月份分布
    log.info("\n  月份分布:")
    mc = df['time'].dt.to_period("M").value_counts().sort_index()
    for m, n in mc.items():
        log.info(f"    {m}: {n:>5} 条")

    # 季节分布 (用 v4 同款映射)
    seasons = assign_season(df['time'])
    log.info("\n  季节分布:")
    for sea in SEASON_LABELS:
        n = int((seasons == sea).sum())
        log.info(f"    {sea:<11}: {n:>5} 条  ({n/len(df)*100:.1f}%)")

    # ---------- 2. 整体指标 ----------
    if "y_true_W" not in df.columns or df["y_true_W"].isna().all():
        log.warning("无真值标签, 仅做预测分布诊断")
        log.info(f"\n  y_pred 统计: 均值={df['y_pred_W'].mean():.1f}W, "
                 f"中位={df['y_pred_W'].median():.1f}W, "
                 f"max={df['y_pred_W'].max():.1f}W, "
                 f">0 占比={(df['y_pred_W']>0).mean()*100:.1f}%")
        return

    y_t = df['y_true_W'].values
    y_p = df['y_pred_W'].values
    s_t = (y_t >= ON_THR_BUSINESS_W).astype(int)    # v6.13: 评估用 BUSINESS
    s_p = df['state_pred'].astype(int).values if 'state_pred' in df else (y_p > 0).astype(int)

    # ---------- 3. 按季节分组评估 ----------
    log.info("\n[2] 按季节分组的回归指标")
    log.info(f"  {'season':<12}{'n':>6}{'ON_n':>6}{'真实均值':>10}{'预测均值':>10}"
             f"{'MAE_W':>10}{'SAE':>10}{'kWh真/预':>20}")
    log.info("  " + "-" * 100)
    for sea in SEASON_LABELS + ["ALL"]:
        if sea == "ALL":
            m = np.ones(len(df), dtype=bool)
        else:
            m = (seasons == sea)
        if m.sum() == 0:
            continue
        on = m & (s_t == 1)
        if on.sum() == 0:
            log.info(f"  {sea:<12}{int(m.sum()):>6}{0:>6}  无 ON 样本")
            continue
        mae = np.abs(y_p[m] - y_t[m]).mean()
        sae = abs(y_p[m].sum() - y_t[m].sum()) / max(y_t[m].sum(), 1e-6)
        kwh_t = y_t[m].sum() / 4 / 1000
        kwh_p = y_p[m].sum() / 4 / 1000
        log.info(f"  {sea:<12}{int(m.sum()):>6}{int(on.sum()):>6}"
                 f"{y_t[on].mean():>9.1f}W{y_p[on].mean():>9.1f}W"
                 f"{mae:>9.1f}W{sae*100:>9.2f}%"
                 f"  {kwh_t:>7.2f}/{kwh_p:>7.2f} kWh")

    # ---------- 4. ON 残差按真实功率分桶 ----------
    log.info("\n[3] ON 样本残差按真实功率分桶 (验证是否系统性低估)")
    on = s_t == 1
    if on.any():
        bins = [0, 200, 400, 600, 800, 1000]
        log.info(f"  {'功率区间(W)':<15}{'n':>6}{'真实均值':>10}{'预测均值':>10}"
                 f"{'残差中位':>10}{'残差均值':>10}{'低估比例':>10}")
        log.info("  " + "-" * 80)
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mm = on & (y_t >= lo) & (y_t < hi)
            if mm.sum() == 0:
                continue
            res = y_p[mm] - y_t[mm]
            under = (res < 0).mean()
            log.info(f"  [{lo:>4}, {hi:>4})    {int(mm.sum()):>6}"
                     f"{y_t[mm].mean():>9.1f}W{y_p[mm].mean():>9.1f}W"
                     f"{np.median(res):>+9.1f}W{res.mean():>+9.1f}W"
                     f"{under*100:>9.1f}%")

    # ---------- 5. 漏报 FN 分析 ----------
    log.info("\n[4] 漏报 (FN) 样本分析")
    fn_mask = (s_t == 1) & (s_p == 0)
    log.info(f"  FN 数量: {int(fn_mask.sum())} / {int(on.sum())} "
             f"({fn_mask.sum()/max(on.sum(),1)*100:.1f}%)")
    if fn_mask.any():
        log.info(f"  FN 样本真实功率: 均值={y_t[fn_mask].mean():.1f}W, "
                 f"中位={np.median(y_t[fn_mask]):.1f}W, "
                 f"max={y_t[fn_mask].max():.1f}W, min={y_t[fn_mask].min():.1f}W")
        log.info(f"  FN 累计漏算能耗: {y_t[fn_mask].sum()/4/1000:.3f} kWh "
                 f"({y_t[fn_mask].sum()/y_t[on].sum()*100:.1f}% of 真实总能耗)")

    # ---------- 6. 与训练分布对比 ----------
    log.info("\n[5] 与训练集 expert 摘要对比")
    es_csv = ARTIFACT_DIR / "metrics" / "expert_summary.csv"
    if es_csv.exists():
        es = pd.read_csv(es_csv)
        log.info("  训练时各 expert 学到的功率分布:")
        for _, r in es.iterrows():
            if r['status'] == 'trained':
                log.info(f"    {r['season']:<11}: 训练 ON n={int(r['n_on'])}, "
                         f"y均值={r['y_mean']:.1f}W, 中位={r['y_median']:.1f}W, "
                         f"std={r['y_std']:.1f}W")
            else:
                log.info(f"    {r['season']:<11}: 未训练 (回退到全局 fallback)")

    log.info("\n" + "=" * 72)
    log.info("诊断完成")


if __name__ == "__main__":
    main()
