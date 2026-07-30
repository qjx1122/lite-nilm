# -*- coding: utf-8 -*-
"""
数据集切分工具 - 训练/评估共用, 保证 split 一致性

v6.10 新增 'stratified_day' 策略:
  - 以"天"为原子单位切分, 整天归到同一 split
  - 月内按天随机抽样 (固定 seed=42 保证可复现)
  - 完整天阈值 FULL_DAY_MIN_SAMPLES (默认 80, 15min 间隔 96 步/天的 ~83%)
  - 碎片天直接归 train (不进 val/test, 避免边界泄漏)
  - 彻底消除"同天被切到不同 split"导致的 lag/rolling 特征跨集泄漏
"""
import numpy as np
import pandas as pd


# v6.10: 完整天阈值 (一天 96 个 15min 步, 取 80 = 83.3%, 允许少量缺测)
FULL_DAY_MIN_SAMPLES = 80


def make_splits(timestamps, strategy: str = None,
                ratios=None, seed: int = 42):
    """
    返回 {train, val, test} 三个 index 数组 (按原始顺序排序)

    参数:
        timestamps: DatetimeIndex 或 list
        strategy:   "time"            纯时序 (前 70%/15%/15%)
                    "stratified"      季节分层时序, 每月内部按 ratios 时序切分
                                      ⚠️ 存在"同天被切到不同 split"的边界泄漏
                    "stratified_day"  [*] v6.10: 季节分层 + 按天随机抽样
                                      整天归到一个 split, 完整天 (≥80 条)
                                      参与切分, 碎片天直接归 train
                    None              读取 common.SPLIT_STRATEGY 默认值
        ratios:     (train, val, test) 三元组, 和必须 = 1.0
                    None 读取 common.SPLIT_RATIOS 默认值
        seed:       随机种子 (仅 stratified_day 用), 固定保证可复现

    返回:
        dict[str, np.ndarray]: {"train": idx, "val": idx, "test": idx}
    """
    # 默认值从 common 读取 (集中配置)
    if strategy is None or ratios is None:
        from common import SPLIT_STRATEGY, SPLIT_RATIOS, validate_split_ratios
        if strategy is None:
            strategy = SPLIT_STRATEGY
        if ratios is None:
            ratios = SPLIT_RATIOS
        ratios = validate_split_ratios(ratios)
    else:
        # 用户显式传入, 也校验
        from common import validate_split_ratios
        ratios = validate_split_ratios(ratios)

    n = len(timestamps)
    ts = pd.to_datetime(timestamps)

    if strategy == "time":
        # 纯时序: 前 ratios[0] 训练, 中 ratios[1] 验证, 后 ratios[2] 测试
        i_tr = int(n * ratios[0])
        i_va = int(n * (ratios[0] + ratios[1]))
        return {
            "train": np.arange(0, i_tr),
            "val":   np.arange(i_tr, i_va),
            "test":  np.arange(i_va, n),
        }

    elif strategy == "stratified":
        # 按月分层时序: 每月内部按比例切分, 再合并
        # 优势: 每个 split 都覆盖所有月份, 解决季节漂移
        # ⚠️ 已知缺陷 (v6.7~6.9): 切分点可能落在某天中间, 导致 6/8 边界
        #    出现"同天被切到不同 split", 让 lag/rolling 特征跨集泄漏。
        #    v6.10 推荐改用 "stratified_day" 彻底消除。
        months = ts.to_period("M")
        tr, va, te = [], [], []
        for m in sorted(months.unique()):
            mask = (months == m)
            idx_m = np.where(mask)[0]
            n_m = len(idx_m)
            # 月样本过少时全归 train (避免 val/test 出空集)
            if n_m < 10:
                tr.append(idx_m)
                continue
            i1 = int(n_m * ratios[0])
            i2 = int(n_m * (ratios[0] + ratios[1]))
            tr.append(idx_m[:i1])
            va.append(idx_m[i1:i2])
            te.append(idx_m[i2:])
        return {
            "train": np.sort(np.concatenate(tr)) if tr else np.array([], int),
            "val":   np.sort(np.concatenate(va)) if va else np.array([], int),
            "test":  np.sort(np.concatenate(te)) if te else np.array([], int),
        }

    elif strategy == "stratified_day":
        # v6.10: 按月分层 + 按天随机抽样 (整天归到一个 split)
        # 设计原则:
        #   1. 完整天 (样本数 ≥ FULL_DAY_MIN_SAMPLES=80) 才参与 val/test 抽样
        #   2. 碎片天 (样本数 < 80) 全部归 train, 不污染 val/test
        #   3. 月内按天随机抽样 (固定 seed=42, 可复现)
        #   4. 每月至少保证 val/test 各有 1 天 (若可用完整天 ≥ 3)
        from numpy.random import default_rng
        rng = default_rng(seed)

        dates = ts.normalize()              # 每条样本归属的"日" (00:00:00)
        months = ts.to_period("M")
        tr, va, te = [], [], []

        # 统计日志用计数
        n_full_days_total = 0
        n_partial_days_total = 0

        for m in sorted(months.unique()):
            mask = (months == m)
            idx_m = np.where(mask)[0]
            dates_m = dates[idx_m]

            # 统计本月每天的样本数
            day_counts = pd.Series(dates_m).value_counts().sort_index()
            full_days    = [d for d, c in day_counts.items()
                            if c >= FULL_DAY_MIN_SAMPLES]
            partial_days = [d for d, c in day_counts.items()
                            if c <  FULL_DAY_MIN_SAMPLES]
            n_full_days_total    += len(full_days)
            n_partial_days_total += len(partial_days)

            # 碎片天的样本全部归 train
            if partial_days:
                partial_set = set(partial_days)
                partial_idx = np.array(
                    [i for i in idx_m if dates[i] in partial_set],
                    dtype=int)
                if len(partial_idx) > 0:
                    tr.append(partial_idx)

            # 完整天才参与 val/test 抽样
            n_full = len(full_days)
            if n_full == 0:
                continue
            if n_full < 3:
                # 完整天不足 3 天, 无法保证 val/test 各 ≥ 1 天 -> 全归 train
                full_set = set(full_days)
                full_idx = np.array(
                    [i for i in idx_m if dates[i] in full_set],
                    dtype=int)
                if len(full_idx) > 0:
                    tr.append(full_idx)
                continue

            # 月内完整天随机洗牌
            shuffled_days = list(full_days)
            rng.shuffle(shuffled_days)

            # [v13.12 修复] 按比例分配, 保证每集至少 1 天
            # 旧版 bug: n_va_days 和 n_te_days 各自独立 round, 导致跨月场景下比例严重偏差
            #   例如 2 月×10 天=20 天, ratios=(0.7, 0.15, 0.15):
            #     每月 val=round(10*0.15)=2, test=round(10*0.15)=2, train=10-2-2=6
            #     累加 20 天: train=12, val=4, test=4 -> 60/20/20 (期望 70/15/15) ❌
            # 修复原理: **先算 train 天数** (保证主目标 = ratios[0]), 再把剩余按 val:test 比例分
            #   n_tr_days = round(n_full * ratios[0])       ← 主目标精准
            #   n_leftover = n_full - n_tr_days             ← 剩余天数
            #   n_va_days = round(n_leftover * val/(val+test))  ← 按 val:test 比例分剩余
            #   n_te_days = n_leftover - n_va_days
            #   每集至少 1 天 (保护)
            _sum_va_te = ratios[1] + ratios[2]
            if _sum_va_te > 0:
                n_tr_days = int(round(n_full * ratios[0]))
                n_leftover = n_full - n_tr_days
                # val 在 leftover 中占的比例 = ratios[1] / (ratios[1]+ratios[2])
                n_va_days = int(round(n_leftover * ratios[1] / _sum_va_te))
                n_te_days = n_leftover - n_va_days
            else:
                # 极端: val+test=0, 全归 train
                n_tr_days, n_va_days, n_te_days = n_full, 0, 0
            # 保证每集至少 1 天 (仅当 n_full >= 3)
            if n_full >= 3:
                if n_va_days < 1:
                    # 从 train 挪 1 天到 val
                    n_va_days = 1
                    n_tr_days = n_full - n_va_days - n_te_days
                if n_te_days < 1:
                    # 从 train 挪 1 天到 test (若 train 至少 2)
                    if n_tr_days > 1:
                        n_te_days = 1
                        n_tr_days -= 1
                    elif n_va_days > 1:
                        n_te_days = 1
                        n_va_days -= 1
            # 保护: 若 va+te 超出, 让 te 减让 (旧版逻辑保留, 极端 corner case)
            while n_va_days + n_te_days >= n_full:
                if n_te_days > 1:
                    n_te_days -= 1
                elif n_va_days > 1:
                    n_va_days -= 1
                else:
                    break
            n_tr_days = n_full - n_va_days - n_te_days

            tr_days = set(shuffled_days[:n_tr_days])
            va_days = set(shuffled_days[n_tr_days:n_tr_days + n_va_days])
            te_days = set(shuffled_days[n_tr_days + n_va_days:])

            # 把整天的样本归到对应 split
            for i in idx_m:
                d = dates[i]
                if   d in tr_days: tr.append(np.array([i]))
                elif d in va_days: va.append(np.array([i]))
                elif d in te_days: te.append(np.array([i]))

        result = {
            "train": np.sort(np.concatenate(tr)) if tr else np.array([], int),
            "val":   np.sort(np.concatenate(va)) if va else np.array([], int),
            "test":  np.sort(np.concatenate(te)) if te else np.array([], int),
        }
        # 把统计信息塞到 result, 让上层 log 可选地输出
        result["_meta"] = {
            "n_full_days":    n_full_days_total,
            "n_partial_days": n_partial_days_total,
            "full_day_threshold": FULL_DAY_MIN_SAMPLES,
            "seed": seed,
        }
        return result

    elif strategy == "global_stratified":
        # [v13.13] 全局按天随机抽样 (不按月分层), 保证 ratios 全局精准
        # 设计动机: stratified_day 按月分层, 在数据跨月且各月天数不均时,
        #   每月独立"至少 1 天 val/test"保护会累积偏差, 无法精准满足全局 ratios.
        #   例: 14 天数据 (5-21~5-31 = 11 天 + 6-01~6-03 = 3 天), ratios=(0.7, 0.15, 0.15)
        #     stratified_day: 6 月太小被强制 1/1/1 → 总累加 9/3/2 (64/21/14%, 偏离 70/15/15)
        #     global_stratified: 全局 14 天直接切 → 10/2/2 (71/14/14%, 精准) ✓
        # 权衡: 牺牲跨月分布均衡性 (原 stratified_day 保证每月都有 val/test),
        #   换取全局比例精准. 适合"数据不跨月/月天数极不均衡"的用户.
        # 保留旧的完整天 vs 碎片天分离逻辑 (碎片天全归 train).
        from numpy.random import default_rng
        rng = default_rng(seed)

        dates = ts.normalize()
        tr, va, te = [], [], []

        day_counts_all = pd.Series(dates).value_counts().sort_index()
        full_days_all = [d for d, c in day_counts_all.items()
                         if c >= FULL_DAY_MIN_SAMPLES]
        partial_days_all = [d for d, c in day_counts_all.items()
                            if c <  FULL_DAY_MIN_SAMPLES]

        # 碎片天全归 train
        if partial_days_all:
            partial_set = set(partial_days_all)
            partial_idx = np.array(
                [i for i in range(len(ts)) if dates[i] in partial_set],
                dtype=int)
            if len(partial_idx) > 0:
                tr.append(partial_idx)

        n_full = len(full_days_all)
        if n_full == 0:
            # 完全没完整天, 全部碎片天已归 train, 返回
            result = {
                "train": np.sort(np.concatenate(tr)) if tr else np.array([], dtype=int),
                "val":   np.array([], dtype=int),
                "test":  np.array([], dtype=int),
                "_meta": {"n_full_days": 0, "n_partial_days": len(partial_days_all),
                          "full_day_threshold": FULL_DAY_MIN_SAMPLES, "seed": seed,
                          "strategy": "global_stratified"},
            }
            return result

        # 全局洗牌 + 按 ratios 切
        shuffled_days = list(full_days_all)
        rng.shuffle(shuffled_days)

        # 精准按比例: train 优先, 再按 val:test 比例分剩余
        _sum_va_te = ratios[1] + ratios[2]
        if _sum_va_te > 0:
            n_tr_days = int(round(n_full * ratios[0]))
            n_leftover = n_full - n_tr_days
            n_va_days = int(round(n_leftover * ratios[1] / _sum_va_te))
            n_te_days = n_leftover - n_va_days
        else:
            n_tr_days, n_va_days, n_te_days = n_full, 0, 0

        # 保护: 至少 1 天 (仅当 n_full >= 3)
        if n_full >= 3:
            if n_va_days < 1:
                n_va_days = 1
                n_tr_days = n_full - n_va_days - n_te_days
            if n_te_days < 1:
                if n_tr_days > 1:
                    n_te_days = 1
                    n_tr_days -= 1
                elif n_va_days > 1:
                    n_te_days = 1
                    n_va_days -= 1
        while n_va_days + n_te_days >= n_full:
            if n_te_days > 1:
                n_te_days -= 1
            elif n_va_days > 1:
                n_va_days -= 1
            else:
                break
        n_tr_days = n_full - n_va_days - n_te_days

        tr_days = set(shuffled_days[:n_tr_days])
        va_days = set(shuffled_days[n_tr_days:n_tr_days + n_va_days])
        te_days = set(shuffled_days[n_tr_days + n_va_days:])

        # 把整天的样本归到对应 split
        full_idx_all = np.array(
            [i for i in range(len(ts)) if dates[i] in set(full_days_all)],
            dtype=int)
        for i in full_idx_all:
            d_i = dates[i]
            if d_i in tr_days:
                tr.append(np.array([i]))
            elif d_i in va_days:
                va.append(np.array([i]))
            elif d_i in te_days:
                te.append(np.array([i]))

        result = {
            "train": np.sort(np.concatenate(tr)) if tr else np.array([], dtype=int),
            "val":   np.sort(np.concatenate(va)) if va else np.array([], dtype=int),
            "test":  np.sort(np.concatenate(te)) if te else np.array([], dtype=int),
            "_meta": {
                "n_full_days":    n_full,
                "n_partial_days": len(partial_days_all),
                "full_day_threshold": FULL_DAY_MIN_SAMPLES,
                "seed": seed,
                "strategy": "global_stratified",
            },
        }
        return result

    else:
        raise ValueError(
            f"未知切分策略: {strategy!r}, "
            f"必须是 'time' / 'stratified' / 'stratified_day' / 'global_stratified'"
        )


# ============================================================
# CLI 自检
# ============================================================
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    print("=" * 60)
    print("split_utils 自检")
    print("=" * 60)

    # 模拟 1000 个时间戳跨 4 个月
    ts = pd.date_range("2025-07-01", periods=1000, freq="3h")
    print(f"\n模拟数据: {len(ts)} 条, 时段 {ts.min()} ~ {ts.max()}")
    print(f"月份分布: {ts.to_period('M').value_counts().sort_index().to_dict()}")

    # 测试 1: 默认 (从 common 读取)
    print("\n[1] 默认配置 (从 common.SPLIT_STRATEGY/SPLIT_RATIOS 读取)")
    sp = make_splits(ts)
    for k, v in sp.items():
        if k.startswith("_"): continue
        print(f"  {k:<6}: {len(v):>5} 条 ({len(v)/len(ts)*100:.1f}%)")

    # 测试 2: 自定义比例
    print("\n[2] stratified 自定义 80/10/10")
    sp2 = make_splits(ts, strategy="stratified", ratios=(0.80, 0.10, 0.10))
    for k, v in sp2.items():
        if k.startswith("_"): continue
        print(f"  {k:<6}: {len(v):>5} 条 ({len(v)/len(ts)*100:.1f}%)")

    # 测试 3: 纯时序
    print("\n[3] time 纯时序 70/15/15")
    sp3 = make_splits(ts, strategy="time", ratios=(0.70, 0.15, 0.15))
    for k, v in sp3.items():
        if k.startswith("_"): continue
        print(f"  {k:<6}: {len(v):>5} 条 ({len(v)/len(ts)*100:.1f}%)")

    # 测试 4: v6.10 stratified_day
    print("\n[4] stratified_day v6.10 (按天切分) 70/15/15")
    # 模拟数据: 3h 步长 -> 8 步/天, 完整天阈值要降低做测试
    # 用真实 15min 步长重做
    ts2 = pd.date_range("2025-07-01", "2025-10-31 23:45", freq="15min")
    print(f"  模拟数据 (15min): {len(ts2)} 条")
    sp4 = make_splits(ts2, strategy="stratified_day", ratios=(0.70, 0.15, 0.15))
    for k, v in sp4.items():
        if k.startswith("_"): continue
        print(f"  {k:<6}: {len(v):>5} 条 ({len(v)/len(ts2)*100:.1f}%)")
    if "_meta" in sp4:
        print(f"  meta: {sp4['_meta']}")

    # 测试 5: 验证 stratified_day 无同天切分
    print("\n[5] 验证 stratified_day 无同天切分:")
    tr_set, va_set, te_set = set(sp4["train"]), set(sp4["val"]), set(sp4["test"])
    tr_dates = {ts2[i].date() for i in tr_set}
    va_dates = {ts2[i].date() for i in va_set}
    te_dates = {ts2[i].date() for i in te_set}
    overlap_tv = tr_dates & va_dates
    overlap_tt = tr_dates & te_dates
    overlap_vt = va_dates & te_dates
    print(f"  train/val 同天数: {len(overlap_tv)}  (期望 0)")
    print(f"  train/test 同天数: {len(overlap_tt)}  (期望 0)")
    print(f"  val/test 同天数: {len(overlap_vt)}   (期望 0)")
    assert len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0, \
        "❌ stratified_day 出现同天泄漏!"
    print("  ✓ 无同天泄漏")

    # 测试 6: 错误处理
    print("\n[6] 错误处理: 比例和≠1 (应自动归一化)")
    sp6 = make_splits(ts, ratios=(7, 1.5, 1.5))  # 故意不归一
    for k, v in sp6.items():
        if k.startswith("_"): continue
        print(f"  {k:<6}: {len(v):>5} 条")

    print("\n[7] 错误处理: 比例含负数 (应抛异常)")
    try:
        make_splits(ts, ratios=(0.7, -0.1, 0.4))
        print("  ❌ 未捕获异常")
    except ValueError as e:
        print(f"  ✓ 正确抛出: {e}")
