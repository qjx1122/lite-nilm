# -*- coding: utf-8 -*-
"""
评估指标计算 + CSV 落盘工具 (训练/验证/测试/推理通用)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score, confusion_matrix)


def compute_classification_metrics(y_true, y_pred, p_score=None):
    """开/关状态分类指标"""
    out = {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if p_score is not None and len(np.unique(y_true)) > 1:
        try:
            out["AUC"] = float(roc_auc_score(y_true, p_score))
        except Exception:
            out["AUC"] = float("nan")
    else:
        out["AUC"] = float("nan")
    # 混淆矩阵展平
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if cm.size == 4:
        tn, fp, fn, tp = cm.tolist()
        out.update({"TN": int(tn), "FP": int(fp),
                    "FN": int(fn), "TP": int(tp)})
    return out


def compute_regression_metrics(y_true, y_pred, sample_period_h=0.25):
    """
    功率回归指标
    sample_period_h: 采样周期 (小时), 15min=0.25, 用于能耗换算
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sae  = float(abs(y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-6))
    nde  = float(np.sqrt(((y_true - y_pred) ** 2).sum()
                         / max((y_true ** 2).sum(), 1e-6)))
    kwh_true = float(y_true.sum() * sample_period_h / 1000.0)
    kwh_pred = float(y_pred.sum() * sample_period_h / 1000.0)
    return {
        "MAE_W": mae, "RMSE_W": rmse, "SAE": sae, "NDE": nde,
        "kWh_true": kwh_true, "kWh_pred": kwh_pred,
        "kWh_err": float(kwh_pred - kwh_true),
        "n_samples": int(len(y_true)),
    }


def save_metrics_csv(rows: list, out_path: Path, append: bool = True):
    """
    rows: List[dict], 每行一个评估记录 (split / model / metric_name / value / ...)
    输出长表 (便于后续对比)

    v6.10 改进:
      - 默认 append=True: 文件存在时按列对齐追加, 不存在时新建带表头
      - 追加时自动按现有 CSV 的列顺序对齐, 缺列补 NaN, 多列保留新行 (兼容跨版本字段)
      - 完整保留所有历史记录, 不去重 (timestamp+version+split+model 自然区分)
      - 若需覆盖, 显式传 append=False

    参数:
        rows     : flatten_metrics_to_rows 返回的指标行列表
        out_path : 输出 CSV 路径
        append   : True=追加 (默认), False=覆盖
    """
    if not rows:
        return pd.DataFrame()
    df_new = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if append and out_path.exists():
        # 追加模式: 读旧文件, 按列对齐, 缺列补 NaN
        try:
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
            # 合并列: 旧文件列优先, 新增列追加在后
            all_cols = list(df_old.columns) + \
                       [c for c in df_new.columns if c not in df_old.columns]
            df_old = df_old.reindex(columns=all_cols)
            df_new = df_new.reindex(columns=all_cols)
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            return df_merged
        except Exception as e:
            # 旧文件读取失败 (例如损坏), 退回覆盖写避免数据丢失
            import logging
            logging.warning(f"[save_metrics_csv] 旧文件读取失败 ({e}), "
                            f"改为覆盖写: {out_path}")
            df_new.to_csv(out_path, index=False, encoding="utf-8-sig")
            return df_new
    else:
        # 覆盖模式或文件不存在
        df_new.to_csv(out_path, index=False, encoding="utf-8-sig")
        return df_new


def save_predictions_csv(timestamps, y_true, y_pred, state_true=None,
                         state_pred=None, p_on=None,
                         y_pred_low=None, y_pred_high=None,
                         out_path: Path = None):
    """
    保存预测明细到 CSV
    必备列: time, y_true_W, y_pred_W, residual_W
    可选列: state_true / state_pred / p_on / y_pred_low_W / y_pred_high_W
    """
    data = {
        "time": pd.to_datetime(timestamps),
        "y_true_W": np.round(y_true, 3),
        "y_pred_W": np.round(y_pred, 3),
        "residual_W": np.round(np.asarray(y_pred) - np.asarray(y_true), 3),
    }
    if state_true is not None:
        data["state_true"] = np.asarray(state_true).astype(int)
    if state_pred is not None:
        data["state_pred"] = np.asarray(state_pred).astype(int)
    if p_on is not None:
        data["p_on"] = np.round(p_on, 4)
    if y_pred_low is not None:
        data["y_pred_low_W"]  = np.round(y_pred_low, 3)
    if y_pred_high is not None:
        data["y_pred_high_W"] = np.round(y_pred_high, 3)
    df = pd.DataFrame(data)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def flatten_metrics_to_rows(split: str, model_name: str,
                            cls_metrics: dict = None,
                            reg_metrics: dict = None,
                            extra: dict = None,
                            source: str = None) -> list:
    """
    把字典型指标拍平成 CSV 长表行

    v6.10: 每行自动注入 project_version 字段 (从 common.PROJECT_VERSION 读取),
            支持跨版本追加历史指标后清楚区分 "这条指标是哪个软件版本跑出来的"。

    v6.11: 新增 source 字段, 区分指标来自哪个训练会话:
        - source="main_train"     主模型 03_train.py 写入
        - source="v42_baseline"   v42 03b_train_v42_baseline.py 写入
        - source="evaluate"       04_evaluate.py 写入
        - source="inference"      05_inference.py 写入
      若不提供, 自动用环境变量 NILM_BASELINE_MODE 推断 main_train / v42_baseline

    参数:
        source : 来源标签, 解决"主模型 fallback 与 v42 fallback 同名互相覆盖"的 bug
    """
    # 延迟导入避免循环依赖
    try:
        from common import PROJECT_VERSION
    except ImportError:
        PROJECT_VERSION = "unknown"

    # v6.11 自动推断 source (若未显式提供)
    if source is None:
        import os
        source = "v42_baseline" if os.environ.get("NILM_BASELINE_MODE") == "1" \
                 else "main_train"

    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = {
        "timestamp": ts,
        "project_version": PROJECT_VERSION,   # v6.10: 区分指标对应的软件版本
        "source": source,                     # v6.11: 区分指标来源
        "split": split,
        "model": model_name,
    }
    if extra:
        base.update(extra)
    for src_name, src in (("classification", cls_metrics),
                          ("regression", reg_metrics)):
        if not src:
            continue
        for k, v in src.items():
            row = dict(base)
            row.update({"metric_type": src_name, "metric": k, "value": v})
            rows.append(row)
    return rows


def build_comparison_table(metrics_rows: list,
                           include_metrics: list = None) -> pd.DataFrame:
    """
    把长表 metrics_rows 透视成多模型对比表 (行=指标, 列=模型)
    便于一眼看清各模型差异.

    参数:
        metrics_rows    : flatten_metrics_to_rows 返回的所有行 (多模型合并)
        include_metrics : 只保留指定指标 (默认全部)

    返回:
        DataFrame, 行索引 = (metric_type, metric), 列 = 各模型, 值 = value
    """
    if not metrics_rows:
        return pd.DataFrame()
    df = pd.DataFrame(metrics_rows)
    if include_metrics:
        df = df[df["metric"].isin(include_metrics)]
    pivot = df.pivot_table(
        index=["metric_type", "metric"],
        columns="model",
        values="value",
        aggfunc="first",
    )
    # 标准列序: main 在前, 其它按字母
    cols = list(pivot.columns)
    main_cols = [c for c in cols if "main" in c.lower()]
    other_cols = sorted([c for c in cols if c not in main_cols])
    pivot = pivot[main_cols + other_cols]
    return pivot.reset_index()


def compute_leak_ood_split(timestamps, train_dates_set) -> tuple:
    """[v13.8] 按"是否在训练集日期集合内"把推理时间戳拆成两组

    参数:
        timestamps:       pd.DatetimeIndex 或可转 datetime 的序列 (推理样本时间戳)
        train_dates_set:  set of str (ISO yyyy-mm-dd) 或 set of date, 训练日实际集合

    返回:
        (mask_leak, mask_ood, meta_dict)
          - mask_leak: bool ndarray, True 表示该样本时间戳所在自然日 ∈ train_dates_set
          - mask_ood:  bool ndarray, mask_leak 的补集 (从未训练过, out-of-distribution)
          - meta_dict: {"n_leak": int, "n_ood": int, "leak_dates": sorted list, "ood_dates": sorted list}
    """
    ts = pd.to_datetime(pd.Series(timestamps))
    # 统一 train_dates_set 到 ISO 字符串, 避免 date 对象 vs str 类型不匹配的坑
    if train_dates_set is None:
        train_dates_set = set()
    normalized_train = set()
    for d in train_dates_set:
        if isinstance(d, str):
            normalized_train.add(d)
        else:
            # date / datetime / Timestamp -> ISO
            normalized_train.add(pd.Timestamp(d).strftime("%Y-%m-%d"))

    sample_dates = ts.dt.strftime("%Y-%m-%d").values
    mask_leak = np.array([d in normalized_train for d in sample_dates])
    mask_ood = ~mask_leak

    leak_dates_used = sorted({d for d in sample_dates[mask_leak]})
    ood_dates_used = sorted({d for d in sample_dates[mask_ood]})
    return mask_leak, mask_ood, {
        "n_leak": int(mask_leak.sum()),
        "n_ood": int(mask_ood.sum()),
        "leak_dates": leak_dates_used,
        "ood_dates": ood_dates_used,
    }


def build_leak_ood_metric_rows(timestamps, y_true, y_pred_main, s_true, s_pred_main,
                               p_on_main, train_dates_set,
                               extra_base: dict = None, source: str = "inference",
                               logger=None,
                               extra_model_preds: dict = None) -> tuple:
    """[v13.8] 检测数据泄漏并生成拆分指标行 (leak/ood 两组)

    典型调用位置: 05_inference.py 里 save_metrics_csv 之前.

    参数:
        timestamps:        推理样本时间戳 (DatetimeIndex)
        y_true:            真实功率 (W)
        y_pred_main:       主模型 "main" 的预测功率 (W); 会被记为 model="main"
        s_true:            真实 ON 状态 (0/1), 用 ON_THR 计算
        s_pred_main:       主模型预测 ON 状态 (0/1); 所有 extra_model_preds 共用该 s_pred
                           (因分类阈值由 clf 决定, L4/L5 只改回归量, 状态不变)
        p_on_main:         主模型 ON 概率 (用于 AUC), 可为 None
        train_dates_set:   训练集实际使用的日期集合 (来自 bundle["train_dates"])
        extra_base:        追加到 metric 行的 extra 字段 (会自动补 note)
        source:            metric 行的 source 字段
        logger:            日志对象
        extra_model_preds: [v13.8-fix1] 可选, 额外模型名 → y_pred_W 数组的 dict.
                           每个额外模型会跑一次 leak/ood 回归指标拆分, 分类结果与 main 共用.
                           典型用法: {"main_final": y_pred_L4L5, "main_L4_calib": y_pred_after_L4}
                           空 dict / None 表示只算 main (向后兼容旧签名).

    返回:
        (leak_rows, ood_rows, meta_dict)
          - leak_rows / ood_rows: list of dict, 可直接送 save_metrics_csv.
                                  每个额外模型会追加自己的 leak/ood 两组回归行 (共用一份分类行).
          - meta_dict: compute_leak_ood_split 的返回

    注: 若 meta.n_leak == 0, 返回 ([], [], meta) 且 WARN 无泄漏; 只有泄漏时才产 rows.

    设计权衡:
      - 分类指标 (F1/Prec/Rec/AUC) 只按 main 一次, 因 3 个主模型变体共用同一个 state_pred
      - 回归指标 (MAE/RMSE/SAE/kWh) 每个模型独立计算, 因 L4/L5 会改变 y_pred_W
      - 对比表打印时也会额外列出 extra 模型的 SAE/MAE, 便于快速判断 L4/L5 收益
    """
    _log = logger.warning if logger is not None else print
    _info = logger.info if logger is not None else print

    mask_leak, mask_ood, meta = compute_leak_ood_split(timestamps, train_dates_set)

    if meta["n_leak"] == 0:
        _info(f"  [v13.8 泄漏检测] [OK] 推理集与训练集日期无重叠, "
              f"OOD 覆盖 {meta['n_ood']} 样本 / {len(meta['ood_dates'])} 天")
        return [], [], meta

    # 有泄漏 -> WARN + 拆分指标
    _log(f"  [v13.8 泄漏检测] [WARN] 推理集与训练集日期存在重叠! "
         f"泄漏 {meta['n_leak']} 样本 / {len(meta['leak_dates'])} 天, "
         f"OOD {meta['n_ood']} 样本 / {len(meta['ood_dates'])} 天")
    _log(f"    泄漏日期: {meta['leak_dates']}")
    _log(f"    OOD 日期 (可信): {meta['ood_dates']}")
    _log(f"    建议: 在 infer.exclude 中明确排除训练区间, 或以 inference_ood 指标为准评估泛化能力.")

    y_true = np.asarray(y_true)
    s_true = np.asarray(s_true)
    s_pred = np.asarray(s_pred_main)
    p_score = np.asarray(p_on_main) if p_on_main is not None else None

    # 合并 main + extras 为统一列表 (main 在前, 保证 CSV/日志顺序稳定)
    # 用 dict 保序: main 首个, 其余按传入顺序
    model_preds = {"main": np.asarray(y_pred_main)}
    if extra_model_preds:
        for _mn, _mp in extra_model_preds.items():
            if _mn == "main":
                continue  # 防止用户误传覆盖
            model_preds[_mn] = np.asarray(_mp)

    def _rows_for(mask, split_name, note):
        """对某一 mask (leak/ood), 生成所有模型的指标行.
        分类只在 main 上算一次; 每个模型各出一份回归.
        """
        n_mask = int(mask.sum())
        if n_mask == 0:
            return []
        merged_extra = dict(extra_base or {})
        merged_extra["note"] = note

        rows = []
        # (a) main 模型: 分类 + 回归
        cls_main = compute_classification_metrics(
            s_true[mask], s_pred[mask],
            p_score=(p_score[mask] if p_score is not None else None))
        reg_main = compute_regression_metrics(y_true[mask], model_preds["main"][mask])
        rows += flatten_metrics_to_rows(split_name, "main",
                                        cls_metrics=cls_main, reg_metrics=reg_main,
                                        extra=merged_extra, source=source)
        # (b) 额外模型: 分类共用 main 结果 (state_pred 相同), 仅算回归
        for _mn, _mp in model_preds.items():
            if _mn == "main":
                continue
            _reg = compute_regression_metrics(y_true[mask], _mp[mask])
            # 分类共用 cls_main (状态阈值不变, F1/Prec/Rec/AUC 完全相同)
            rows += flatten_metrics_to_rows(split_name, _mn,
                                            cls_metrics=cls_main, reg_metrics=_reg,
                                            extra=merged_extra, source=source)
        return rows

    leak_rows = _rows_for(mask_leak, "inference_leak",
                          f"[v13.8] 训练泄漏部分 ({meta['n_leak']} 样本, {len(meta['leak_dates'])} 天), 指标偏乐观")
    ood_rows = _rows_for(mask_ood, "inference_ood",
                         f"[v13.8] 真 OOD 部分 ({meta['n_ood']} 样本, {len(meta['ood_dates'])} 天), 代表真实泛化能力")

    # 打印对比表让运维一眼看清差异
    if leak_rows and ood_rows:
        _log("  [v13.8 泄漏拆分对比]")
        # 分类指标: main 一列即可 (共用)
        _log(f"    {'指标':<14}{'model':<20}{'inference_leak':>18}{'inference_ood':>18}")
        # 分类 5 指标 (只出 main)
        for m in ["F1", "Precision", "Recall"]:
            v_leak = next((r["value"] for r in leak_rows if r["metric"] == m and r["model"] == "main"), None)
            v_ood  = next((r["value"] for r in ood_rows  if r["metric"] == m and r["model"] == "main"), None)
            if v_leak is not None and v_ood is not None:
                _log(f"    {m:<14}{'main':<20}{v_leak:>18.4f}{v_ood:>18.4f}")
        # 回归 2 指标: 每个模型都出一行
        for m in ["SAE", "MAE_W"]:
            for _mn in model_preds.keys():
                v_leak = next((r["value"] for r in leak_rows if r["metric"] == m and r["model"] == _mn), None)
                v_ood  = next((r["value"] for r in ood_rows  if r["metric"] == m and r["model"] == _mn), None)
                if v_leak is not None and v_ood is not None:
                    _log(f"    {m:<14}{_mn:<20}{v_leak:>18.4f}{v_ood:>18.4f}")

    return leak_rows, ood_rows, meta


def compute_raw_daily_counts(csv_path, time_col: str,
                              time_filter_spec: str = None,
                              logger=None) -> dict:
    """[v13.16] 读取 CSV, 按天 group 统计原始采样点数.

    用于给 build_daily_metrics_rows 传 bus_daily_counts / branch_daily_counts.

    参数:
        csv_path:         CSV 路径 (总线用 event_time 或 分路用 time)
        time_col:         时间列名 (总线通常 "event_time", 分路通常 "time")
        time_filter_spec: [可选] JSON 字符串, 若提供则先应用时段过滤
                          (让统计口径与实际推理评估范围一致)
        logger:           logger, 用于打印统计摘要

    返回:
        {"yyyy-mm-dd": int} 每天的采集点数. 若 CSV 缺失/损坏, 返回 {}.
    """
    import pandas as pd
    from pathlib import Path
    try:
        p = Path(csv_path)
        if not p.exists():
            if logger:
                logger.info(f"  [v13.16] CSV 不存在, 跳过 daily counts: {csv_path}")
            return {}
        df = pd.read_csv(p, encoding="utf-8")
        if time_col not in df.columns:
            if logger:
                logger.info(f"  [v13.16] CSV 缺 '{time_col}' 列, 跳过 daily counts: "
                            f"{csv_path} (实际列: {df.columns.tolist()[:5]}...)")
            return {}
        # 解析时间
        try:
            from time_utils import parse_timestamps
            ts = parse_timestamps(df[time_col])
        except Exception:
            ts = pd.to_datetime(df[time_col], errors="coerce")
        df = df.assign(_ts=ts).dropna(subset=["_ts"])

        # [可选] 应用时段过滤 (与训练/推理阶段一致口径)
        if time_filter_spec:
            try:
                from time_filter_utils import cli_arg_to_spec, apply_time_filter
                spec = cli_arg_to_spec(time_filter_spec)
                if spec is not None:
                    # apply_time_filter 期望原时间列名, 这里临时把 _ts 覆盖回原列
                    df2 = df.copy()
                    df2[time_col] = df2["_ts"]
                    df2 = apply_time_filter(df2, time_col, spec,
                                            f"raw_counts:{p.name}", logger=None)
                    df = df2.assign(_ts=pd.to_datetime(df2[time_col]))
            except Exception as e:
                if logger:
                    logger.info(f"  [v13.16] time_filter 应用失败, 用未过滤计数: {e}")

        # 按天分组
        counts = df.groupby(df["_ts"].dt.strftime("%Y-%m-%d")).size().to_dict()
        if logger and counts:
            n_days = len(counts)
            n_total = sum(counts.values())
            n_min = min(counts.values())
            n_max = max(counts.values())
            logger.info(f"  [v13.16] {p.name} 日采集点统计: {n_days} 天, "
                        f"总计 {n_total} 点, 单日 min={n_min} max={n_max}")
        return counts
    except Exception as e:
        if logger:
            logger.info(f"  [v13.16] compute_raw_daily_counts 失败: {e}")
        return {}


def build_daily_metrics_rows(timestamps, y_true, y_pred, s_true, s_pred,
                             split_name: str, on_thr_w: float = None,
                             p_on=None, date_labels: dict = None,
                             sample_period_h: float = 0.25,
                             extra: dict = None,
                             model_name: str = "main",
                             bus_daily_counts: dict = None,
                             branch_daily_counts: dict = None) -> list:
    """[v13.14] 按天聚合主模型评估指标, 用于逐日质量追踪.

    动机: 现有指标是整体聚合 (train/val/test/inference 各一个数字), 无法追踪
      "哪一天预测得好 / 哪一天崩了". 按天聚合能:
      1. 定位单日异常 (如 6/25 SAE 突增, 6/20 Recall 掉到 0.6)
      2. 结合 dataset 归属列, 一眼看清 val/test 里哪些日子拉低整体指标
      3. 与 analyze_on_periods.py 的 daily 汇总配对 (一份看开机模式, 一份看预测质量)

    参数:
        timestamps:    样本时间戳序列 (DatetimeIndex 或可转)
        y_true:        真实功率 (W)
        y_pred:        主模型预测功率 (W, 通常是 main_final = L4+L5 生产输出)
        s_true:        真实 ON 状态 (0/1)
        s_pred:        主模型预测 ON 状态 (0/1)
        split_name:    "train" / "val" / "test" / "inference" 或自定义
        on_thr_w:      ON 阈值 (W), 仅记录到 extra, 不影响计算 (s_true/s_pred 已定)
        p_on:          主模型 ON 概率 (用于日级 AUC, 可选)
        date_labels:   {"yyyy-mm-dd": "train"/"val"/"test"/"未使用"/"used"/"excluded"}
                       若提供, 输出行加 `dataset` 列
        sample_period_h: 每采样点代表的小时数 (默认 0.25 = 15min)
        extra:         每行附加的固定字段 (如 bus_csv, model_file, project_version)
        model_name:    通常 "main", 也可传 "main_final" / "main_L4_calib" 等区分
        bus_daily_counts:    [v13.16] {"yyyy-mm-dd": int} 当天总线原始 CSV 采样点数
                             (与对齐后 n_samples 不同, 反映采集完整性;
                             None 时输出空字符串, 向后兼容)
        branch_daily_counts: [v13.16] {"yyyy-mm-dd": int} 当天分路原始 CSV 采样点数
                             (语义同上)

    返回: list[dict], 每 dict 一行, 字段:
        date, split, model, n_samples, n_bus_raw, n_branch_raw,
        Accuracy, Precision, Recall, F1, AUC,
        MAE_W, RMSE_W, SAE, kWh_true, kWh_pred, kWh_err,
        TP, FP, FN, TN,
        is_on_day, off_day_fp, off_day_false_on_kWh, on_only_mae_w  ([v14.2] P7 口径),
        coverage_samples_96, bus_coverage_288, branch_coverage_96,
        partial_day, no_positive_day, f1_lt_90_on, sae_gt_20_on ([v14.8] P4 监控口径),
        [dataset (若 date_labels 提供)],
        [+ extra 字段]
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 accuracy_score, roc_auc_score, confusion_matrix)

    ts = pd.to_datetime(pd.Series(timestamps))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    s_true = np.asarray(s_true, dtype=int)
    s_pred = np.asarray(s_pred, dtype=int)
    has_p = p_on is not None
    if has_p:
        p_on_arr = np.asarray(p_on, dtype=float)

    df = pd.DataFrame({
        "ts": ts,
        "y_true": y_true, "y_pred": y_pred,
        "s_true": s_true, "s_pred": s_pred,
        "date": ts.dt.strftime("%Y-%m-%d"),
    })
    if has_p:
        df["p_on"] = p_on_arr

    rows = []
    for date_str, g in df.groupby("date", sort=True):
        st = g["s_true"].values
        sp = g["s_pred"].values
        yt = g["y_true"].values
        yp = g["y_pred"].values
        n = int(len(g))

        # 分类指标
        acc  = float(accuracy_score(st, sp))
        prec = float(precision_score(st, sp, zero_division=0))
        rec  = float(recall_score(st, sp, zero_division=0))
        f1   = float(f1_score(st, sp, zero_division=0))
        # AUC: 需要 p_on 且该日至少有一个正类和一个负类
        auc = None
        if has_p:
            pv = g["p_on"].values
            if len(np.unique(st)) >= 2:
                try:
                    auc = float(roc_auc_score(st, pv))
                except Exception:
                    auc = None
        # 混淆矩阵
        try:
            tn, fp, fn, tp = confusion_matrix(st, sp, labels=[0, 1]).ravel()
        except Exception:
            tn = fp = fn = tp = 0

        # 回归指标
        mae = float(np.mean(np.abs(yp - yt))) if n > 0 else 0.0
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2))) if n > 0 else 0.0
        kwh_true = float(yt.sum()) * sample_period_h / 1000.0
        kwh_pred = float(yp.sum()) * sample_period_h / 1000.0
        kwh_err = kwh_pred - kwh_true
        # [v13.14] SAE 边界保护: kwh_true ≈ 0 时 (全 OFF 天), SAE 无意义, 记为 None
        # 传统定义 SAE = |kwh_err| / kwh_true 在 kwh_true=0 时爆炸 (如 8e7),
        # 会污染整体统计. 全 OFF 天用户关心的是 kwh_pred 绝对值 (是否有误报能耗),
        # 不是相对误差. 阈值 1e-3 kWh (= 1 Wh, 极小的待机也 > 此值).
        if kwh_true < 1e-3:
            sae = None  # 语义: "该日无真实用电, SAE 不适用"
        else:
            sae = abs(kwh_err) / kwh_true

        # [v13.16] 新增 2 列: 当天总线/分路原始采集点数
        # 语义: 反映"当天原始采集是否完整"(总线 5min 满 288 点, 分路 15min 满 96 点),
        # 与对齐后 n_samples 不同 (n_samples 已受时段过滤+对齐 inner-join 影响).
        # None 时输出 "" 空字符串, 向后兼容.
        n_bus_raw = ""
        if bus_daily_counts is not None:
            n_bus_raw = int(bus_daily_counts.get(date_str, 0))
        n_branch_raw = ""
        if branch_daily_counts is not None:
            n_branch_raw = int(branch_daily_counts.get(date_str, 0))

        # [v14.2] P7 口径列 (ON/OFF 日拆分, 防"全 OFF 日 F1=0 污染观感"):
        #   is_on_day:            当天真值含 ON (TP+FN>0) 记 1, 全 OFF 记 0
        #   off_day_fp:           仅全 OFF 日有意义: 当日 FP 数 (ON 日恒 0)
        #   off_day_false_on_kWh: 仅全 OFF 日有意义: 当日误报电量 (ON 日恒 0.0)
        #   on_only_mae_w:        仅真 ON 点上的 MAE (无 ON 点 -> "" 空字符串);
        #                         全 OFF 日看 off_day_*, ON 日看 on_only_mae_w + SAE,
        #                         避免用整体 F1/MAE 同时衡量两类日
        _is_on_day = int((tp + fn) > 0)
        _off_fp = int(fp) if not _is_on_day else 0
        _false_on_kwh = round(kwh_pred, 6) if not _is_on_day else 0.0
        _on_mask = st == 1
        _on_only_mae = (float(np.mean(np.abs(yp[_on_mask] - yt[_on_mask])))
                        if _on_mask.any() else None)

        # [v14.8] P4 生产化日报监控字段：
        #   coverage_samples_96: 对齐后有效样本覆盖率 (15min 满日=96)
        #   bus/branch coverage: 原始采集覆盖率 (总线5min满日=288, 分路15min满日=96)
        #   no_positive_day: TP+FP+FN=0, 此时 F1=0 是 zero-division 口径, 非模型失败
        #   f1_lt_90_on/sae_gt_20_on: 只在 ON 日标记异常, 避免 OFF 日污染告警
        _coverage_samples_96 = float(n / 96.0) if n >= 0 else 0.0
        _bus_coverage_288 = ""
        if isinstance(n_bus_raw, int):
            _bus_coverage_288 = round(float(n_bus_raw) / 288.0, 6)
        _branch_coverage_96 = ""
        if isinstance(n_branch_raw, int):
            _branch_coverage_96 = round(float(n_branch_raw) / 96.0, 6)
        _no_positive_day = int((tp + fp + fn) == 0)
        _partial_day = int(_coverage_samples_96 < 0.90)
        _f1_lt_90_on = int(bool(_is_on_day) and f1 < 0.90)
        _sae_gt_20_on = int(bool(_is_on_day) and sae is not None and sae > 0.20)

        row = {
            "date": date_str,
            "split": split_name,
            "model": model_name,
            "n_samples": n,
            "n_bus_raw": n_bus_raw,             # [v13.16]
            "n_branch_raw": n_branch_raw,       # [v13.16]
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "AUC": round(auc, 6) if auc is not None else "",
            "MAE_W": round(mae, 3),
            "RMSE_W": round(rmse, 3),
            "SAE": round(sae, 6) if sae is not None else "",
            "kWh_true": round(kwh_true, 6),
            "kWh_pred": round(kwh_pred, 6),
            "kWh_err": round(kwh_err, 6),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "is_on_day": _is_on_day,                       # [v14.2]
            "off_day_fp": _off_fp,                         # [v14.2]
            "off_day_false_on_kWh": _false_on_kwh,         # [v14.2]
            "on_only_mae_w": (round(_on_only_mae, 3)
                              if _on_only_mae is not None else ""),  # [v14.2]
            "coverage_samples_96": round(_coverage_samples_96, 6),  # [v14.8 P4]
            "bus_coverage_288": _bus_coverage_288,                  # [v14.8 P4]
            "branch_coverage_96": _branch_coverage_96,              # [v14.8 P4]
            "partial_day": _partial_day,                            # [v14.8 P4]
            "no_positive_day": _no_positive_day,                    # [v14.8 P4]
            "f1_lt_90_on": _f1_lt_90_on,                            # [v14.8 P4]
            "sae_gt_20_on": _sae_gt_20_on,                          # [v14.8 P4]
        }
        if date_labels is not None:
            row["dataset"] = date_labels.get(date_str, "")
        if on_thr_w is not None:
            row["on_thr_w"] = float(on_thr_w)
        if extra:
            row.update(extra)
        rows.append(row)

    return rows


def save_daily_metrics_csv(rows: list, out_path, logger=None) -> "pd.DataFrame":
    """[v13.14] 把 build_daily_metrics_rows 的 rows 保存到 CSV.

    若 out_path 已存在, 会**覆盖** (不追加, 因每次训练/推理是新的 daily 视图).
    """
    import pandas as pd
    from pathlib import Path
    if not rows:
        if logger is not None:
            logger.warning(f"  [v13.14 daily metrics] rows 为空, 跳过写 {out_path}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    if logger is not None:
        logger.info(f"  [v13.14] 逐日主模型指标 -> {p} ({len(df)} 行)")
    return df
