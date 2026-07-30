# -*- coding: utf-8 -*-
"""
多基线模型对比工具 (v5 新增)

支持的基线类型 (--baseline 参数):
    rf           : 单阶段 RandomForest (从主 bundle 的 'rf' 字段加载)
    fallback     : MoE 的全局 fallback 回归器 (无季节路由对照)
    naive_mean   : 最简单的"训练集 ON 均值"基线 (随机性下限)
    naive_zero   : 全 0 (理论下限)
    <path>.pkl   : 加载任意外部模型 bundle 文件 (如 v4.2 对照)

设计原则:
    1. 强制公平对比: 不同基线共享主模型的 scaler/feat_cols/season_labels
       (除非加载外部 pkl, 那时使用外部 bundle 的特征工程, 各自独立)
    2. 各用各阈值: 每个模型用自己 bundle 内的最优阈值, 反映其真实最优表现
    3. 无标签场景: 当 y_true 不可用时, 仅对比预测一致性 (互相 MAE)
    4. 统一接口: BaselineRunner.run() 返回 (预测序列, 指标字典, 模型名)
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# 内置基线别名
BUILTIN_BASELINES = {"rf", "fallback", "naive_mean", "naive_zero"}


class BaselineModel:
    """统一基线模型包装器"""

    def __init__(self, name: str, kind: str, predictor, bundle=None,
                 needs_main_features: bool = True,
                 description: str = ""):
        """
        参数:
            name        : 显示名称 (用于日志/CSV 列名)
            kind        : "rf" / "fallback" / "naive_mean" / "naive_zero" / "external"
            predictor   : 可调用对象 / sklearn 模型 / 常量 / 字典
            bundle      : 若为外部 pkl 加载, 这里存其完整 bundle
            needs_main_features: True=共享主模型 X_s, False=用 bundle 自己的特征
        """
        self.name = name
        self.kind = kind
        self.predictor = predictor
        self.bundle = bundle
        self.needs_main_features = needs_main_features
        self.description = description

    def __repr__(self):
        return f"<BaselineModel {self.name} ({self.kind})>"


class BaselineRegistry:
    """基线模型注册中心 -- 负责按名称构造 BaselineModel 实例"""

    def __init__(self, main_bundle: dict, logger=None):
        self.main_bundle = main_bundle
        self.log = logger

    def build(self, name_or_path: str) -> BaselineModel | None:
        """根据别名或文件路径构造一个 BaselineModel

        v6.11 路径解析修复:
          - 相对路径优先按当前工作目录解析 (与用户直觉一致)
          - 失败时回退到 PROJECT_ROOT 相对解析 (处理 04_evaluate 从 scripts/
            目录运行时, 用户传 'models/xxx.pkl' 这种相对路径的情况)
          - 仍找不到时 ERROR 级别报错 (而非 warning), 暴露隐式错误
        """
        if name_or_path in BUILTIN_BASELINES:
            return self._build_builtin(name_or_path)
        # 否则视为文件路径
        path = Path(name_or_path)
        # v6.11 修复: 尝试 PROJECT_ROOT 相对解析
        if not path.exists() and not path.is_absolute():
            try:
                from common import PROJECT_ROOT
                alt_path = PROJECT_ROOT / name_or_path
                if alt_path.exists():
                    path = alt_path
                    if self.log:
                        self.log.info(f"  [baseline] 路径相对 PROJECT_ROOT 解析: {alt_path}")
            except ImportError:
                pass
        if path.exists() and path.suffix == ".pkl":
            return self._build_external(path)
        if self.log:
            # v6.11: warning -> error 级别, 避免静默丢基线
            self.log.error(f"  [baseline] ❌ 无法识别或文件不存在: {name_or_path} "
                           f"(已尝试 cwd 与 PROJECT_ROOT 两种解析). 该基线被跳过, "
                           f"指标对比表将缺失对应模型!")
        return None

    # ---------- 内置基线 ----------
    def _build_builtin(self, alias: str) -> BaselineModel | None:
        if alias == "rf":
            rf = self.main_bundle.get("rf")
            if rf is None:
                if self.log:
                    self.log.warning("  [baseline/rf] 主 bundle 无 'rf' 字段, 跳过")
                return None
            return BaselineModel(
                name="rf",
                kind="rf",
                predictor=rf,
                needs_main_features=True,
                description="单阶段 RandomForest (无两阶段, 无 MoE, 无后处理)",
            )

        if alias == "fallback":
            reg = self.main_bundle.get("reg")
            if reg is None:
                if self.log:
                    self.log.warning("  [baseline/fallback] 主 bundle 无 'reg' 字段, 跳过")
                return None
            return BaselineModel(
                name="fallback",
                kind="fallback",
                predictor=reg,
                needs_main_features=True,
                description="MoE 全局兜底回归器 (无季节路由对照)",
            )

        if alias == "naive_mean":
            # 用主 bundle 已保存的 expert_summary 估算 ON 均值
            es = self.main_bundle.get("expert_summary", [])
            on_means = [e.get("y_mean") for e in es
                        if e.get("status") == "trained" and e.get("y_mean")]
            if on_means:
                global_mean = float(np.mean(on_means))
            else:
                global_mean = 500.0  # 经验兜底
            if self.log:
                self.log.info(f"  [baseline/naive_mean] 使用训练集 ON 均值 {global_mean:.1f} W")
            return BaselineModel(
                name="naive_mean",
                kind="naive_mean",
                predictor=global_mean,
                needs_main_features=True,
                description=f"训练集 ON 平均功率 ({global_mean:.0f}W) - 随机性下限",
            )

        if alias == "naive_zero":
            return BaselineModel(
                name="naive_zero",
                kind="naive_zero",
                predictor=0.0,
                needs_main_features=True,
                description="全 0 预测 - 理论下限",
            )

        return None

    # ---------- 外部 pkl ----------
    def _build_external(self, path: Path) -> BaselineModel | None:
        try:
            bundle = joblib.load(path)
        except Exception as e:
            if self.log:
                self.log.warning(f"  [baseline/external] 加载 {path.name} 失败: {e}")
            return None
        # 检查关键字段
        if "scaler" not in bundle or "feat_cols" not in bundle:
            if self.log:
                self.log.warning(f"  [baseline/external] {path.name} 缺少 scaler/feat_cols, 跳过")
            return None
        name = path.stem  # 文件名 (无扩展)
        ver  = bundle.get("version", "unknown")
        if self.log:
            n_feat = len(bundle.get("feat_names", []))
            self.log.info(f"  [baseline/external] 已加载 {path.name} "
                          f"(version={ver}, n_features={n_feat})")
        return BaselineModel(
            name=name,
            kind="external",
            predictor=None,
            bundle=bundle,
            needs_main_features=False,  # 用各自的特征工程
            description=f"外部模型 (version={ver}, file={path.name})",
        )


# ============================================================
# 多基线推理执行器
# ============================================================
class BaselineRunner:
    """
    统一执行多个基线模型, 返回所有预测序列 (dict 形式)
    """

    def __init__(self, registry: BaselineRegistry, logger=None):
        self.registry = registry
        self.log = logger

    def run_all(self, baselines: list[str],
                df_aligned: pd.DataFrame,
                top_cols: list,
                X_main_scaled: np.ndarray,
                state_pred_main: np.ndarray,
                weather_df: pd.DataFrame = None) -> dict:
        """
        参数:
            baselines       : 用户指定的基线列表 (别名 + 路径混合)
            df_aligned      : 对齐后的总线 DataFrame (含 y_ac 列若有标签)
            top_cols        : 主模型用的特征列名
            X_main_scaled   : 主模型已标准化的特征矩阵
            state_pred_main : 主模型预测的 ON/OFF 状态 (用于硬门控基线)
            weather_df      : 可选, 用于外部模型 (如 v5 vs v4.2 需重建特征)

        返回:
            {baseline_name: {"y_pred": np.array, "model": BaselineModel}}
        """
        results = {}
        for name in baselines:
            model = self.registry.build(name)
            if model is None:
                continue
            if self.log:
                self.log.info(f"  [baseline] 运行: {model.name}  ({model.description})")
            y_pred = self._predict_one(
                model, df_aligned, top_cols, X_main_scaled,
                state_pred_main, weather_df=weather_df,
            )
            if y_pred is not None:
                results[model.name] = {"y_pred": y_pred, "model": model}
        return results

    def _predict_one(self, model: BaselineModel,
                     df_aligned: pd.DataFrame,
                     top_cols: list,
                     X_main_scaled: np.ndarray,
                     state_pred_main: np.ndarray,
                     weather_df: pd.DataFrame = None) -> np.ndarray | None:
        n = len(df_aligned)

        # ---- 内置 naive ----
        if model.kind == "naive_zero":
            return np.zeros(n)
        if model.kind == "naive_mean":
            # ON 时预测均值, OFF 时 0
            mean = float(model.predictor)
            return state_pred_main.astype(float) * mean

        # ---- RF (无后处理) ----
        if model.kind == "rf":
            y = np.clip(model.predictor.predict(X_main_scaled), 0, None)
            return y

        # ---- MoE Fallback (全局回归, 应用主模型的 ON 门控) ----
        if model.kind == "fallback":
            y_reg = np.clip(model.predictor.predict(X_main_scaled), 0, None)
            return state_pred_main.astype(float) * y_reg

        # ---- 外部 pkl (独立特征工程) ----
        if model.kind == "external":
            return self._predict_external(model, df_aligned, top_cols, weather_df)

        return None

    def _predict_external(self, model: BaselineModel,
                          df_aligned: pd.DataFrame,
                          main_top_cols: list,
                          main_weather_df: pd.DataFrame) -> np.ndarray | None:
        """
        对外部 pkl 模型, 用它自己的 bundle 还原特征工程链
        (兼容 v4.2 60维 / v5 72维 / 其它版本)
        """
        from feature_utils import build_features
        from postprocess import apply_postprocess
        from expert_utils import assign_season

        b = model.bundle
        scaler  = b["scaler"]
        clf     = b.get("clf")
        reg     = b.get("reg")
        moe     = b.get("moe")
        ext_top_cols = b["feat_cols"]
        ext_best_thr = float(b.get("best_thr", 0.5))
        post_min_on = int(b.get("post_min_on", 1))
        post_fill_short_off = int(b.get("post_fill_short_off", 0))

        # 检查列对齐
        missing = [c for c in ext_top_cols if c not in df_aligned.columns]
        if missing:
            if self.log:
                self.log.warning(f"  [baseline/{model.name}] 缺列 {missing[:3]}..., 跳过")
            return None

        # 该模型是否需要温度特征
        ext_use_weather = bool(b.get("use_weather_features", False))
        ext_use_temp_season = bool(b.get("use_temp_based_season", False))

        # 重建特征 (用各自的 bundle 配置)
        ext_weather_df = main_weather_df if ext_use_weather else None
        # v6: 用外部模型自己的 LUT (若有)
        ext_lut = b.get("temp_power_lut")
        X_df = build_features(df_aligned, ext_top_cols, weather_df=ext_weather_df,
                              temp_power_lut=ext_lut)
        if list(X_df.columns) != b.get("feat_names", list(X_df.columns)):
            # 维度/列序不一致时, 按 bundle 的 feat_names 重排
            X_df = X_df.reindex(columns=b["feat_names"], fill_value=0)

        X_s = scaler.transform(X_df.values.astype(np.float32))

        # Stage-1 推理 + 后处理
        if clf is None:
            if self.log:
                self.log.warning(f"  [baseline/{model.name}] 无 clf, 跳过")
            return None
        p_on = clf.predict_proba(X_s)[:, 1]
        raw_state = (p_on >= ext_best_thr).astype(int)

        # Stage-2 推理 (MoE 优先)
        if moe is not None:
            # 计算外部模型自己的 season 标签
            if ext_use_temp_season and ext_weather_df is not None:
                daily_t = ext_weather_df["temperature_2m"].resample("D").mean()
                ts_dates = pd.DatetimeIndex(df_aligned.index).normalize()
                daily_avg = daily_t.reindex(ts_dates, method="ffill").values
                ext_season = assign_season(
                    df_aligned.index, daily_avg_temp=daily_avg,
                    use_temperature=True,
                    summer_th=b.get("summer_temp_threshold", 22.0),
                    winter_th=b.get("winter_temp_threshold", 12.0),
                )
            else:
                ext_season = assign_season(df_aligned.index, use_temperature=False)
            quantile_alpha = float(b.get("quantile_alpha", 0.5))
            p_reg = np.clip(moe.predict(X_s, ext_season, alpha=quantile_alpha),
                            0, None)
        else:
            p_reg = np.clip(reg.predict(X_s), 0, None)

        state_filt, y_pred_filt = apply_postprocess(
            raw_state, p_reg,
            min_on=post_min_on, fill_short_off=post_fill_short_off,
        )
        return y_pred_filt


# ============================================================
# 多模型预测的列合并 + CSV 输出辅助
# ============================================================
def merge_predictions(main_pred: np.ndarray,
                      baseline_results: dict,
                      timestamps,
                      y_true: np.ndarray = None,
                      extra_cols: dict = None) -> pd.DataFrame:
    """
    合并主模型 + 所有基线预测为一张宽表 DataFrame, 便于落 CSV
    """
    data = {"time": pd.to_datetime(timestamps)}
    if y_true is not None:
        data["y_true_W"] = np.round(y_true, 3)
    data["y_pred_W_main"] = np.round(main_pred, 3)
    if y_true is not None:
        data["residual_W_main"] = np.round(main_pred - y_true, 3)

    for name, info in baseline_results.items():
        y_b = info["y_pred"]
        data[f"y_pred_W_{name}"] = np.round(y_b, 3)
        if y_true is not None:
            data[f"residual_W_{name}"] = np.round(y_b - y_true, 3)

    if extra_cols:
        for col, vals in extra_cols.items():
            data[col] = vals

    return pd.DataFrame(data)


def cross_model_consistency(main_pred: np.ndarray,
                            baseline_results: dict,
                            on_thr_w: float = None) -> pd.DataFrame:
    """
    无标签场景: 计算主模型 vs 每个基线的"一致性指标"
    (互相 MAE / 相关系数 / 状态一致率)

    [v13.5 bug 修复] on_thr_w 参数化 (原硬编码 10W)
    若不传 -> 从 common.ON_THR_W 兜底 (维持旧行为)
    调用方 (05_inference.py) 应传 bundle 里的 ON_THR 保证一致
    """
    rows = []
    # [v13.5] 阈值优先级: 传入 > common.ON_THR_W (兜底)
    if on_thr_w is None:
        from common import ON_THR_W as _default_thr
        on_thr_w = _default_thr
    for name, info in baseline_results.items():
        y_b = info["y_pred"]
        mae = float(np.abs(main_pred - y_b).mean())
        corr = float(np.corrcoef(main_pred, y_b)[0, 1]) if y_b.std() > 0 else float("nan")
        # [v13.5] 状态一致率 (阈值从 bundle 传入, 与主评估一致)
        s_main = (main_pred >= on_thr_w).astype(int)
        s_b    = (y_b       >= on_thr_w).astype(int)
        state_agree = float((s_main == s_b).mean())
        rows.append({
            "baseline": name,
            "kind": info["model"].kind,
            "MAE_vs_main_W": round(mae, 2),
            "Pearson_vs_main": round(corr, 4),
            "state_agree_rate": round(state_agree, 4),
            "n_samples": int(len(main_pred)),
        })
    return pd.DataFrame(rows)
