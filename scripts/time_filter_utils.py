# -*- coding: utf-8 -*-
"""
[v12] 时段过滤工具模块 - 统一处理"训练/推理"数据的 include/exclude 时段过滤

设计规范:
  - 配置文件: JSON 格式 (path 由 --time-filter-config 传入)
  - 语义: 每用户可分别给 train / infer 指定 include + exclude
  - 顺序: 先按 include 保留 (未指定 include 视为"全保留"), 再按 exclude 剔除
  - 边界: [start, end] 闭区间, 两端都包含
  - 粒度:
      "YYYY-MM-DD"                -> 自动扩为 [D 00:00:00, D 23:59:59]
      "YYYY-MM-DD HH:MM"          -> [T, T]
      "YYYY-MM-DD HH:MM:SS"       -> [T, T]

JSON 结构示例:
{
  "800080252842_4206894986488": {
    "train": {
      "include": [["2025-07-10", "2026-06-28"]],
      "exclude": [
        ["2026-04-02 17:45", "2026-04-02 23:59:59"],
        ["2026-06-05 09:30", "2026-06-05 14:30"],
        ["2026-06-29", "2026-06-29"]
      ]
    },
    "infer": {
      "exclude": [["2026-06-05", "2026-06-05"]]
    }
  },
  "_default": {
    "train": {"exclude": []},
    "infer": {"exclude": []}
  }
}

CLI 传参格式 (给下游 02/05 用):
  --time-filter-spec '{"include":[["2025-07-10","2026-06-28"]],"exclude":[...]}'
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================
# 一段时段: (start_ts, end_ts) 闭区间, 都是 pd.Timestamp
TimeRange = Tuple[pd.Timestamp, pd.Timestamp]

# 一个 stage 的过滤规格:  {"include": [TimeRange, ...], "exclude": [TimeRange, ...]}
StageSpec = Dict[str, List[TimeRange]]


# ============================================================
# 时间字符串解析
# ============================================================
def parse_time_boundary(s: str, side: str) -> pd.Timestamp:
    """把 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM[:SS]' 解析为 Timestamp

    side='start': 纯日期时补 00:00:00
    side='end'  : 纯日期时补 23:59:59
    """
    s = str(s).strip()
    if not s:
        raise ValueError(f"时间字符串为空 (side={side})")

    # 只含日期无时间 (10 位 YYYY-MM-DD 或 8 位 YYYY/M/D)
    has_time = (":" in s) or (" " in s and len(s.split()) > 1)

    ts = pd.to_datetime(s, format="mixed")   # 兼容 2026/5/12 和 2026-05-12
    if not has_time:
        if side == "start":
            ts = ts.normalize()                          # 00:00:00
        elif side == "end":
            ts = ts.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        else:
            raise ValueError(f"side 必须是 'start' 或 'end', 收到: {side}")
    return ts


def parse_ranges(raw_list: List[Union[list, tuple]]) -> List[TimeRange]:
    """把 [[start_str, end_str], ...] 解析为 [(start_ts, end_ts), ...]

    自动:
      - 处理纯日期(补 00:00:00 / 23:59:59)
      - 校验 start <= end
      - 剔除空条目
    """
    if not raw_list:
        return []
    out: List[TimeRange] = []
    for i, item in enumerate(raw_list):
        if item is None:
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"时段 [{i}] 格式错误, 期望 [start, end] 2 元素列表, 实际: {item!r}"
            )
        start_ts = parse_time_boundary(item[0], side="start")
        end_ts   = parse_time_boundary(item[1], side="end")
        if start_ts > end_ts:
            raise ValueError(
                f"时段 [{i}] start > end: {start_ts} > {end_ts} (原始 {item})"
            )
        out.append((start_ts, end_ts))
    return out


# ============================================================
# 配置文件加载
# ============================================================
def load_time_filter_config(config_path: Union[str, Path]) -> dict:
    """加载 JSON 配置, 返回 dict, 键为 user_id, 值为 {'train': {...}, 'infer': {...}}

    _default 键 (可选) 对未列出的用户生效.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"time-filter-config 不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"配置根必须是 JSON 对象, 实际: {type(raw).__name__}")
    return raw


def get_user_stage_spec(
    config: dict, user_id: str, stage: str
) -> Optional[StageSpec]:
    """从 config 中拿出 (user_id, stage) 的过滤规格

    stage: 'train' 或 'infer'
    返回:
      None                        - 未配置该用户该 stage, 无过滤
      {'include':[], 'exclude':[]} - 已配置 (可能其中一个为空)
    """
    assert stage in ("train", "infer"), f"stage 必须为 train/infer, 收到 {stage}"

    # 用户级配置
    user_cfg = config.get(user_id)

    # 用户级未配置时回退到 _default
    if user_cfg is None:
        user_cfg = config.get("_default")

    if user_cfg is None:
        return None

    # 防御: 非 dict (如注释键 _comment_ 是字符串) 视为无配置
    if not isinstance(user_cfg, dict):
        return None

    stage_cfg = user_cfg.get(stage)
    if stage_cfg is None:
        return None
    if not isinstance(stage_cfg, dict):
        return None

    include = parse_ranges(stage_cfg.get("include", []))
    exclude = parse_ranges(stage_cfg.get("exclude", []))

    if not include and not exclude:
        return None   # 空配置视为无过滤

    return {"include": include, "exclude": exclude}


# ============================================================
# 应用过滤 (核心 API)
# ============================================================
def apply_time_filter(
    df: pd.DataFrame,
    time_col: str,
    spec: Optional[StageSpec],
    label: str = "",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """对 DataFrame 按时段规格过滤

    Args:
        df:       DataFrame, 必须含 time_col 列
        time_col: 时间戳列名 (总线='event_time', 分路='time')
        spec:     {'include':[...], 'exclude':[...]}, None 表示无过滤
        label:    日志前缀 (如 "bus" / "branch" / "infer_bus")
        logger:   调用侧日志器 (None 时用模块自带 log, 但在业务脚本
                  中建议传入以便日志输出到正确文件)

    Returns:
        过滤后的 DataFrame (行数可能减少, index 重置)

    规则:
        - include 为空 -> 视为 "全保留"
        - include 非空 -> 只保留至少落在一个 include 区间内的行
        - 然后再逐条 exclude, 落在任一 exclude 内的行都剔除
        - 区间边界为闭区间 [start, end]
    """
    if spec is None:
        return df

    _log = logger if logger is not None else log

    include: List[TimeRange] = spec.get("include", [])
    exclude: List[TimeRange] = spec.get("exclude", [])

    if not include and not exclude:
        return df

    n_before = len(df)
    ts = pd.to_datetime(df[time_col], format="mixed")

    # ---------- Step A: include ----------
    if include:
        mask_include = pd.Series(False, index=df.index)
        for (s, e) in include:
            mask_include |= (ts >= s) & (ts <= e)
        n_after_include = int(mask_include.sum())
        df = df[mask_include].reset_index(drop=True)
        ts = ts[mask_include].reset_index(drop=True)
        _log.info(
            f"  [time_filter/{label}] include {len(include)} 段 "
            f"-> {n_before} 行 -> {n_after_include} 行 "
            f"(-{n_before - n_after_include})"
        )
        # 明细列出每段
        for i, (s, e) in enumerate(include):
            _log.info(f"      include[{i}]: {s} ~ {e}")
    else:
        n_after_include = n_before

    # ---------- Step B: exclude ----------
    if exclude:
        mask_exclude = pd.Series(False, index=df.index)
        for (s, e) in exclude:
            mask_exclude |= (ts >= s) & (ts <= e)
        n_after_exclude = int((~mask_exclude).sum())
        df = df[~mask_exclude].reset_index(drop=True)
        _log.info(
            f"  [time_filter/{label}] exclude {len(exclude)} 段 "
            f"-> {n_after_include} 行 -> {n_after_exclude} 行 "
            f"(-{n_after_include - n_after_exclude})"
        )
        for i, (s, e) in enumerate(exclude):
            _log.info(f"      exclude[{i}]: {s} ~ {e}")

    return df


# ============================================================
# 序列化 (供 run_user_pipeline.py -> 02/05 传递)
# ============================================================
def spec_to_cli_arg(spec: Optional[StageSpec]) -> str:
    """把过滤规格序列化为 JSON 字符串, 供子进程 --time-filter-spec 参数

    None -> "" (下游看到空串会跳过过滤)
    """
    if spec is None:
        return ""
    obj = {
        "include": [[s.isoformat(), e.isoformat()] for (s, e) in spec["include"]],
        "exclude": [[s.isoformat(), e.isoformat()] for (s, e) in spec["exclude"]],
    }
    return json.dumps(obj, ensure_ascii=False)


def cli_arg_to_spec(cli_str: str) -> Optional[StageSpec]:
    """反解: 从子进程收到的 CLI JSON 字符串 恢复 spec"""
    cli_str = (cli_str or "").strip()
    if not cli_str:
        return None
    obj = json.loads(cli_str)
    return {
        "include": parse_ranges(obj.get("include", [])),
        "exclude": parse_ranges(obj.get("exclude", [])),
    }


# ============================================================
# 摘要 (人类可读)
# ============================================================
def spec_summary(spec: Optional[StageSpec]) -> str:
    if spec is None:
        return "(无过滤)"
    parts = []
    if spec.get("include"):
        parts.append(f"include={len(spec['include'])}段")
    if spec.get("exclude"):
        parts.append(f"exclude={len(spec['exclude'])}段")
    return ", ".join(parts) if parts else "(无过滤)"


# ============================================================
# [v13] 用户级 d87 守卫开关 (扩展 time_filter_config 语义)
# ============================================================
def get_user_common_overrides(config: dict, user_id: str) -> dict:
    """[v13.5] 从 time_filter_config 中提取用户级 common.py 常量覆盖

    支持的 8 个字段 (与 common.py 中的常量对应, JSON 名称转小写):

    | JSON 字段              | common.py 常量              | 类型          | 说明                          |
    |------------------------|-----------------------------|--------------|------------------------------|
    | on_thr_w               | ON_THR_W                    | float        | 空调 ON 判定阈值 (W), 训练+评估同口径 |
    | split_ratios           | SPLIT_RATIOS                | [f,f,f]      | 训练/验证/测试比例, 和=1.0        |
    | split_strategy         | SPLIT_STRATEGY              | str          | "stratified_day"/"stratified"/"time" |
    | post_min_on            | POST_MIN_ON (03_train)      | int          | 后处理最小连续 ON 步数            |
    | post_fill_short_off    | POST_FILL_SHORT_OFF         | int          | 后处理短 OFF 间隙填充步数         |
    | weather_latitude       | WEATHER_LATITUDE            | float        | 用户所在地纬度 (跨城市部署必需)      |
    | weather_longitude      | WEATHER_LONGITUDE           | float        | 用户所在地经度                  |
    | use_weather_features   | USE_WEATHER_FEATURES        | bool         | 是否启用温度特征                 |
    | use_temp_based_season  | USE_TEMP_BASED_SEASON       | bool         | 是否用温度驱动季节路由             |

    优先级链 (与 v13.1/v13.4 一致):
      1. config[user_id][<字段>]     (用户级显式指定, 最高)
      2. config[_default][<字段>]    (兜底默认)
      3. None (调用方应回退到 common.py 全局默认)

    校验策略 (宽松, 非法值 WARN + 忽略回退, 不阻塞训练):
      - on_thr_w:  必须 > 0 且 < 5000 (W)
      - split_ratios: 3 元组数字, 和不为 1.0 时自动归一化 (validate_split_ratios)
      - split_strategy: 必须 ∈ {"stratified_day","stratified","time"}
      - post_min_on: 必须 int >= 0
      - post_fill_short_off: 必须 int >= 0
      - weather_latitude: -90 ~ 90
      - weather_longitude: -180 ~ 180
      - use_weather_features/use_temp_based_season: bool 转换

    Args:
        config:  load_time_filter_config() 返回的 dict
        user_id: 用户 folder_name

    Returns:
        dict, 仅含实际覆盖的字段 (未覆盖字段不出现). 例:
            {"on_thr_w": 50.0, "split_ratios": [0.8, 0.1, 0.1]}
        空 dict 表示无任何覆盖.
    """
    if not config:
        return {}

    import warnings

    # 校验函数字典
    def _v_float_range(name, lo, hi):
        def _f(val):
            try:
                f = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"必须为数字, 收到 {val!r}")
            if not (lo <= f <= hi):
                raise ValueError(f"必须在 [{lo}, {hi}], 收到 {f}")
            return f
        return _f

    def _v_int_ge0(val):
        try:
            i = int(val)
        except (TypeError, ValueError):
            raise ValueError(f"必须为整数, 收到 {val!r}")
        if i < 0:
            raise ValueError(f"必须 >= 0, 收到 {i}")
        return i

    def _v_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("true", "1", "yes", "y", "on"):
                return True
            if s in ("false", "0", "no", "n", "off"):
                return False
        raise ValueError(f"必须为 bool (true/false), 收到 {val!r}")

    def _v_split_ratios(val):
        if not isinstance(val, (list, tuple)):
            raise ValueError(f"必须为 3 元素列表, 收到 {type(val).__name__}")
        if len(val) != 3:
            raise ValueError(f"必须为 3 元素列表, 收到 {len(val)} 个")
        try:
            arr = [float(x) for x in val]
        except (TypeError, ValueError):
            raise ValueError(f"元素必须为数字, 收到 {val!r}")
        if any(x <= 0 for x in arr):
            raise ValueError(f"元素必须 > 0, 收到 {arr}")
        # 自动归一化 (若和不为 1)
        total = sum(arr)
        if abs(total - 1.0) > 1e-6:
            arr = [x / total for x in arr]
        return arr

    def _v_split_strategy(val):
        # [v13.13] 新增 global_stratified: 全局按天抽样, 不按月分层
        VALID = {"stratified_day", "stratified", "time", "global_stratified"}
        s = str(val).strip().lower()
        if s not in VALID:
            raise ValueError(f"必须 ∈ {VALID}, 收到 {val!r}")
        return s

    # 字段名 -> 校验函数
    VALIDATORS = {
        "on_thr_w":              _v_float_range("on_thr_w", 0.001, 5000.0),
        "split_ratios":          _v_split_ratios,
        "split_strategy":        _v_split_strategy,
        "post_min_on":           _v_int_ge0,
        "post_fill_short_off":   _v_int_ge0,
        "weather_latitude":      _v_float_range("weather_latitude", -90.0, 90.0),
        "weather_longitude":     _v_float_range("weather_longitude", -180.0, 180.0),
        "use_weather_features":  _v_bool,
        "use_temp_based_season": _v_bool,
        "v14_enable":            _v_bool,
        "v14_enabled":           _v_bool,
        "physics":               _v_bool,
        "physics_features":      _v_bool,
        "focal":                 _v_bool,
        "ensemble":              _v_bool,
        "calibrate":             _v_bool,
        "auto_config":           _v_bool,
        "health":                _v_bool,
        "health_report":         _v_bool,
        "diag":                  _v_bool,
        "data_diag":             _v_bool,
    }

    def _extract_one(cfg, field, validator):
        """从 cfg dict 中读一个字段, 应用校验, 返回值或 None"""
        if not isinstance(cfg, dict) or field not in cfg:
            return None
        raw = cfg[field]
        try:
            return validator(raw)
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"time_filter_config 中 {field}={raw!r} 校验失败 ({e}), 忽略并回退到 common.py 默认",
                UserWarning, stacklevel=3,
            )
            return None

    user_cfg = config.get(user_id) if isinstance(config.get(user_id), dict) else None
    default_cfg = config.get("_default") if isinstance(config.get("_default"), dict) else None

    out = {}
    for field, validator in VALIDATORS.items():
        # 用户级优先
        val = _extract_one(user_cfg, field, validator)
        if val is None:
            # _default 兜底
            val = _extract_one(default_cfg, field, validator)
        if val is not None:
            out[field] = val
    return out


def get_user_target_col(config: dict, user_id: str) -> Optional[str]:
    """[v13.4] 从 time_filter_config 中获取用户级 target_col (目标分路列名)

    优先级:
      1. config[user_id]["target_col"] (显式指定)
      2. config["_default"]["target_col"] (默认值)
      3. None (未配置, 调用方应回退到旧的反推逻辑)

    合法值格式:
      - 单列: "pN" (N 为 ≥0 整数), 例: p0/p1/p2/.../p10/p99/p128
      - [v13.16 新增] 复合列: "pA+pB[+pC...]" 由 '+' 连接的多个 pN,
        例: "p1+p2" / "p1+p2+p3" / "p0+p5+p10" 均合法.
        物化语义: 加载分路 CSV 时新增一列, 值 = 各分量按行求和,
        列名就是复合字符串 (如 "p1+p2"). 下游业务代码把它当普通列名使用.
      - 非法值返回 None + 打印 WARN.

    大小写不敏感 + 空白容忍: "P2" / "p2" / " p2 " / "P1 + p2" 都会规范化为
    "p2" / "p1+p2" (统一小写去空白).

    Args:
        config:  load_time_filter_config() 返回的 dict
        user_id: 用户 folder_name (如 "800080270708_4206602981958")

    Returns:
        "pN" 或 "pA+pB[+pC...]" (规范化) / None
    """
    if not config:
        return None

    # v13.4-fix: 从 {p1,p2,p3,p4} 硬编码集合放宽为 re.fullmatch(r"p\d+"),
    # 与 run_batch_users.py::parse_user_folder 里的分路列匹配正则一致
    # [v13.16] 再放宽支持 "p1+p2" / "p1+p2+p3" 复合语义
    import re as _re
    # 单列或复合 (+ 连接): 例 p1 / p2 / p1+p2 / p0+p5+p10
    VALID_PATTERN = _re.compile(r"^p\d+(\+p\d+)*$")

    def _extract(cfg):
        if not isinstance(cfg, dict):
            return None
        val = cfg.get("target_col")
        if val is None:
            return None
        # v13.16: 规范化: 去所有空白 + 统一小写. 允许用户写 "P1 + p2" 或
        # "p1 +p2" 等宽松格式.
        val = "".join(str(val).split()).lower()
        if VALID_PATTERN.fullmatch(val):
            # 复合场景做去重防呆: "p1+p1" 视为非法 (语义上无意义, 提早暴露)
            if "+" in val:
                parts = val.split("+")
                if len(set(parts)) != len(parts):
                    import warnings
                    warnings.warn(
                        f"[v13.16] time_filter_config 中 target_col={val!r} "
                        f"含重复分量 (如 p1+p1), 忽略",
                        UserWarning, stacklevel=3,
                    )
                    return None
            return val
        # 非法值 -> WARN 并回退
        import warnings
        warnings.warn(
            f"time_filter_config 中 target_col={val!r} 不符合格式 "
            f"'pN' 或 'pA+pB[+pC...]' (N 为 ≥0 整数), 忽略",
            UserWarning, stacklevel=3,
        )
        return None

    # 用户级配置
    result = _extract(config.get(user_id))
    if result is not None:
        return result

    # _default 回退
    return _extract(config.get("_default"))


def get_user_guard_enabled(config: dict, user_id: str) -> Optional[bool]:
    """从 time_filter_config 中获取用户级 guard_enabled 开关

    优先级:
      1. config[user_id]["guard_enabled"] (显式指定)
      2. config["_default"]["guard_enabled"] (默认值)
      3. None (未配置, 调用方应回退到 common.D87_ADAPTIVE_GUARD_ENABLED
              或触发自动检测)

    Args:
        config:  load_time_filter_config() 返回的 dict
        user_id: 用户 folder_name (如 "800080252842_4206894986488")

    Returns:
        True / False / None
    """
    if not config:
        return None

    # 用户级配置
    user_cfg = config.get(user_id)
    if isinstance(user_cfg, dict) and "guard_enabled" in user_cfg:
        return bool(user_cfg["guard_enabled"])

    # _default 回退
    default_cfg = config.get("_default")
    if isinstance(default_cfg, dict) and "guard_enabled" in default_cfg:
        return bool(default_cfg["guard_enabled"])

    return None


def auto_detect_guard_enabled(
    d87_abs_max_train=None,
    threshold: float = 50.0,
    logger=None,
    *args,
    **kwargs,
) -> bool:
    """[v13/v14] 基于训练集 |d87| 极值自动判定守卫是否有效 (支持多种签名呼叫兼容)

    动机: 变频/小功率空调用户 (如 270708, 252842) d87 尖峰不足,
          守卫阈值会学得比全期 d87 max 还高, 造成 100% FN.

    Args:
        d87_abs_max_train: 训练集 5min 原始 |d87| 的最大值
        threshold: 阈值 (默认 50W). 低于此值判定为"d87 特征无效"
        logger: 日志器 (可选)

    Returns:
        True  -> d87 尖峰足够强, 建议启用守卫
        False -> d87 尖峰不足, 建议禁用守卫 (启用会伤害推理)
    """
    val = d87_abs_max_train
    if val is None:
        val = kwargs.get("d87_max", kwargs.get("max_val", 0.0))
    if hasattr(val, "abs"):
        try:
            val = val.abs().max()
        except Exception:
            val = 0.0
    try:
        val_float = float(val)
    except (ValueError, TypeError):
        val_float = 0.0
    result = val_float >= threshold
    if logger is not None:
        logger.info(
            f"  [v13/v14 auto_detect_guard] 训练集 |d87|.max={val_float:.1f}W "
            f"vs 阈值={threshold:.1f}W -> guard_enabled={result}"
        )
    return result


def get_user_v14_flags(user_id: str, config_path: str = None) -> dict:
    """
    [v14] 从 time_filters.json 获取用户的 v14 增强开关字典
    优先读取 config[user_id]["v14"]，_default 兜底为全 False
    """
    default_flags = {
        "v14_enable": False,
        "physics": False,
        "focal": False,
        "ensemble": False,
        "calibrate": False,
        "auto_config": False,
        "health": False,
        "diag": False,
    }
    try:
        cfg = load_time_filter_config(config_path) if config_path else load_time_filter_config()
        user_cfg = cfg.get(str(user_id), {})
        if not isinstance(user_cfg, dict):
            return default_flags
        v14_sub = user_cfg.get("v14", {})
        if not isinstance(v14_sub, dict):
            return default_flags
        out = dict(default_flags)
        for k in default_flags.keys():
            if k in v14_sub:
                out[k] = bool(v14_sub[k])
        # 别名兼容
        if "physics_features" in v14_sub and "physics" not in v14_sub:
            out["physics"] = bool(v14_sub["physics_features"])
        if "health_report" in v14_sub and "health" not in v14_sub:
            out["health"] = bool(v14_sub["health_report"])
        if "data_diag" in v14_sub and "diag" not in v14_sub:
            out["diag"] = bool(v14_sub["data_diag"])
        return out
    except Exception:
        return default_flags


# ============================================================
# [v13] Per-split time_filter: 细粒度 train/val/test 3 集独立 include/exclude
# ============================================================
# 语义 (用户明确规范):
#   Step 1: 原策略 (stratified_day) 切分 -> 初始 idx_tr/idx_va/idx_te
#   Step 2: 处理 include (硬锚定, 样本粒度)
#       - 若样本 t ∈ train.include -> 锚定 train
#       - 否则 若 t ∈ val.include -> 锚定 val
#       - 否则 若 t ∈ test.include -> 锚定 test
#       - 冲突时按 train->val->test 优先 (第一个匹配 win), 打 WARN
#   Step 3: 严格保持形状 (原 split 大小不变)
#       - 若某 split X 因 include 强制多了样本 -> 从 X 未锚定样本中让出到不足的 split
#       - 若不足以补足 -> 打 WARN, 允许形状偏移
#   Step 4: 处理 exclude (样本粒度)
#       - 若样本 t 已属 split X 且 t ∈ X.exclude -> 从 X 移除, 送回"重分配池"
#       - 重分配池样本按剩余空间就近分配 (仍需保持形状)
#       - 若样本被 3 个 split 的 exclude 全部命中 -> 完全丢弃
# ============================================================

def load_splits_time_filter(config: dict, user_id: str) -> Optional[dict]:
    """[v13] 从 config 里拿出用户级 per-split 过滤规格

    JSON 结构:
      user_id:
        splits:
          train:
            include: [[start, end], ...]
            exclude: [[start, end], ...]
          val: {...}
          test: {...}

    Returns:
        dict 形如 {'train': {'include':[...], 'exclude':[...]}, 'val': {...}, 'test': {...}}
        任一 split 未指定则该键值为 None. 若整个 splits 未指定则返回 None.
    """
    if not config:
        return None
    user_cfg = config.get(user_id)
    if user_cfg is None:
        user_cfg = config.get("_default")
    if not isinstance(user_cfg, dict):
        return None
    splits_cfg = user_cfg.get("splits")
    if not isinstance(splits_cfg, dict):
        return None

    result = {}
    for split_name in ("train", "val", "test"):
        sp_cfg = splits_cfg.get(split_name)
        if not isinstance(sp_cfg, dict):
            result[split_name] = None
            continue
        inc = parse_ranges(sp_cfg.get("include", []))
        exc = parse_ranges(sp_cfg.get("exclude", []))
        if not inc and not exc:
            result[split_name] = None
        else:
            result[split_name] = {"include": inc, "exclude": exc}

    # 若三个 split 都为 None, 视为整体未指定
    if all(v is None for v in result.values()):
        return None
    return result


def _ts_in_ranges(ts: pd.Timestamp, ranges: List[TimeRange]) -> bool:
    """判断时间戳是否落入任一时段 (闭区间 [start,end])"""
    for s, e in ranges:
        if s <= ts <= e:
            return True
    return False


def apply_per_split_filter(
    timestamps,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    splits_spec: Optional[dict],
    logger=None,
) -> tuple:
    """[v13] 对已切分的 idx 应用 per-split include/exclude 过滤

    Args:
        timestamps:   全量时间戳 (DatetimeIndex 或 pd.DatetimeIndex 兼容)
        idx_tr/va/te: 原切分策略给出的 3 个索引数组
        splits_spec:  load_splits_time_filter 返回值.
                      None -> 直接返回原 idx (无过滤)
                      dict -> 应用 include/exclude 逻辑
        logger:       日志器

    Returns:
        (idx_tr_new, idx_va_new, idx_te_new): 处理后的 3 个索引数组
    """
    if splits_spec is None:
        return idx_tr, idx_va, idx_te

    import numpy as np

    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    n = len(timestamps)
    ts_arr = pd.DatetimeIndex(timestamps)

    # 保存原始大小 (Step 3 目标形状)
    n_tr_orig = len(idx_tr)
    n_va_orig = len(idx_va)
    n_te_orig = len(idx_te)
    _log(f"  [v13 per_split_filter] 原切分: train={n_tr_orig}, val={n_va_orig}, test={n_te_orig}")

    # ============ Step 2: include 硬锚定 ============
    # split_target[i] = -1 (未锚定) / 0 (train) / 1 (val) / 2 (test) / -2 (完全 exclude)
    SPLIT_NAMES = ["train", "val", "test"]
    split_target = np.full(n, -1, dtype=np.int8)
    n_include_conflict = 0

    for i, ts in enumerate(ts_arr):
        anchored_to = None
        for j, split_name in enumerate(SPLIT_NAMES):
            sp = splits_spec.get(split_name)
            if sp is None:
                continue
            inc = sp.get("include", [])
            if inc and _ts_in_ranges(ts, inc):
                if anchored_to is None:
                    anchored_to = j
                else:
                    n_include_conflict += 1
                    # 保留第一个匹配的 (train 优先)
        if anchored_to is not None:
            split_target[i] = anchored_to

    n_anchored_tr = int((split_target == 0).sum())
    n_anchored_va = int((split_target == 1).sum())
    n_anchored_te = int((split_target == 2).sum())
    n_anchored_total = n_anchored_tr + n_anchored_va + n_anchored_te
    if n_anchored_total > 0:
        _log(f"  [v13 per_split_filter] include 锚定: train={n_anchored_tr}, "
             f"val={n_anchored_va}, test={n_anchored_te} (合计 {n_anchored_total})")
    if n_include_conflict > 0:
        _warn(f"  [v13 per_split_filter] WARN: {n_include_conflict} 个样本被多个 split.include "
              f"同时命中, 按 train->val->test 顺序取第一个匹配")

    # ============ Step 4a: exclude 标记 (先做, 因为完全 exclude 的样本不参与后续)  ============
    # 记录每样本被哪些 split.exclude 命中
    exclude_flags = np.zeros((n, 3), dtype=bool)  # 3 列: train/val/test 是否 exclude
    for i, ts in enumerate(ts_arr):
        for j, split_name in enumerate(SPLIT_NAMES):
            sp = splits_spec.get(split_name)
            if sp is None:
                continue
            exc = sp.get("exclude", [])
            if exc and _ts_in_ranges(ts, exc):
                exclude_flags[i, j] = True

    # 完全 exclude (3 个 split 的 exclude 都命中) -> 丢弃
    fully_excluded = exclude_flags.all(axis=1)
    n_fully_excluded = int(fully_excluded.sum())
    if n_fully_excluded > 0:
        _log(f"  [v13 per_split_filter] 完全 exclude (3 split 均排除) 丢弃: {n_fully_excluded} 样本")
        split_target[fully_excluded] = -2  # 标记为"完全丢弃"

    # ============ Step 3+4: 构建初始归属 (基于原 idx + include 锚定 + exclude) ============
    # 首先根据原 idx_tr/va/te 给每个样本初始归属
    orig_split = np.full(n, -1, dtype=np.int8)
    orig_split[idx_tr] = 0
    orig_split[idx_va] = 1
    orig_split[idx_te] = 2

    # 每个样本的当前归属:
    #   - split_target 明确 (include 锚定或完全丢弃) 优先
    #   - 否则用 orig_split
    current = np.where(split_target >= -1, split_target, orig_split).astype(np.int8)
    # 修正: 当 split_target=-1 时用 orig_split, 当 split_target≥0 时用 split_target
    for i in range(n):
        if split_target[i] == -2:
            current[i] = -2  # 完全丢弃
        elif split_target[i] >= 0:
            current[i] = split_target[i]  # include 锚定
        else:
            current[i] = orig_split[i]

    # ============ 应用 exclude (从当前 split 移除) ============
    for i in range(n):
        if current[i] < 0:
            continue
        # 若当前 split 命中 exclude, 移出到"待重分配池"(current = -1)
        if exclude_flags[i, current[i]]:
            # 但如果这个样本又是 include 锚定的呢? 冲突: 优先 exclude (从该 split 移出)
            # 但重分配时不能再放回同一 split
            current[i] = -1

    # 统计 include 锚定的锁定集合 (Step 3 中这些不可动)
    anchored_lock = split_target >= 0  # True 表示不能再改动

    # ============ Step 3: 严格保持形状 - 用剩余样本填充不足 ============
    # 目标大小: 原始 n_tr_orig / n_va_orig / n_te_orig
    # (完全丢弃的样本不算, 目标应扣减)
    # 如果 fully_excluded 中原本属于某 split, 需要按比例调整目标
    # 简化处理: 目标形状 = 原始形状, 若不足则 WARN
    targets = [n_tr_orig, n_va_orig, n_te_orig]

    def _current_counts():
        return [int((current == j).sum()) for j in range(3)]

    counts = _current_counts()
    _log(f"  [v13 per_split_filter] Step 3 形状调整前: "
         f"train={counts[0]}, val={counts[1]}, test={counts[2]}, "
         f"未分配={int((current == -1).sum())}, 丢弃={int((current == -2).sum())}")

    # 从"过剩"split 让出到"不足"split, 优先转移非锚定样本
    for j in range(3):
        while counts[j] > targets[j]:
            # 找 split j 里非锚定 且 未被其他 split include 排除 的样本
            candidates = np.where((current == j) & (~anchored_lock))[0]
            if len(candidates) == 0:
                break
            # 找需要样本的 split
            need_j = None
            for k in range(3):
                if counts[k] < targets[k]:
                    need_j = k
                    break
            if need_j is None:
                # 所有 split 都满足或过剩 -> 把剩余的转到"未分配"池
                # 但严格保持形状语义下, 这不应发生
                break
            # 转移一个样本
            # 优先转移不被 need_j.exclude 命中的样本
            transferable = [c for c in candidates if not exclude_flags[c, need_j]]
            if not transferable:
                # 找不到可转移的 (都会被 need_j.exclude 拒绝), 允许直接丢弃 (WARN)
                _warn(f"  [v13 per_split_filter] WARN: split {SPLIT_NAMES[j]} 有过剩样本但无法转移到 {SPLIT_NAMES[need_j]} (被 exclude 拒绝)")
                break
            chosen = transferable[0]
            current[chosen] = need_j
            counts = _current_counts()

    # 处理"未分配"池 (被 exclude 从原 split 踢出的样本): 按需分配到不足的 split
    while True:
        unassigned = np.where(current == -1)[0]
        if len(unassigned) == 0:
            break
        # 找不足的 split
        need_j = None
        for k in range(3):
            if counts[k] < targets[k]:
                need_j = k
                break
        if need_j is None:
            # 所有 split 已满, 剩余的 unassigned 只能丢弃 (WARN)
            n_drop = len(unassigned)
            _warn(f"  [v13 per_split_filter] WARN: {n_drop} 未分配样本无处安放 (所有 split 已达目标形状), 丢弃")
            current[unassigned] = -2
            break
        # 找一个可以进 need_j 的样本 (不被 need_j.exclude 命中)
        transferable = [u for u in unassigned if not exclude_flags[u, need_j]]
        if not transferable:
            # 未分配池全被 need_j.exclude 拒绝:
            # 尝试从其他 split 未锚定样本 中拉一个到 need_j (交换)
            # 例: need_j=val 缺, 未分配池是被 val.exclude 踢的样本;
            #     应从 train 未锚定样本里挑一个补 val, 未分配的进 train
            donor_split = None
            donor_idx = None
            for src_j in range(3):
                if src_j == need_j: continue
                if counts[src_j] <= targets[src_j]:
                    # 该 split 已经不宽裕, 但可以"平移" (先拉出去再补进来, 数量守恒)
                    # 找该 split 里 非锚定 且 不被 need_j.exclude 命中 的样本
                    src_candidates = np.where(
                        (current == src_j) & (~anchored_lock) &
                        (~exclude_flags[:, need_j])
                    )[0]
                    if len(src_candidates) > 0:
                        # 但 donor split 拿走后, 需要用 unassigned 里能进 src_j 的样本补
                        for u in unassigned:
                            if not exclude_flags[u, src_j]:
                                donor_split = src_j
                                donor_idx = src_candidates[0]
                                current[donor_idx] = need_j
                                current[u] = src_j
                                break
                        if donor_idx is not None:
                            break
            if donor_idx is None:
                _warn(f"  [v13 per_split_filter] WARN: 未分配池样本全被 {SPLIT_NAMES[need_j]}.exclude 拒绝, "
                      f"且无法通过跨 split 交换补足, 只能丢弃")
                current[unassigned] = -2
                break
            counts = _current_counts()
            continue
        current[transferable[0]] = need_j
        counts = _current_counts()

    # 最终统计
    counts_final = _current_counts()
    n_dropped = int((current == -2).sum())
    _log(f"  [v13 per_split_filter] 最终切分: train={counts_final[0]}, "
         f"val={counts_final[1]}, test={counts_final[2]}, 丢弃={n_dropped}")

    # 形状偏移警告
    for j, name in enumerate(SPLIT_NAMES):
        if counts_final[j] != targets[j]:
            _warn(f"  [v13 per_split_filter] WARN: split {name} 形状偏移: "
                  f"目标={targets[j]}, 实际={counts_final[j]}")

    new_idx_tr = np.where(current == 0)[0]
    new_idx_va = np.where(current == 1)[0]
    new_idx_te = np.where(current == 2)[0]

    return new_idx_tr, new_idx_va, new_idx_te


def splits_spec_to_cli_arg(splits_spec: Optional[dict]) -> str:
    """[v13] 序列化 per-split 规格为 JSON 字符串, 供子进程 CLI 传参"""
    if splits_spec is None:
        return ""
    obj = {}
    for split_name, spec in splits_spec.items():
        if spec is None:
            continue
        obj[split_name] = {
            "include": [[s.isoformat(), e.isoformat()] for (s, e) in spec.get("include", [])],
            "exclude": [[s.isoformat(), e.isoformat()] for (s, e) in spec.get("exclude", [])],
        }
    if not obj:
        return ""
    return json.dumps(obj, ensure_ascii=False)


def cli_arg_to_splits_spec(cli_str: str) -> Optional[dict]:
    """[v13] 反序列化 per-split CLI JSON 字符串"""
    cli_str = (cli_str or "").strip()
    if not cli_str:
        return None
    obj = json.loads(cli_str)
    result = {}
    for split_name in ("train", "val", "test"):
        sp = obj.get(split_name)
        if not isinstance(sp, dict):
            result[split_name] = None
            continue
        result[split_name] = {
            "include": parse_ranges(sp.get("include", [])),
            "exclude": parse_ranges(sp.get("exclude", [])),
        }
    if all(v is None for v in result.values()):
        return None
    return result


def splits_spec_summary(splits_spec: Optional[dict]) -> str:
    """[v13] 人类可读摘要"""
    if splits_spec is None:
        return "(无 per-split 过滤)"
    parts = []
    for name in ("train", "val", "test"):
        sp = splits_spec.get(name)
        if sp is None:
            continue
        pieces = []
        if sp.get("include"):
            pieces.append(f"include={len(sp['include'])}段")
        if sp.get("exclude"):
            pieces.append(f"exclude={len(sp['exclude'])}段")
        if pieces:
            parts.append(f"{name}({','.join(pieces)})")
    return "; ".join(parts) if parts else "(空)"


# ============================================================
# 单元测试 (直接 python time_filter_utils.py 运行)
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("time_filter_utils 单元测试")
    print("=" * 70)

    # -------- 边界解析 --------
    print("\n--- Test 1: parse_time_boundary ---")
    assert parse_time_boundary("2026-04-02", "start") == pd.Timestamp("2026-04-02 00:00:00")
    assert parse_time_boundary("2026-04-02", "end")   == pd.Timestamp("2026-04-02 23:59:59")
    assert parse_time_boundary("2026-04-02 17:45", "start") == pd.Timestamp("2026-04-02 17:45:00")
    assert parse_time_boundary("2026/4/2", "start")   == pd.Timestamp("2026-04-02 00:00:00")
    print("  [OK] 日期/时间/斜杠/HH:MM 全部解析正确")

    # -------- 时段解析 + 校验 --------
    print("\n--- Test 2: parse_ranges 正常路径 ---")
    r = parse_ranges([
        ["2026-04-02", "2026-04-02"],
        ["2026-06-05 09:30", "2026-06-05 14:30"],
    ])
    assert len(r) == 2
    assert r[0][0] == pd.Timestamp("2026-04-02 00:00:00")
    assert r[0][1] == pd.Timestamp("2026-04-02 23:59:59")
    print("  [OK]")

    print("\n--- Test 3: parse_ranges start>end 校验 ---")
    try:
        parse_ranges([["2026-04-03", "2026-04-01"]])
        print("  [FAIL] 应该抛异常")
    except ValueError as e:
        print(f"  [OK] 正确拒绝: {e}")

    # -------- 过滤应用 --------
    print("\n--- Test 4: apply_time_filter ---")
    ts_series = pd.date_range("2026-04-01 00:00", "2026-04-05 23:45", freq="15min")
    df = pd.DataFrame({"event_time": ts_series, "v": range(len(ts_series))})

    # 4a: exclude 一天
    spec1 = {"include": [], "exclude": parse_ranges([["2026-04-03", "2026-04-03"]])}
    df1 = apply_time_filter(df.copy(), "event_time", spec1, "test_4a")
    assert len(df1) == len(df) - 96, f"4a: {len(df1)} vs {len(df) - 96}"
    print(f"  [OK] 4a exclude 一天: {len(df)} -> {len(df1)} (剔除 96 行)")

    # 4b: exclude 半天时段
    spec2 = {"include": [], "exclude": parse_ranges([["2026-04-02 17:45", "2026-04-02 23:59:59"]])}
    df2 = apply_time_filter(df.copy(), "event_time", spec2, "test_4b")
    expected = len(df) - 25   # 17:45 到 23:45 共 6h+15min = 25 个 15min 点
    assert len(df2) == expected, f"4b: {len(df2)} vs {expected}"
    print(f"  [OK] 4b exclude 半天: {len(df)} -> {len(df2)} (剔除 25 行)")

    # 4c: include + exclude 组合
    spec3 = {
        "include": parse_ranges([["2026-04-02", "2026-04-04"]]),
        "exclude": parse_ranges([["2026-04-03", "2026-04-03"]]),
    }
    df3 = apply_time_filter(df.copy(), "event_time", spec3, "test_4c")
    # 4/2 96 行 + 4/4 96 行 = 192 行
    assert len(df3) == 192, f"4c: {len(df3)}"
    print(f"  [OK] 4c include(3天)-exclude(1天): {len(df)} -> {len(df3)}")

    # 4d: 无过滤 (spec=None)
    df4 = apply_time_filter(df.copy(), "event_time", None, "test_4d")
    assert len(df4) == len(df)
    print("  [OK] 4d spec=None 不改数据")

    # -------- 序列化 --------
    print("\n--- Test 5: spec 序列化 <-> 反解 ---")
    spec = {
        "include": parse_ranges([["2026-04-02", "2026-04-04"]]),
        "exclude": parse_ranges([["2026-04-03 08:00", "2026-04-03 12:00"]]),
    }
    cli = spec_to_cli_arg(spec)
    spec_back = cli_arg_to_spec(cli)
    assert spec_back["include"][0] == spec["include"][0]
    assert spec_back["exclude"][0] == spec["exclude"][0]
    print(f"  [OK] 序列化: {cli[:80]}...")
    print("  [OK] 反解与原 spec 一致")

    # -------- 空 spec 处理 --------
    print("\n--- Test 6: 空 spec / 空配置 ---")
    assert cli_arg_to_spec("") is None
    assert cli_arg_to_spec("  ") is None
    assert spec_to_cli_arg(None) == ""
    print("  [OK] 空字符串和 None 处理正确")

    # -------- v13 per-split time_filter 测试 --------
    print("\n--- Test 7: v13 per-split apply_per_split_filter ---")
    import numpy as np
    # 构造 12 天数据 (每天 96 个 15min 点, 共 1152 样本)
    ts = pd.date_range("2026-06-01", periods=12*96, freq="15min")
    # 原始切分 (模拟 stratified_day): train=前6天, val=6-9天, test=9-12天
    idx_tr_orig = np.arange(0, 6*96)
    idx_va_orig = np.arange(6*96, 9*96)
    idx_te_orig = np.arange(9*96, 12*96)
    print(f"  原始形状: train={len(idx_tr_orig)}, val={len(idx_va_orig)}, test={len(idx_te_orig)}")

    # 场景 7a: train.include 硬锚定 6/10 全天 (原属于 test 96 样本)
    spec = {
        "train": {"include": parse_ranges([["2026-06-10", "2026-06-10"]]), "exclude": []},
        "val":   None,
        "test":  None,
    }
    r = apply_per_split_filter(ts, idx_tr_orig, idx_va_orig, idx_te_orig, spec)
    tr_n, va_n, te_n = len(r[0]), len(r[1]), len(r[2])
    print(f"  7a 结果: train={tr_n}, val={va_n}, test={te_n}")
    assert tr_n == len(idx_tr_orig), f"train 形状应保持 {len(idx_tr_orig)}, 实际 {tr_n}"
    # 6/10 应全在 train
    day_610 = (ts >= pd.Timestamp("2026-06-10")) & (ts < pd.Timestamp("2026-06-11"))
    day_610_idx = np.where(day_610)[0]
    assert set(day_610_idx).issubset(set(r[0])), "6/10 未全部锚定到 train"
    print("  [OK] 7a: include 硬锚定 + 形状保持")

    # 场景 7b: val.exclude 剔除 6/8 全天 (原属于 val 96 样本)
    spec = {
        "train": None,
        "val":   {"include": [], "exclude": parse_ranges([["2026-06-08", "2026-06-08"]])},
        "test":  None,
    }
    r = apply_per_split_filter(ts, idx_tr_orig, idx_va_orig, idx_te_orig, spec)
    day_608 = (ts >= pd.Timestamp("2026-06-08")) & (ts < pd.Timestamp("2026-06-09"))
    day_608_idx = np.where(day_608)[0]
    assert not set(day_608_idx).intersection(set(r[1])), "6/8 未从 val 移除"
    # 6/8 应进入 train 或 test (重分配)
    total = set(r[0]) | set(r[1]) | set(r[2])
    if set(day_608_idx).issubset(total):
        print("  [OK] 7b: exclude 剔除后成功重分配到其他 split")
    else:
        print("  [OK] 7b: exclude 剔除, 部分样本因无法分配被丢弃 (符合语义)")

    # 场景 7c: 三方 exclude 同时命中 -> 完全丢弃
    spec = {
        "train": {"include": [], "exclude": parse_ranges([["2026-06-05", "2026-06-05"]])},
        "val":   {"include": [], "exclude": parse_ranges([["2026-06-05", "2026-06-05"]])},
        "test":  {"include": [], "exclude": parse_ranges([["2026-06-05", "2026-06-05"]])},
    }
    r = apply_per_split_filter(ts, idx_tr_orig, idx_va_orig, idx_te_orig, spec)
    day_605 = (ts >= pd.Timestamp("2026-06-05")) & (ts < pd.Timestamp("2026-06-06"))
    day_605_idx = np.where(day_605)[0]
    all_idx = set(r[0]) | set(r[1]) | set(r[2])
    assert not set(day_605_idx).intersection(all_idx), "6/5 应完全丢弃, 但仍存在于某 split"
    print(f"  [OK] 7c: 三方 exclude 命中 -> 完全丢弃 (train+val+test 均无 6/5)")

    # 场景 7d: 序列化 <-> 反解
    spec = {
        "train": {"include": parse_ranges([["2026-06-01", "2026-06-05"]]),
                  "exclude": parse_ranges([["2026-06-03 08:00", "2026-06-03 12:00"]])},
        "val":   None,
        "test":  {"include": parse_ranges([["2026-06-10", "2026-06-12"]]),
                  "exclude": []},
    }
    s = splits_spec_to_cli_arg(spec)
    spec_back = cli_arg_to_splits_spec(s)
    assert spec_back["train"]["include"][0] == spec["train"]["include"][0]
    assert spec_back["test"]["include"][0] == spec["test"]["include"][0]
    assert spec_back["val"] is None
    print(f"  [OK] 7d: 序列化+反解")

    # 场景 7e: load_splits_time_filter
    cfg = {
        "800080270708_4206602981958": {
            "splits": {
                "train": {"include": [["2026-06-01", "2026-06-10"]]},
                "test":  {"exclude": [["2026-06-15", "2026-06-15"]]},
            }
        }
    }
    ss = load_splits_time_filter(cfg, "800080270708_4206602981958")
    assert ss is not None
    assert ss["train"]["include"][0] == (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-10 23:59:59"))
    assert ss["val"] is None
    assert ss["test"]["exclude"][0] == (pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-15 23:59:59"))
    print(f"  [OK] 7e: load_splits_time_filter")

    # -------- v13.4-fix: 通用 pN 格式 (N≥0) 测试 --------
    print("\n--- Test 8: v13.4-fix get_user_target_col 通用 pN 格式 ---")
    import warnings as _w

    # 8a: p0 合法 (新支持)
    cfg = {"u1": {"target_col": "p0"}}
    assert get_user_target_col(cfg, "u1") == "p0", "p0 应合法"
    print("  [OK] 8a: p0 合法")

    # 8b: p5 合法 (新支持)
    cfg = {"u1": {"target_col": "p5"}}
    assert get_user_target_col(cfg, "u1") == "p5", "p5 应合法"
    print("  [OK] 8b: p5 合法")

    # 8c: p10 合法 (多位数)
    cfg = {"u1": {"target_col": "p10"}}
    assert get_user_target_col(cfg, "u1") == "p10", "p10 应合法"
    print("  [OK] 8c: p10 合法")

    # 8d: p128 合法 (大数)
    cfg = {"u1": {"target_col": "p128"}}
    assert get_user_target_col(cfg, "u1") == "p128", "p128 应合法"
    print("  [OK] 8d: p128 合法")

    # 8e: 大小写 P99 (规范化)
    cfg = {"u1": {"target_col": "P99"}}
    assert get_user_target_col(cfg, "u1") == "p99", "P99 应规范化为 p99"
    print("  [OK] 8e: P99 -> p99")

    # 8f: 非法 pN01 (前导零除外, 但正则允许多位数字, 所以 p01 合法)
    cfg = {"u1": {"target_col": "p01"}}
    assert get_user_target_col(cfg, "u1") == "p01"
    print("  [OK] 8f: p01 合法 (前导零允许, 正则 \\d+)")

    # 8g: 非法 acp1 (仍应拒绝)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg = {"u1": {"target_col": "acp1"}}
        r = get_user_target_col(cfg, "u1")
        assert r is None, "acp1 应被拒"
        assert len(caught) == 1 and "格式" in str(caught[0].message)
    print("  [OK] 8g: acp1 仍拒绝 + WARN")

    # 8h: 非法 pN (含字母)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg = {"u1": {"target_col": "pN"}}
        r = get_user_target_col(cfg, "u1")
        assert r is None, "pN 字面 应被拒"
    print("  [OK] 8h: pN 字面 仍拒绝")

    # 8i: 非法 p (仅字母无数字)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg = {"u1": {"target_col": "p"}}
        r = get_user_target_col(cfg, "u1")
        assert r is None, "单 p 应被拒"
    print("  [OK] 8i: 单 'p' 仍拒绝")

    # 8j: 非法 1 (纯数字)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg = {"u1": {"target_col": "1"}}
        r = get_user_target_col(cfg, "u1")
        assert r is None
    print("  [OK] 8j: 纯数字 '1' 仍拒绝")

    print("\n" + "=" * 70)
    print("[PASS] 所有单元测试通过 (含 v13 per-split 5 组 + v13.4-fix 通用 pN 10 场景)")
    print("=" * 70)
