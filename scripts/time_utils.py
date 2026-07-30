# -*- coding: utf-8 -*-
"""
时间解析工具 (统一处理多种时间格式)

支持的输入格式 (兼容电表/分路设备/手工导出的常见变体):
    2026/3/18 0:00:00       (斜杠 + 单位数月日/小时, 无前导零)
    2026/03/18 00:00:00     (斜杠 + 两位数, 标准)
    2026-3-18 0:00:00       (短横线 + 单位数, 数据库导出常见)
    2026-03-18 00:00:00     (ISO 标准)
    2026-03-18T00:00:00     (ISO 带 T 分隔符)
    2026/3/18               (仅日期)
    2026-03-18              (ISO 仅日期)

设计:
    1. 先用正则把日期分隔符统一为 "/"  (一次性向量化, 速度快)
    2. 优先尝试两种最常见 format 解析 (errors='coerce' 失败转 NaT)
    3. 残余 NaT 行回退到 pandas dateutil 自动推断 (慢但兜底全部格式)
"""
from __future__ import annotations
import pandas as pd
import re


# 一次编译, 反复使用
_DASH_TO_SLASH = re.compile(r"-")


def parse_timestamps(series: pd.Series | list, errors: str = "coerce",
                     logger=None) -> pd.Series:
    """
    将一列字符串时间统一解析为 pandas Timestamp.

    参数:
        series:  时间字符串序列 (pd.Series / list / array)
        errors:  'coerce' 失败置 NaT, 'raise' 抛异常 (默认 coerce)
        logger:  可选日志器, 输出失败行数

    返回:
        pd.Series of datetime64[ns]
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    # 步骤 1: 字符串预清洗 (向量化, 速度快)
    s = series.astype(str).str.strip()
    # 把 '-' 替换为 '/', 把 'T' 替换为 ' '
    # 例: '2026-3-18T0:00:00' -> '2026/3/18 0:00:00'
    s_norm = s.str.replace("-", "/", regex=False) \
              .str.replace("T", " ", regex=False)

    # 步骤 2: 优先尝试两种最常见 format (按出现频率排序)
    candidate_formats = [
        "%Y/%m/%d %H:%M:%S",   # 标准带秒 (覆盖 90%+ 行)
        "%Y/%m/%d %H:%M",      # 不含秒
        "%Y/%m/%d",            # 仅日期
    ]

    # 用 errors='coerce' 多轮尝试 (每轮只处理还未成功的)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = pd.Series(True, index=series.index)

    for fmt in candidate_formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s_norm[remaining], format=fmt, errors="coerce")
        ok = parsed.notna()
        result.loc[remaining[remaining].index[ok]] = parsed[ok]
        remaining.loc[remaining[remaining].index[ok]] = False

    # 步骤 3: 残余行回退到 dateutil 自动推断 (慢但鲁棒)
    if remaining.any():
        n_remain = int(remaining.sum())
        if logger:
            logger.debug(f"  [time_parse] {n_remain} 行未匹配预定义 format, "
                         f"回退到 dateutil 自动推断")
        try:
            parsed = pd.to_datetime(s_norm[remaining], errors="coerce",
                                    format="mixed")
        except (TypeError, ValueError):
            # 旧版 pandas 不支持 format='mixed', 退回最朴素调用
            parsed = pd.to_datetime(s_norm[remaining], errors="coerce")
        result.loc[remaining] = parsed

    # 步骤 4: 错误处理
    n_fail = int(result.isna().sum() - series.isna().sum())
    if n_fail > 0:
        if errors == "raise":
            sample = s[result.isna() & series.notna()].head(3).tolist()
            raise ValueError(f"时间解析失败 {n_fail} 行, 示例: {sample}")
        elif logger:
            sample = s[result.isna() & series.notna()].head(3).tolist()
            logger.warning(f"  [time_parse] 共 {n_fail} 行解析失败, "
                           f"已置为 NaT, 示例: {sample}")
    return result


def format_timestamp(ts) -> str:
    """
    统一输出格式 (用于 CSV 写出, 与电表上送规约一致)
    """
    if pd.isna(ts):
        return ""
    return pd.Timestamp(ts).strftime("%Y/%m/%d %H:%M:%S")


# ============================================================
# CLI 自检
# ============================================================
if __name__ == "__main__":
    test_cases = [
        "2026/3/18 0:00:00",
        "2026/03/18 00:00:00",
        "2026-3-18 0:00:00",
        "2026-03-18 00:00:00",
        "2026-03-18T00:00:00",
        "2026/3/18",
        "2026-03-18",
        "invalid_garbage",
    ]
    print("自检: 测试 8 种时间字符串格式解析")
    print("-" * 60)
    result = parse_timestamps(test_cases)
    for orig, parsed in zip(test_cases, result):
        flag = "✓" if pd.notna(parsed) else "⚠️"
        print(f"  {flag}  {orig:<28} -> {parsed}")
