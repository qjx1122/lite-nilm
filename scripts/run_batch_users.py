# -*- coding: utf-8 -*-
"""
批量用户训练 + 评估 + 推理流水线 (v6.12.6+v6.15.0-graceful-v9)
=================================================================
扫描 data/trains/ 和 data/infers/ 取用户 ID 并集, 对每个有训练数据的用户:
  1. 自动解析 train/infer 总线 + 分路 CSV (按命名规范, 兼容 -1 / -infer 后缀)
  2. 自动反推 target_col (总线 -Ch{N}- 优先, 分路有则用, 否则退到分路第1个pN列)
  3. 调用 run_user_pipeline.py 完整跑 02->03->04->05
  4. 产物按新目录结构分流归档 (见下)

------ 新目录结构 (v9) ------
data/
  trains/<user_id>/                  训练用 csv (总线 + 分路)
  infers/<user_id>/                  推理用 csv (可选, 缺失则跳过 05)
models/
  <user_id>/                         所有模型 pkl + model_meta.json
logs/
  <user_id>/                         单用户运行日志
  _batch/                            批量执行日志
artifacts/
  summary_metrics_all_users.csv      [汇总] 每用户 4 行 (train/val/test/inference)
  batch_run_summary.csv              [汇总] 批量执行记录 (每次跑完覆盖)
  batch_execution_state.csv          [v13.17 断点续跑] 实时增量状态: 每个用户跑完立即
                                     追加/更新 (user_id, status, success, started_at,
                                     finished_at, duration_s, message, target_col, run_id).
                                     原子写 (.tmp + os.replace), 崩溃时最多丢当前正跑的.
                                     用 --resume 启用断点续跑 (默认跳过 ok/soft_skip;
                                     加 --resume-skip-failed 也跳过 fail).
  skipped_users.csv                  [汇总] 软跳过用户原因 (若有)
  trains/<user_id>/                  训练评估 metrics + 对比表 + plots
  infers/<user_id>/                  推理评估 metrics + 预测结果 + plots

------ 文件命名规范 (设备编号在前, 用户编号在后) ------
  文件夹: <device>_<user>            例: 800080252842_4206894986488
  总线:   e241_<device>_<user>-Ch{N}-<start>-<end>[-1|-infer].csv
  分路:   <user>-<start>-<end>[-infer].csv
  (因目录已区分训推, 后缀 -1 / -infer 已可省略, 但解析兼容历史命名)

------ 优雅降级 ------
  - 缺训练数据 (data/trains/<u>/ 为空 或不存在)  -> 跳过整个用户 (硬要求)
  - 缺推理数据 (data/infers/<u>/ 为空 或不存在)  -> 跳过 05 推理, 仍跑训练/验证/测试
  - 03 数据质量门触发 (对齐过少 / 单类标签 / 切分空) -> 软跳过 + skip_reason 入汇总

使用:
  python scripts/run_batch_users.py                 # 跑 data/trains/ 下全部用户
  python scripts/run_batch_users.py --users 800080252842_4206894986488 ...
  python scripts/run_batch_users.py --data-dir /path/to/data --skip-existing
"""
import argparse
import os
import re
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

# Windows GBK 控制台兼容: 强制 stdout/stderr 用 UTF-8
# 否则 print("✅ 用户切换") 等含 Unicode 符号会抛 UnicodeEncodeError
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------- 命名规范正则 [v6.12.6+v6.15.0-graceful-v9 简化] ----------
# 新目录结构: data/trains/<user_id>/  +  data/infers/<user_id>/
# 因目录已区分训推, 文件名去掉 -1 / -infer 后缀:
#   总线: e241_<device>_<user>-Ch{N}-<start>-<end>.csv
#   分路: <user>-<start>-<end>.csv
# 兼容: 为支持迁移期, 仍接受历史的 -1/-infer 后缀文件名 (匹配后等效处理)
RE_BUS = re.compile(
    r"^e241_(?P<device>[^_]+)_(?P<user>[^-]+)-Ch(?P<ch>\d+)-"
    r"(?P<start>\d{6})-(?P<end>\d{6})(?P<suffix>(-1|-infer)?)\.csv$"
)
RE_BR = re.compile(
    r"^(?P<user>[^-]+)-(?P<start>\d{6})-(?P<end>\d{6})(?P<suffix>(-1|-infer)?)\.csv$"
)


def _parse_one_dir(d: Path):
    """[v9] 解析单个子目录 (data/trains/<u>/ 或 data/infers/<u>/),
    返回 (bus_path, br_path, ch_seen_set). 该目录下应只有 1 个总线 + 1 个分路.
    若有多个, 取字典序最小的一个 (并不报错, 但日志层可加警告).
    """
    bus = br = None
    ch_set = set()
    if not d.exists() or not d.is_dir():
        return None, None, ch_set
    csv_files = sorted([f for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".csv"])
    for f in csv_files:
        if m := RE_BUS.match(f.name):
            if bus is None:
                bus = f
            ch_set.add(int(m["ch"]))
        elif m := RE_BR.match(f.name):
            if br is None:
                br = f
    return bus, br, ch_set


def parse_user_folder(folder_name: str, train_dir_root: Path, infer_dir_root: Path,
                       time_filter_config: dict = None):
    """[v9] 新解析: 从 trains/<folder_name>/ 和 infers/<folder_name>/ 提取数据.

    返回 dict:
        {folder_name, device, user, target_col, train_bus, train_br, infer_bus, infer_br}
    缺失项为 None.
    """
    info = {
        "folder_name": folder_name,
        "device": None,
        "user": None,
        "target_col": None,
        "train_bus": None,
        "train_br": None,
        "infer_bus": None,
        "infer_br": None,
        "errors": [],
    }

    # 从文件夹名解析 device_user
    parts = folder_name.split("_", 1)
    if len(parts) != 2:
        info["errors"].append(f"文件夹名 {folder_name} 不符合 <device>_<user> 格式")
        return info
    info["device"], info["user"] = parts[0], parts[1]

    # 扫训练子目录
    train_dir = train_dir_root / folder_name
    info["train_bus"], info["train_br"], ch_train = _parse_one_dir(train_dir)
    # 扫推理子目录
    infer_dir = infer_dir_root / folder_name
    info["infer_bus"], info["infer_br"], ch_infer = _parse_one_dir(infer_dir)

    # 通道编号 (训练优先, 训练无则用推理)
    ch_seen = ch_train if ch_train else ch_infer

    # 反推 target_col
    # [v13 新增] 优先级 0: 配置文件显式指定 (time_filter_config 中的 target_col 字段)
    # [v6.12.6+v6.15.0-graceful-v8 旧规则] 优先级 1-4:
    #   1. 从训练总线文件名 -Ch{N}- 提取 pN  (称为 ch_target, 例 Ch1 -> p1, Ch2 -> p2)
    #   2. 若 ch_target 在训练分路 CSV 列中存在 -> 用 ch_target
    #   3. 若 ch_target 不在分路列中 -> 退化到分路 CSV 中第 1 个 pN 列
    #   4. 极端兜底: 分路无任何 pN 列 / 总线无 -ChN- 标识 -> 默认 p1
    #
    # 历史背景:
    #   - 早期 v5 直接读分路 CSV 第 1 个 pN, 忽略 -Ch{N}-, 会导致某些 "Ch2 但分路含
    #     p1 p2 p3 p4" 的用户错选 p1.
    #   - v8 改为以总线 -Ch{N}- 为权威(因为这是真实测量通道编号), 但保留分路实际列
    #     存在性校验, 避免因 "Ch1 + 分路只有 p2" 这类历史用户回归 (会退到 p2).
    #   - v13 增加最高优先级 (配置文件), 覆盖场景: 数据文件命名不规范, 或用户希望
    #     覆盖默认反推 (如 Ch1 反推 p1, 但业务上要用 p2)

    # 读训练分路 CSV 的实际列, 找所有 pN 列 (给下面所有路径共用)
    br_p_cols = []
    if info["train_br"] is not None:
        try:
            br_cols = pd.read_csv(info["train_br"], nrows=1).columns.tolist()
            br_p_cols = [c for c in br_cols if re.fullmatch(r"p\d+", c.strip())]
        except Exception as e:
            info["errors"].append(f"读取分路 CSV 列名失败: {e}")

    # [v13] Step 0: 优先从配置文件读 (最高优先级)
    config_target_col = None
    if time_filter_config is not None:
        try:
            from time_filter_utils import get_user_target_col
            config_target_col = get_user_target_col(time_filter_config, folder_name)
        except Exception as e:
            info["errors"].append(f"[v13] 读配置 target_col 失败: {e}")

    if config_target_col is not None:
        # 配置存在: 宽松验证分路列
        # [v13.16] 复合列 (含 '+') 要求每个分量都在 br_p_cols 里
        if "+" in config_target_col:
            comp_parts = config_target_col.split("+")
            if br_p_cols:
                missing = [p for p in comp_parts if p not in br_p_cols]
                if not missing:
                    info["target_col"] = config_target_col
                    info["errors"].append(
                        f"[v13.16] target_col={config_target_col} 复合列来自配置 "
                        f"(所有分量 {comp_parts} 均在分路列 {br_p_cols} 中 ✓)"
                    )
                else:
                    info["errors"].append(
                        f"[v13.16 WARN] 配置复合 target_col={config_target_col} 缺分量 "
                        f"{missing} (分路实际列 {br_p_cols}), 回退到旧反推逻辑"
                    )
                    config_target_col = None
            else:
                # 分路读不到 pN 列, 但配置有值 -> 信任配置
                info["target_col"] = config_target_col
                info["errors"].append(
                    f"[v13.16 WARN] 分路 CSV 无 pN 列可校验, 直接采用配置 "
                    f"复合 target_col={config_target_col}"
                )
        elif config_target_col in br_p_cols:
            info["target_col"] = config_target_col
            info["errors"].append(
                f"[v13] target_col={config_target_col} 来自配置文件 (分路列={br_p_cols} 匹配 ✓)"
            )
        elif br_p_cols:
            # WARN: 配置指定的列不在分路中, 回退旧逻辑
            info["errors"].append(
                f"[v13 WARN] 配置指定 target_col={config_target_col} 不在分路列 {br_p_cols} 中, "
                f"回退到旧反推逻辑"
            )
            config_target_col = None  # 触发下面走旧逻辑
        else:
            # 分路读不到列, 但配置有值 -> 信任配置 (WARN)
            info["target_col"] = config_target_col
            info["errors"].append(
                f"[v13 WARN] 分路 CSV 无 pN 列可校验, 直接采用配置 target_col={config_target_col}"
            )

    if config_target_col is None:
        # [旧逻辑 v8] Step 1: 从总线名 -Ch{N}- 反推 ch_target
        ch_target = None
        if ch_seen:
            if len(ch_seen) > 1:
                info["errors"].append(f"用户 {folder_name} 含多种 Ch{ch_seen}, 仅取最小")
            ch_target = f"p{min(ch_seen)}"

        # Step 3: 按规则决定 target_col
        if ch_target and ch_target in br_p_cols:
            # 主路径: 总线通道与分路列对得上
            info["target_col"] = ch_target
        elif br_p_cols:
            # 退化 1: 总线通道在分路中无对应, 用分路里第 1 个 pN
            info["target_col"] = br_p_cols[0]
            if ch_target:
                info["errors"].append(
                    f"[警告] 总线 -Ch{min(ch_seen)}- 反推 {ch_target} 在分路中不存在 "
                    f"(分路列={br_p_cols}), 退化到 {br_p_cols[0]}"
                )
        elif ch_target:
            # 退化 2: 分路无任何 pN 列 (或分路读失败), 退到总线反推值
            info["target_col"] = ch_target
            info["errors"].append(
                f"[警告] 分路 CSV 无 pN 列, 直接用总线反推 {ch_target}"
            )
        else:
            # 兜底: 既无分路 pN 列, 也无总线 Ch{N} 标识 -> 默认 p1
            info["target_col"] = "p1"
            info["errors"].append(
                f"用户 {folder_name} 无法反推 target_col (分路无pN且总线无 -ChN-), 默认 p1"
            )

    # 缺失检查 -- 区分硬错误 (errors) 和警告 (warnings)
    if info["train_bus"] is None:
        info["errors"].append(f"缺训练总线文件 (data/trains/{folder_name}/e241_*-Ch?-*.csv)")
    if info["train_br"] is None:
        info["errors"].append(f"缺训练分路文件 (data/trains/{folder_name}/<user>-*-*.csv)")
    # 推理数据缺失只是降级, 不阻塞训练/验证/测试
    if info["infer_bus"] is None:
        info["errors"].append("[警告] 缺推理总线文件 -> 跳过 05 推理 (仍跑训练/验证/测试)")
    if info["infer_br"] is None:
        info["errors"].append("[警告] 缺推理分路文件 -> 推理仅出预测无评估指标")

    return info


def discover_users(data_dir: Path, time_filter_config: dict = None):
    """[v9/v13] 扫描 data_dir/trains/ 和 data/infers/ 取用户 ID 并集.

    新结构:
        data/trains/<user_id>/    必须存在才算"可训练"
        data/infers/<user_id>/    可选, 不存在则跳过推理

    [v13] time_filter_config: 若提供, parse_user_folder 会优先从中读 target_col

    返回所有合法用户的信息列表 (按 user_id 字典序).
    """
    train_root = data_dir / "trains"
    infer_root = data_dir / "infers"
    seen_ids = set()
    # 优先以 trains/ 下的目录为主 (只有训练数据才能跑流水线)
    if train_root.exists():
        for d in train_root.iterdir():
            if d.is_dir() and "_" in d.name:
                seen_ids.add(d.name)
    # 推理子目录补充 (若某用户只有 infers 没有 trains, 后面 is_runnable 会判为不可执行)
    if infer_root.exists():
        for d in infer_root.iterdir():
            if d.is_dir() and "_" in d.name:
                seen_ids.add(d.name)

    users = []
    for uid in sorted(seen_ids):
        users.append(parse_user_folder(uid, train_root, infer_root,
                                        time_filter_config=time_filter_config))
    return users


def is_runnable(info):
    """判断该用户是否可执行流水线 (优雅降级策略)

    必要条件 (硬要求):
      - 训练总线 + 训练分路 + target_col (用于 02->03->04)

    可选条件 (有则跑, 无则跳过对应步骤, 不阻塞):
      - infer_bus 缺失     -> 跳过 05 推理, 仅完成 train/val/test
      - infer_br  缺失     -> 跑 05 但只出预测, 无 inference 评估指标
    """
    if info["train_bus"] is None or info["train_br"] is None:
        return False
    if info["target_col"] is None:
        return False
    return True


def get_execution_plan(info):
    """返回该用户的执行计划描述 (用于扫描表展示)"""
    if not is_runnable(info):
        return "❌ 跳过 (缺训练数据)"
    if info["infer_bus"] is None:
        return "⚠️  仅训练 (无推理总线)"
    if info["infer_br"] is None:
        return "⚠️  训练+推理无评估 (无推理分路)"
    return "✅ 全流程 (训练+评估+推理)"


# ============================================================
# v13.17: 批量执行状态 CSV (支持断点续跑)
# ============================================================
_EXECUTION_STATE_CSV_NAME = "batch_execution_state.csv"
_EXECUTION_STATE_COLS = ["user_id", "status", "success",
                         "started_at", "finished_at", "duration_s",
                         "message", "target_col", "run_id"]


def _execution_state_path(output_dir) -> "Path":
    """返回状态 CSV 的路径 (统一在 output_dir 根下)."""
    return Path(output_dir) / _EXECUTION_STATE_CSV_NAME


def _load_execution_state(output_dir, logger_print=None) -> "pd.DataFrame":
    """[v13.17] 读取批量执行状态 CSV.

    返回 DataFrame; 若文件不存在, 返回带列头的空 DataFrame (不抛异常).
    若文件损坏 (JSON/CSV 解析失败), 打印 WARN 后返回空表 (走全部重跑).
    """
    p = _execution_state_path(output_dir)
    if not p.exists():
        if logger_print:
            logger_print(f"  [v13.17 续跑] 状态文件不存在: {p} -> 全部用户重新执行")
        return pd.DataFrame(columns=_EXECUTION_STATE_COLS)
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        # 兼容性检查: 若缺关键列, 视为损坏
        for c in ["user_id", "status"]:
            if c not in df.columns:
                if logger_print:
                    logger_print(f"  [v13.17 续跑] 状态文件缺列 {c!r}, 视为损坏 -> 全部重跑")
                return pd.DataFrame(columns=_EXECUTION_STATE_COLS)
        # 补齐可能缺失的新列 (向后兼容旧格式)
        for c in _EXECUTION_STATE_COLS:
            if c not in df.columns:
                df[c] = ""
        if logger_print:
            done_users = df[df["status"].isin(["ok", "soft_skip"])]["user_id"].tolist()
            logger_print(f"  [v13.17 续跑] 已加载状态文件: {p}")
            logger_print(f"  [v13.17 续跑] 历史记录 {len(df)} 行, "
                         f"已完成 (ok/soft_skip) {len(done_users)} 用户")
        return df
    except Exception as e:
        if logger_print:
            logger_print(f"  [v13.17 续跑] 状态文件读取失败 ({e!r}), 视为损坏 -> 全部重跑")
        return pd.DataFrame(columns=_EXECUTION_STATE_COLS)


def _get_completed_users(state_df, retry_failed: bool = True) -> set:
    """[v13.17] 从状态 DataFrame 提取已完成用户集合.

    Args:
        state_df:     _load_execution_state() 返回的 DataFrame
        retry_failed: True (默认): fail 用户重跑, 只跳 ok/soft_skip
                      False: fail 也跳过 (需手工删行才重试)

    Returns:
        set[str] 应跳过的 user_id 集合
    """
    if state_df is None or len(state_df) == 0:
        return set()
    if retry_failed:
        keep = state_df[state_df["status"].isin(["ok", "soft_skip"])]
    else:
        keep = state_df[state_df["status"].isin(["ok", "soft_skip", "fail"])]
    return set(keep["user_id"].astype(str).tolist())


def _upsert_execution_state(output_dir, row: dict, logger_print=None) -> None:
    """[v13.17] 原子写: 追加/更新单行到状态 CSV.

    - 若 user_id 已存在, 更新对应行 (支持重跑覆盖)
    - 用 .tmp + os.replace() 原子替换, 避免中断时 CSV 损坏
    - 每个用户跑完立即调用, 保证崩溃时最多丢当前正在跑的用户
    """
    p = _execution_state_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 载入现有 (若有)
    if p.exists():
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception:
            df = pd.DataFrame(columns=_EXECUTION_STATE_COLS)
    else:
        df = pd.DataFrame(columns=_EXECUTION_STATE_COLS)
    # 补齐所有列
    for c in _EXECUTION_STATE_COLS:
        if c not in df.columns:
            df[c] = ""
    # 保证 row 字段齐全
    for c in _EXECUTION_STATE_COLS:
        row.setdefault(c, "")
    # upsert (删旧行 + append 新行)
    df = df[df["user_id"].astype(str) != str(row["user_id"])]
    new_row = pd.DataFrame([row])[_EXECUTION_STATE_COLS]
    if len(df) == 0:
        # 空 df + concat 会触发 pandas 3.0 FutureWarning; 直接用 new_row
        df = new_row
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    # 原子写: 先写 .tmp 再 os.replace
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(str(tmp), str(p))
    if logger_print:
        logger_print(f"    [v13.17 状态] {row['user_id']}: {row['status']} -> {p.name}")


def run_single_user(info, output_dir, skip_existing=False, log_file=None,
                    force_retrain=False,
                    train_time_filter_spec="", infer_time_filter_spec="",
                    guard_enabled="",
                    splits_time_filter_spec="",
                    common_overrides_spec="",
                    v14_flags_spec="",
                    bus_guard_json="",
                    power_temp_calib_json="",
                    time_prior_json=""):
    """对单个用户调用 run_user_pipeline.py

    [v6.12.6+v6.15.0-graceful-v7] 三态返回:
      返回 (status: str, message: str)
      status ∈ {"ok", "soft_skip", "fail"}
        - "ok"        : 子进程 exit 0  (真成功, 产出模型 + 指标)
        - "soft_skip" : 子进程 exit 10 (03_train.py 数据质量门触发,
                        无模型无指标, 但有 artifacts/<user>/skip_reason.json)
        - "fail"      : 其它退出码 / 启动异常 / 超时

    [v10] force_retrain: 透传给 run_user_pipeline.py 的 --force-retrain
    [v12] train/infer_time_filter_spec: 时段过滤 JSON 字符串, 透传给 pipeline
    """
    user_id = info["folder_name"]
    # [v9 新路径] 已跳过判断改成: 训练评估目录存在且非空 (models/<u>/ 也可以,但 trains 更轻量)
    train_out = Path(output_dir) / "trains" / user_id
    if skip_existing and train_out.exists() and any(train_out.iterdir()):
        return "ok", f"已跳过 ({train_out} 已存在)"

    # 使用 sys.executable 确保跨平台兼容 (Windows 通常无 python3 命令)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_user_pipeline.py"),
        "--user-id", user_id,
        "--target-col", info["target_col"],
        "--train-bus", str(info["train_bus"]),
        "--train-branch", str(info["train_br"]),
        "--output-dir", str(output_dir),
    ]
    # 优雅降级: infer_bus 缺失则不传 --infer-bus -> 单用户流水线会跳过 05
    if info["infer_bus"] is not None:
        cmd += ["--infer-bus", str(info["infer_bus"])]
    if info["infer_br"] is not None:
        cmd += ["--infer-branch", str(info["infer_br"])]
    # [v10] 透传 --force-retrain
    if force_retrain:
        cmd += ["--force-retrain"]
    # [v12] 透传时段过滤 spec (JSON 字符串)
    if train_time_filter_spec:
        cmd += ["--train-time-filter-spec", train_time_filter_spec]
    if infer_time_filter_spec:
        cmd += ["--infer-time-filter-spec", infer_time_filter_spec]
    # [v13] 透传用户级 d87 守卫开关
    if guard_enabled in ("true", "false"):
        cmd += ["--guard-enabled", guard_enabled]
    # [v13] 透传 per-split time_filter 规格
    if splits_time_filter_spec:
        cmd += ["--splits-time-filter-spec", splits_time_filter_spec]
    # [v13.5] 透传 8 项 common 常量覆盖
    if common_overrides_spec:
        cmd += ["--common-overrides", common_overrides_spec]
    # [v14] 透传 v14 增强配置
    if v14_flags_spec:
        cmd += ["--v14-flags", v14_flags_spec]

    # v6.12.6+v6.15.0-graceful-v3: 强制子进程 stdout/stderr 使用 UTF-8 输出,
    # 与本批处理脚本 + 子进程 reconfigure 形成"端到端 UTF-8 一致" (Windows GBK 兼容)
    sub_env = dict(os.environ)
    sub_env.setdefault("PYTHONIOENCODING", "utf-8")
    # [v14.1] 总线一致性守卫配置透传给 run_user_pipeline -> 05_inference
    if bus_guard_json:
        sub_env["NILM_BUS_GUARD_JSON"] = bus_guard_json
        sub_env["NILM_BUS_GUARD_ENABLED"] = "1"
    else:
        sub_env.pop("NILM_BUS_GUARD_JSON", None)
        # 不强制 pop ENABLED, 允许外部环境全局打开
    # [v14.2] 推理端可选标定配置透传 (功率温桶标定 / 时段先验抑制)
    if power_temp_calib_json:
        sub_env["NILM_POWER_TEMP_CALIB_JSON"] = power_temp_calib_json
    else:
        sub_env.pop("NILM_POWER_TEMP_CALIB_JSON", None)
    if time_prior_json:
        sub_env["NILM_TIME_PRIOR_JSON"] = time_prior_json
    else:
        sub_env.pop("NILM_TIME_PRIOR_JSON", None)
    try:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*70}\n {user_id}  ({datetime.now()})\n{'='*70}\n")
                lf.write(f"CMD: {' '.join(cmd)}\n\n")
                lf.flush()
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   cwd=PROJECT_ROOT, timeout=1200, env=sub_env)
        else:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=1200, env=sub_env)
        if r.returncode == 0:
            return "ok", "成功"
        # [v6.12.6+v6.15.0-graceful-v7] 退出码 10 = 软跳过 (数据质量门触发)
        if r.returncode == 10:
            # [v9 新路径] 读 artifacts/trains/<user>/skip_reason.json 拿具体原因
            reason = "数据质量门触发"
            try:
                import json as _json
                skip_f = Path(output_dir) / "trains" / user_id / "skip_reason.json"
                if skip_f.exists():
                    info_js = _json.loads(skip_f.read_text(encoding="utf-8"))
                    reason = info_js.get("skip_reason", reason)
                    detail = info_js.get("detail", "")
                    if detail:
                        reason = f"{reason} ({detail})"
            except Exception:
                pass
            return "soft_skip", f"软跳过: {reason}"
        # 真失败: 把日志末尾打回终端, 帮用户快速定位 (避免 "code=1" 黑盒)
        err_tail = ""
        if log_file and Path(log_file).exists():
            try:
                lines = Path(log_file).read_text(encoding="utf-8", errors="ignore").splitlines()
                # 找本次 user_id 段落以来的最后 50 行
                start = max((i for i, ln in enumerate(lines) if user_id in ln), default=0)
                tail = lines[max(start, len(lines)-50):]
                err_tail = "\n".join(tail[-50:])
            except Exception:
                pass
        msg = f"流水线返回非零 (code={r.returncode})"
        if err_tail:
            print(f"\n  ────── 子进程日志末尾 (后 50 行) ──────")
            print(err_tail)
            print(f"  ──────────────────────────────────────")
            print(f"  完整日志: {log_file}")
        return "fail", msg
    except subprocess.TimeoutExpired:
        return "fail", "超时 (>20 分钟)"
    except FileNotFoundError as e:
        # Windows 常见: 找不到 python 命令本身
        return "fail", f"FileNotFoundError: {e} (检查 Python 解释器路径)"
    except Exception as e:
        return "fail", f"异常: {type(e).__name__}: {e}"


def collect_skip_reasons(output_dir: Path, summary_dir: Path):
    """[v9 新路径] 收集所有用户的 skip_reason.json 汇总成 CSV

    扫描 artifacts/trains/<user_id>/skip_reason.json (软跳过都属训练阶段失败),
    汇总成 artifacts/skipped_users.csv.
    """
    import json as _json
    _USER_DIR_RE = re.compile(r"^\d+_\d+$")
    train_root = output_dir / "trains"
    rows = []
    if not train_root.exists():
        return None, 0
    for user_dir in sorted(train_root.iterdir()):
        if not user_dir.is_dir() or not _USER_DIR_RE.match(user_dir.name):
            continue
        skip_f = user_dir / "skip_reason.json"
        if not skip_f.exists():
            continue
        try:
            info = _json.loads(skip_f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] 读取 {skip_f} 失败: {e}")
            continue
        row = {"user_id": user_dir.name}
        row.update(info)
        rows.append(row)
    if not rows:
        return None, 0
    # 列顺序: user_id, skip_reason, detail, 其余字段按出现顺序
    fixed = ["user_id", "skip_reason", "detail"]
    other = []
    for r in rows:
        for k in r.keys():
            if k not in fixed and k not in other:
                other.append(k)
    cols = fixed + other
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    out = summary_dir / "skipped_users.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out, len(df)


def aggregate_metrics(output_dir: Path, summary_dir: Path):
    """[v6.12.6+v6.15.0-graceful-v9] 汇总所有用户指标到单一 summary_metrics_all_users.csv

    新目录结构:
      output_dir (= artifacts/) 下:
        trains/<user_id>/{train_val_metrics.csv, test_metrics.csv, skip_reason.json, ...}
        infers/<user_id>/{inference_metrics.csv, ...}

    输出: artifacts/summary_metrics_all_users.csv
      每用户 4 行 (stage = train / val / test / inference), 仅最终主模型:
        train / val / test  → main
        inference           → main_final (退化 main)
      列: user_id, stage, status, Accuracy, Precision, Recall, F1, AUC,
          TN, FP, FN, TP, MAE_W, RMSE_W, SAE, NDE,
          kWh_true, kWh_pred, kWh_err, n_samples

    软跳过用户 4 行均为占位 (指标 NaN, status='soft_skip:<reason>'),
    无推理用户的 inference 行为占位 (status='no_inference').
    """
    import json as _json
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ---- 列定义 ----
    METRIC_COLS = [
        "Accuracy", "Precision", "Recall", "F1", "AUC",
        "TN", "FP", "FN", "TP",
        "MAE_W", "RMSE_W", "SAE", "NDE",
        "kWh_true", "kWh_pred", "kWh_err", "n_samples",
    ]
    HEADER_COLS = ["user_id", "stage", "status"] + METRIC_COLS
    INT_COLS = {"TN", "FP", "FN", "TP", "n_samples"}

    # 各 stage 的源 CSV 与所在子目录 + 模型优选顺序
    # (stage_key, parent_subdir, src_csv_name, split_val_in_csv, model_preference)
    STAGE_PLAN = [
        ("train",     "trains", "train_val_metrics.csv", "train",     ["main"]),
        ("val",       "trains", "train_val_metrics.csv", "val",       ["main"]),
        ("test",      "trains", "test_metrics.csv",      "test",      ["main"]),
        ("inference", "infers", "inference_metrics.csv", "inference", ["main_final", "main"]),
    ]

    def _empty_row(uid: str, stage: str, status: str) -> dict:
        row = {c: None for c in HEADER_COLS}
        row["user_id"] = uid
        row["stage"]   = stage
        row["status"]  = status
        return row

    # 收集所有用户 id (取 trains/ 与 infers/ 子目录并集)
    _USER_DIR_RE = re.compile(r"^\d+_\d+$")
    train_root = output_dir / "trains"
    infer_root = output_dir / "infers"
    all_users = set()
    for root in (train_root, infer_root):
        if root.exists():
            for d in root.iterdir():
                if d.is_dir() and _USER_DIR_RE.match(d.name):
                    all_users.add(d.name)

    rows = []
    src_cache: dict = {}  # (user_id, parent_subdir, src_csv) -> DataFrame or None

    def _read_metrics_csv(uid: str, parent_subdir: str, src_csv_name: str):
        key = (uid, parent_subdir, src_csv_name)
        if key in src_cache:
            return src_cache[key]
        f = output_dir / parent_subdir / uid / src_csv_name
        if not f.exists():
            src_cache[key] = None
            return None
        try:
            src_cache[key] = pd.read_csv(f)
        except Exception as e:
            print(f"  [WARN] {uid}: 读取 {f.name} 失败: {e}")
            src_cache[key] = None
        return src_cache[key]

    for user_id in sorted(all_users):
        # 先看是否软跳过
        skip_reason = None
        skip_f = train_root / user_id / "skip_reason.json" if train_root.exists() else None
        if skip_f is not None and skip_f.exists():
            try:
                skip_reason = _json.loads(skip_f.read_text(encoding="utf-8")).get("skip_reason", "skipped")
            except Exception:
                skip_reason = "skipped"

        for stage, parent_subdir, src_csv, split_val, model_pref in STAGE_PLAN:
            # 软跳过用户: 全 4 stage 都是占位
            if skip_reason is not None:
                rows.append(_empty_row(user_id, stage, f"soft_skip:{skip_reason}"))
                continue

            df = _read_metrics_csv(user_id, parent_subdir, src_csv)
            if df is None or len(df) == 0:
                # 没源 csv: 训练阶段 = no_train_metrics, 推理阶段 = no_inference
                placeholder = "no_inference" if parent_subdir == "infers" else f"no_{stage}_metrics"
                rows.append(_empty_row(user_id, stage, placeholder))
                continue
            if "split" not in df.columns or "model" not in df.columns:
                rows.append(_empty_row(user_id, stage, f"bad_{stage}_csv"))
                continue
            df_s = df[df["split"] == split_val]
            if len(df_s) == 0:
                rows.append(_empty_row(user_id, stage, f"no_{stage}_rows"))
                continue

            # 按模型优选顺序找
            chosen = None
            chosen_model = None
            for m in model_pref:
                sub = df_s[df_s["model"] == m]
                if len(sub) > 0:
                    chosen = sub
                    chosen_model = m
                    break
            if chosen is None:
                first_model = df_s["model"].iloc[0]
                chosen = df_s[df_s["model"] == first_model]
                chosen_model = first_model

            row = _empty_row(user_id, stage, f"ok:{chosen_model}")
            for _, r in chosen.iterrows():
                mname = r.get("metric")
                mval  = r.get("value")
                if mname in METRIC_COLS:
                    row[mname] = mval
            rows.append(row)

    if not rows:
        return []

    df_all = pd.DataFrame(rows)[HEADER_COLS]
    for c in INT_COLS:
        try:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce").astype("Int64")
        except Exception:
            pass
    out = summary_dir / "summary_metrics_all_users.csv"
    df_all.to_csv(out, index=False, encoding="utf-8-sig")
    return [("summary_metrics_all_users.csv", len(df_all), out)]


def main():
    ap = argparse.ArgumentParser(description="批量用户训练+评估+推理流水线")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"),
                    help="用户数据根目录 (默认 data/)")
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts"),
                    help="批量产物归档目录 (默认 artifacts/)")
    ap.add_argument("--users", nargs="*", default=None,
                    help="仅跑指定用户文件夹名 (省略 = 全部)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="若 artifacts/<user>/ 已存在则跳过")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅扫描+打印计划, 不实际执行")
    ap.add_argument("--continue-on-error", action="store_true", default=True,
                    help="某用户失败时继续跑其他用户 (默认 True)")
    # [v10] 模型复用控制
    ap.add_argument("--force-retrain", action="store_true",
                    help="[v10] 强制重新训练所有用户 (即使 models/<user>/ 已有完整模型);"
                         " 默认 False = 有模型就跳过训练只跑推理")
    # [v12] 时段过滤配置
    ap.add_argument("--time-filter-config", default="",
                    help="[v12] JSON 配置文件路径, 定义每用户 train/infer 的 include+exclude 时段. "
                         "结构: {user_id: {'train':{'include':[[start,end],...],'exclude':[...]}, "
                         "'infer': {...}}, '_default': {...}}. "
                         "支持任意时段 (非整天粒度), 闭区间. 未列出的用户会读 _default 键, 也可完全省略")
    # [v13.17] 断点续跑
    ap.add_argument("--resume", action="store_true",
                    help="[v13.17] 启用断点续跑: 读取 artifacts/batch_execution_state.csv, "
                         "跳过上次已完成 (ok/soft_skip) 的用户, 只跑剩余 + 上次 fail 的. "
                         "状态文件不存在或损坏时自动降级到全部重跑. "
                         "默认关闭 (与历史行为完全一致, 零回归)")
    ap.add_argument("--resume-skip-failed", action="store_true",
                    help="[v13.17] 配合 --resume 使用: 连上次 fail 的用户也跳过 (需手工删行才重试). "
                         "默认 False = fail 用户续跑时会重跑")
    ap.add_argument("--v14-flags", default="",
                    help="[v14] JSON 字符串, 透传给 run_user_pipeline.py 的 v14 开关")
    args = ap.parse_args()

    # [v12] 加载时段过滤配置
    time_filter_config = {}
    if args.time_filter_config.strip():
        from time_filter_utils import load_time_filter_config
        time_filter_config = load_time_filter_config(args.time_filter_config)
        print(f"  [v12] 已加载时段过滤配置: {args.time_filter_config}")
        print(f"        含用户级配置 {sum(1 for k in time_filter_config if not k.startswith('_'))} 条, "
              f"_default 键: {'有' if '_default' in time_filter_config else '无'}")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    # [v9 新结构] summary 文件直接放 artifacts/ 根目录
    summary_dir = output_dir
    # 批量级日志: logs/_batch/batch_run_<ts>.log (用户级日志在 logs/<user_id>/)
    batch_log_dir = PROJECT_ROOT / "logs" / "_batch"
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_log_dir / f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("="*70)
    print(f"  批量用户流水线 (v6.12.6+v6.15.0)")
    print(f"  data-dir   : {data_dir}  (扫描 trains/ + infers/ 子目录)")
    print(f"  output-dir : {output_dir}  (产出 trains/ infers/ summary_metrics_all_users.csv)")
    print(f"  batch log  : {log_path}")
    # [v10] 模型复用模式标识
    if args.force_retrain:
        print(f"  训练策略   : --force-retrain 强制重训所有用户")
    else:
        print(f"  训练策略   : 有完整模型则复用, 缺失则训练 (加 --force-retrain 强制重训)")
    print("="*70)

    # 1. 扫描 (v13: 传入 time_filter_config 让 parse_user_folder 优先读配置里的 target_col)
    users = discover_users(data_dir, time_filter_config=time_filter_config)
    if args.users:
        users = [u for u in users if u["folder_name"] in args.users]
    print(f"\n发现 {len(users)} 个用户文件夹:")
    print(f"{'用户ID':<40} {'target':<6} {'tr_bus':<7} {'tr_br':<7} {'inf_bus':<8} {'inf_br':<7} {'执行计划'}")
    print("-"*125)
    runnable, skipped = [], []
    for u in users:
        flags = [
            ("✓" if u["train_bus"] is not None else "✗"),
            ("✓" if u["train_br"]  is not None else "✗"),
            ("✓" if u["infer_bus"] is not None else "✗"),
            ("✓" if u["infer_br"]  is not None else "✗"),
        ]
        ok = is_runnable(u)
        plan = get_execution_plan(u)
        print(f"{u['folder_name']:<40} {u['target_col'] or '?':<6} "
              f"{flags[0]:<7} {flags[1]:<7} {flags[2]:<8} {flags[3]:<7} {plan}")
        if u["errors"]:
            for e in u["errors"]:
                print(f"    └─ {e}")
        (runnable if ok else skipped).append(u)

    # [v13.17] 断点续跑: 在 dry-run 之前应用, 让计划也反映真实待跑列表
    resume_completed = set()
    resume_state_df = None
    if args.resume:
        print(f"\n[v13.17 续跑] --resume 开启, 检查 {_execution_state_path(output_dir)}")
        resume_state_df = _load_execution_state(output_dir, logger_print=print)
        resume_completed = _get_completed_users(
            resume_state_df, retry_failed=not args.resume_skip_failed)
        if resume_completed:
            before_n = len(runnable)
            _kept, _skipped_resume = [], []
            for u in runnable:
                if u["folder_name"] in resume_completed:
                    _skipped_resume.append(u["folder_name"])
                else:
                    _kept.append(u)
            runnable = _kept
            policy = "只跳过 ok/soft_skip" if not args.resume_skip_failed else "跳过 ok/soft_skip/fail"
            print(f"[v13.17 续跑] 跳过策略: {policy}")
            print(f"[v13.17 续跑] 原计划 {before_n} 用户, 跳过已完成 {len(_skipped_resume)} 用户, "
                  f"实际待跑 {len(runnable)} 用户")
            if _skipped_resume:
                for uid in _skipped_resume:
                    print(f"    [SKIP-resume] {uid}")
        else:
            print(f"[v13.17 续跑] 无已完成用户可跳过, 按原计划执行 {len(runnable)} 用户")

    if args.dry_run:
        print(f"\n[dry-run] 不执行. 计划执行 {len(runnable)} 用户, 跳过 {len(skipped)} 用户")
        return 0
    if not runnable:
        print("\n无可执行用户, 退出"); return 1

    # 2. 执行
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t_start = datetime.now()
    # [v13.17] 本次批处理运行 ID (供状态 CSV 追溯是哪一次跑写的)
    run_id = t_start.strftime("%Y%m%d_%H%M%S")
    # [v6.12.6+v6.15.0-graceful-v7] 三态状态符
    # ✅/❌ 这俩是 BMP 内字符通常可显示, 但 ⏭ (U+23ED) 在 Windows GBK 控制台
    # 无法 encode (即使有 stdout reconfigure 也存在转发到日志文件的双解码风险).
    # 用 [SKIP] 纯 ASCII 等宽标签, Windows / Linux / macOS 都安全.
    STATUS_ICON  = {"ok": "[OK]  ", "soft_skip": "[SKIP]", "fail": "[FAIL]"}
    STATUS_LABEL = {"ok": "成功", "soft_skip": "软跳过", "fail": "失败"}
    for i, u in enumerate(runnable, 1):
        print(f"\n{'#'*70}\n  [{i}/{len(runnable)}] {u['folder_name']} (target={u['target_col']})")
        print(f"{'#'*70}")

        # [v12] 该用户的时段过滤规格 (train / infer 独立)
        _train_spec_str = ""
        _infer_spec_str = ""
        _guard_enabled = ""   # [v13] 空 = 未指定
        _splits_spec_str = ""   # [v13] per-split 过滤
        _common_overrides_str = ""   # [v13.5] 8 项 common 常量覆盖
        _v14_flags_str = getattr(args, "v14_flags", "")   # [v14] v14 增强开关
        _bus_guard_json = ""  # [v14.1] 总线一致性守卫
        if time_filter_config:
            from time_filter_utils import (get_user_stage_spec, spec_to_cli_arg,
                                            spec_summary, get_user_guard_enabled,
                                            load_splits_time_filter, splits_spec_to_cli_arg,
                                            splits_spec_summary,
                                            get_user_common_overrides,
                                            get_user_v14_flags)
            _train_spec = get_user_stage_spec(time_filter_config, u["folder_name"], "train")
            _infer_spec = get_user_stage_spec(time_filter_config, u["folder_name"], "infer")
            _train_spec_str = spec_to_cli_arg(_train_spec)
            _infer_spec_str = spec_to_cli_arg(_infer_spec)
            if _train_spec is not None or _infer_spec is not None:
                print(f"  [v12] 时段过滤: train={spec_summary(_train_spec)}, "
                      f"infer={spec_summary(_infer_spec)}")
            # [v13] 用户级 d87 守卫开关
            _guard_val = get_user_guard_enabled(time_filter_config, u["folder_name"])
            if _guard_val is True:
                _guard_enabled = "true"
                print(f"  [v13] d87 守卫: 强制开启 (来自配置)")
            elif _guard_val is False:
                _guard_enabled = "false"
                print(f"  [v13] d87 守卫: 强制关闭 (来自配置)")
            else:
                print(f"  [v13] d87 守卫: 未指定, 走全局 D87_ADAPTIVE_GUARD_ENABLED (可能被自动降级)")
            # [v13] per-split time_filter (train/val/test 独立)
            _splits_spec = load_splits_time_filter(time_filter_config, u["folder_name"])
            if _splits_spec is not None:
                _splits_spec_str = splits_spec_to_cli_arg(_splits_spec)
                print(f"  [v13] per-split time_filter: {splits_spec_summary(_splits_spec)}")
            # [v13.5] 8 项 common 常量覆盖 (on_thr_w / split_ratios / split_strategy /
            # post_min_on / post_fill_short_off / weather_latitude / weather_longitude /
            # use_weather_features / use_temp_based_season)
            _common_overrides = get_user_common_overrides(time_filter_config, u["folder_name"])
            if _common_overrides:
                import json as _json_batch
                _common_overrides_str = _json_batch.dumps(_common_overrides, ensure_ascii=False)
                print(f"  [v13.5] common 覆盖 {len(_common_overrides)} 项: "
                      f"{', '.join(f'{k}={v}' for k, v in _common_overrides.items())}")
            if not _v14_flags_str:
                _v14_dict = get_user_v14_flags(time_filter_config, u["folder_name"])
                if any(_v14_dict.values()):
                    import json as _json_batch
                    _v14_flags_str = _json_batch.dumps(_v14_dict, ensure_ascii=False)
                    print(f"  [v14] v14 增强配置: enabled={_v14_dict.get('v14_enable', False)}")
            # [v14.1] bus_guard / bus_consistency_guard 用户级配置
            _ucfg = time_filter_config.get(u["folder_name"], {})
            if isinstance(_ucfg, dict):
                _bg = _ucfg.get("bus_guard") or _ucfg.get("bus_consistency_guard")
                if _bg is True:
                    _bg = {"enabled": True}
                if isinstance(_bg, dict) and _bg.get("enabled", True):
                    import json as _json_batch
                    _bg_out = dict(_bg)
                    _bg_out.setdefault("enabled", True)
                    _bus_guard_json = _json_batch.dumps(_bg_out, ensure_ascii=False)
                    print(f"  [v14.1] bus_guard 启用: keys={list(_bg_out.keys())}")
                elif _ucfg.get("bus_guard_enabled") is True:
                    import json as _json_batch
                    _bus_guard_json = _json_batch.dumps({"enabled": True}, ensure_ascii=False)
                    print(f"  [v14.1] bus_guard_enabled=true")
            # [v14.2] power_temp_calib / time_prior 用户级配置 (推理端可选标定)
            _power_temp_calib_json = ""
            _time_prior_json = ""
            if isinstance(_ucfg, dict):
                import json as _json_batch2
                _pc = _ucfg.get("power_temp_calib")
                if isinstance(_pc, dict) and _pc.get("enable", False):
                    _power_temp_calib_json = _json_batch2.dumps(_pc, ensure_ascii=False)
                    print(f"  [v14.2] power_temp_calib 启用: {_power_temp_calib_json}")
                _tpr = _ucfg.get("time_prior")
                if isinstance(_tpr, dict) and _tpr.get("enable", False):
                    _time_prior_json = _json_batch2.dumps(_tpr, ensure_ascii=False)
                    print(f"  [v14.2] time_prior 启用: {_time_prior_json}")

        t0 = datetime.now()
        status, msg = run_single_user(u, output_dir, args.skip_existing, log_path,
                                       force_retrain=args.force_retrain,
                                       train_time_filter_spec=_train_spec_str,
                                       infer_time_filter_spec=_infer_spec_str,
                                       guard_enabled=_guard_enabled,
                                       splits_time_filter_spec=_splits_spec_str,
                                       common_overrides_spec=_common_overrides_str,
                                       v14_flags_spec=_v14_flags_str,
                                       bus_guard_json=_bus_guard_json,
                                       power_temp_calib_json=_power_temp_calib_json,
                                       time_prior_json=_time_prior_json)
        t1 = datetime.now()
        dt = (t1 - t0).total_seconds()
        icon  = STATUS_ICON.get(status, "?")
        label = STATUS_LABEL.get(status, status)
        # 单行清晰打印, 让"成功 / 软跳过 / 失败"一眼可辨
        # 若 msg 本身已含状态标签 (如"软跳过: xxx") 则不重复贴标签
        if status == "ok":
            line = f"  {icon} {u['folder_name']}: {label}  (耗时 {dt:.1f}s)"
        else:
            line = f"  {icon} {u['folder_name']}: {msg}  (耗时 {dt:.1f}s)"
        print(line)
        results.append({"user_id": u["folder_name"],
                        "status": status,         # 三态: ok / soft_skip / fail
                        "ok": (status == "ok"),    # 兼容旧字段 (仅真成功为 True)
                        "message": msg,
                        "duration_s": dt,
                        "target_col": u["target_col"]})

        # [v13.17] 每个用户完成后 立即 增量写入状态 CSV (支持崩溃恢复)
        # 原子写: .tmp + os.replace, 中断时最多丢当前正在跑的这一个用户
        try:
            _upsert_execution_state(output_dir, {
                "user_id": u["folder_name"],
                "status": status,
                "success": (status == "ok"),
                "started_at": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": t1.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_s": round(dt, 2),
                "message": (msg or "").replace("\n", " ")[:500],   # 单行且截断防超宽
                "target_col": u["target_col"] or "",
                "run_id": run_id,
            }, logger_print=print)
        except Exception as _wse:
            # 状态写入失败不影响主流程, 只打印 WARN
            print(f"    [v13.17 状态] WARN: 写入状态 CSV 失败 ({_wse}), 继续下一个用户")

    # 3. 汇总指标
    print(f"\n{'='*70}\n  汇总指标到 {summary_dir}\n{'='*70}")
    written = aggregate_metrics(output_dir, summary_dir)
    for fname, n, path in written:
        print(f"  ✓ {fname:<40} {n:>5} 行 -> {path}")

    # 3b. [v6.12.6+v6.15.0-graceful-v5] 汇总软跳过原因
    skip_csv, n_skipped = collect_skip_reasons(output_dir, summary_dir)
    if skip_csv is not None:
        print(f"  ✓ {'skipped_users.csv':<40} {n_skipped:>5} 行 -> {skip_csv}")
        # 按 skip_reason 分布做一行汇总, 让用户一眼看到哪些原因占大头
        try:
            sk_df = pd.read_csv(skip_csv)
            if "skip_reason" in sk_df.columns and len(sk_df) > 0:
                dist = sk_df["skip_reason"].value_counts().to_dict()
                dist_str = ", ".join([f"{k}={v}" for k, v in dist.items()])
                print(f"    [skip 原因分布] {dist_str}")
        except Exception:
            pass

    # 4. 总结
    # [v6.12.6+v6.15.0-graceful-v7] 直接基于 results 里的 status 字段计数,
    # 不再依赖 skipped_users.csv 反推 (status 由子进程退出码 0/10/其他 决定)
    print(f"\n{'='*70}\n  批量执行总结")
    print(f"{'='*70}")
    n_ok   = sum(1 for r in results if r["status"] == "ok")
    n_soft = sum(1 for r in results if r["status"] == "soft_skip")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"  总用户数              : {len(users)}")
    print(f"  可执行                : {len(runnable)}")
    print(f"  跳过 (数据缺文件)      : {len(skipped)}")
    print(f"  执行成功 (产出模型)    : {n_ok}")
    print(f"  软跳过 (数据质量门)    : {n_soft}")
    print(f"  执行失败 (真异常)      : {n_fail}")
    print(f"  总耗时                : {(datetime.now() - t_start).total_seconds():.1f}s")
    print(f"  详细日志              : {log_path}")
    print(f"  汇总指标目录          : {summary_dir}")
    # 保存执行记录: status 列直接来自子进程退出码, category 列保留兼容
    for r in results:
        r["category"] = r["status"]   # 兼容字段
    pd.DataFrame(results).to_csv(summary_dir / "batch_run_summary.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"  执行记录              : {summary_dir / 'batch_run_summary.csv'}")
    # [v13.17] 状态 CSV 提示
    _state_p = _execution_state_path(output_dir)
    if _state_p.exists():
        try:
            _sdf = pd.read_csv(_state_p, encoding="utf-8-sig")
            n_all = len(_sdf)
            n_ok_state = int((_sdf["status"] == "ok").sum()) if "status" in _sdf.columns else 0
            print(f"  批量执行状态 (v13.17) : {_state_p}  ({n_all} 行, ok {n_ok_state})")
            if args.resume:
                print(f"                          续跑模式已用; 下次同命令再加 --resume 会自动跳过 ok/soft_skip")
            else:
                print(f"                          若中途中断, 下次加 --resume 可断点续跑")
        except Exception:
            print(f"  批量执行状态 (v13.17) : {_state_p}")
    # 退出码: 真异常 (fail) 才 return 2; 软跳过不影响退出码
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
