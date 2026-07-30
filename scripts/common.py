# -*- coding: utf-8 -*-
"""
公共工具模块 (Windows + Conda 适配版)
- 项目路径定义 (pathlib, 跨平台)
- 中文字体自动适配
- 统一 logging (控制台 + 文件双输出)
- NILM 业务常量
"""
from pathlib import Path
import os
import sys
import time
import logging
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================
# 0. 项目版本号 (v6.10 新增)
# ------------------------------------------------------------
# 单一来源 (single source of truth), 所有指标输出 / 模型 meta 共享
# 升级版本时只改这一处, flatten_metrics_to_rows 会自动注入到每行指标
# ============================================================
PROJECT_VERSION = "v6.12.6+v6.15.0"
PROJECT_VERSION_DESC = (
    "v6.12.6 基线 + v6.15.0 自适应守卫阈值叠加 (回退分支 graceful-v9): "
    "保留 v6.12.6 步级状态机守卫 + 单一 ON_THR=10W + 单口径指标 + MAX_ON_HOURS=12h 硬编码; "
    "叠加 v6.15.0 自适应守卫 (A 软最大平滑 + B 样本量自适应 AF/MF + C 概率融合); "
    "graceful-v3 修复: subprocess 显式 UTF-8 编码 (修 Windows GBK 父端解码崩); "
    "graceful-v4 新增: 批量汇总宽表 (train/val/test/inference _metrics_all_users.csv); "
    "graceful-v5 新增: 03_train.py 3 道数据质量门 (对齐过少/单类/切分空) + 软跳过 + skipped_users.csv 原因汇总; "
    "graceful-v6 新增: 4 个宽表中软跳过用户保留占位行 (指标 NaN, 新增 status 列) + 软跳过路径清理顶层 artifacts; "
    "graceful-v7 修复: 软跳过用户单行打印误显示 '成功' 的 bug (sys.exit(10) + 三态 [OK]/[SKIP]/[FAIL]); "
    "graceful-v8 改造: target_col 反推改为总线 -Ch{N}- 优先, 分路有则用, 没有则退到分路第1个pN列, 都没有则默认 p1; "
    "graceful-v9 重构: 全新目录布局 (data/trains|infers, models/<u>, logs/<u>, artifacts/{summary,trains,infers}), 单一 summary_metrics_all_users.csv 替代 4 个宽表 (每用户 4 行 stage); "
    "graceful-v9.1 修复: parse_user_folder 函数签名改名后 2 处遗漏的 folder.name 引用 (NameError 'folder is not defined') 在'多种 Ch'警告路径上爆炸; "
    "graceful-v10 新增: --force-retrain 控制模型复用; 默认 '有完整模型则复用只跑推理, 缺失或强制则训练'; 复用模式下保留上次 train/val/test 评估, 只更新本次 inference 行; 实测 3 用户复用模式总耗时从 147s -> 8s (~20x); "
    "graceful-v11 新增: D87_ADAPTIVE_GUARD_ENABLED 总开关 (默认 False), 关闭时训练侧不写 d87 元数据、推理侧跳过 5a 步级状态机 + 6a baseline 压制; 实测变频用户 252842 F1 0.22->0.78 (SAE 70%->44%), 部分定频用户如 252844 会有回归 (F1 0.97->0.67 SAE 12%->65%), 需按用户特性选择开关值; "
    "已移除: v6.13(ON_THR解耦) / v6.14(增量训练+MF=1.25) / v6.16(双口径+守卫训练对称+MAX_ON自适应+启动确认)"
)
PROJECT_VERSION_DESC_LEGACY = (
    "v6.12.6 步级状态机守卫 (替代日级); "
    "v6.12.5 ALLOW_FACTOR 0.9; "
    "v6.12.4 d87 守卫双源约束标定; "
    "v6.12.2 d87 守卫 d73 自适应缩放; "
    "v6.12.1 d87 守卫自适应阈值"
)

# ============================================================
# 1. 项目路径 (相对路径, Windows/Linux 通用)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR    = PROJECT_ROOT / "models"
LOG_DIR      = PROJECT_ROOT / "logs"
PRED_DIR     = ARTIFACT_DIR / "predictions"
METRIC_DIR   = ARTIFACT_DIR / "metrics"

# v6.12.6+v6.15.0-graceful-v9 新增: 批量分用户的永久目录
# (业务脚本 01-06 仍用 ARTIFACT_DIR/MODEL_DIR/LOG_DIR 作为"本次运行临时目录",
#  由 run_user_pipeline.py 在归档时把临时产物分流到下面的永久目录)
TRAIN_DATA_DIR     = DATA_DIR / "trains"      # 训练数据按 <user_id>/ 分目录
INFER_DATA_DIR     = DATA_DIR / "infers"      # 推理数据按 <user_id>/ 分目录
TRAIN_ARTIFACT_DIR = ARTIFACT_DIR / "trains"  # 训练评估指标按 <user_id>/ 分目录
INFER_ARTIFACT_DIR = ARTIFACT_DIR / "infers"  # 推理评估指标按 <user_id>/ 分目录

for d in (ARTIFACT_DIR, MODEL_DIR, LOG_DIR, PRED_DIR, METRIC_DIR,
          TRAIN_DATA_DIR, INFER_DATA_DIR,
          TRAIN_ARTIFACT_DIR, INFER_ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 默认数据文件路径
#   训练: BUS_CSV / BR_CSV         (由 merge_data.py 或 run_user_pipeline.py 写入)
#   推理: INFER_BUS_CSV / INFER_BR_CSV  (生产部署/OOD 评估的新数据)
# 设计动机:
#   v6.12.6+v6.15.0 训推路径分离 -- 避免 "推理数据覆盖训练数据" 的隐患,
#   也避免训练用合并大表 vs 推理用单时段数据的同名混淆
BUS_CSV       = DATA_DIR / "merged_bus.csv"      # 训练总线 (历史合并大表)
BR_CSV        = DATA_DIR / "merged_branch.csv"   # 训练分路
INFER_BUS_CSV = DATA_DIR / "infer_bus.csv"       # 推理总线 (新增 - 生产数据)
INFER_BR_CSV  = DATA_DIR / "infer_branch.csv"    # 推理分路 (新增 - 生产数据, 可选)

# 默认模型文件
MODEL_PKL    = MODEL_DIR / "nilm_ac_two_stage.pkl"      # v5 主模型 (含温度)
MODEL_V42_PKL = MODEL_DIR / "nilm_ac_two_stage_v42.pkl" # v4.2 对照模型 (无温度)

# ============================================================
# 2. NILM 业务常量
# ============================================================
SENT_VALUE = -2147483648    # INT32_MIN, 电表上送缺测占位符
# v6.12.6 单一 ON_THR_W=10W (训练标签 + 业务评估同口径)
# 注: v6.13 引入的 ON_THR_TRAIN_W/ON_THR_BUSINESS_W 解耦已在本分支移除
#     保留三个别名指向同一值, 保证下游代码兼容
ON_THR_W          = 10.0    # 唯一阈值 (W)
ON_THR_TRAIN_W    = ON_THR_W   # 别名: 训练标签阈值 (= ON_THR_W)
ON_THR_BUSINESS_W = ON_THR_W   # 别名: 业务评估阈值 (= ON_THR_W, v6.12.6 同口径)

# ============================================================
# v6.15 自适应守卫阈值配置 (替代 v6.14.2 手工 MARGIN_FACTOR=1.25)
# ============================================================
# 设计思想:
#   v6.14.2 之前: ALLOW_FACTOR/MARGIN_FACTOR 都是硬编码常量, 一刀切
#     问题1: 用户3 (n_on=7, n_off=694) 与 用户1 (n_on=80, n_off=4800) 用同一组参数,
#            前者 P10/P99 噪声极大, 后者很稳定 -> 同一 MF=1.3 在前者就会卡边界
#     问题2: max(on_c, off_c) 是硬翻转, 当 on_c=127, off_c=126 时, MF=1.30 vs 1.25
#            会让阈值在 127<->131 跳变, 边界事件被 4W 决定生死
#     问题3: 守卫与模型概率完全解耦, p_on=1.000 (强信号) 与 p_on=0.4 用同一阈值
#
# v6.15 三层自适应:
#   (A) 软最大平滑    : gap 小时取均值, gap 大时取 max (避免绑定翻转)
#   (B) 样本量自适应  : n_on/n_off 越少, AF/MF 越保守 (避免分位数噪声主导)
#   (C) 概率融合守卫  : 推理时 D87_GUARD_TH × (1 - GAMMA × p_on), 强信号放宽

# ============================================================
# [v11] d87 启动尖峰自适应守卫 总开关
# ------------------------------------------------------------
# True  = 启动 v6.12.x + v6.15.0 全套自适应守卫机制:
#         训练侧: 学 ON/OFF 段 d87 分布 -> ALLOW_FACTOR/MARGIN_FACTOR -> 阈值
#         推理侧: d73 自适应缩放 + 软最大 + 概率融合 + 步级状态机压制
#         效果: 大部分定频空调用户 F1 显著提升, 但对变频空调用户 (d87 尖峰退化)
#              会误压真 ON 事件, Recall 崩溃 (参考 800080252842 case)
# False = 完全关闭 d87 守卫机制:
#         训练侧: 不写 d87 元数据 (bundle['d87_guard_meta']['enabled'] = False)
#         推理侧: 跳过整个 5a/6a 守卫块 (不做步级状态机, 不对 baseline 压制)
#         效果: 模型/后处理直接决定 ON/OFF, 变频空调不再被误压;
#              但定频空调用户 (曾靠守卫拦下 FP 的) 可能 FP 上升
# ============================================================
D87_ADAPTIVE_GUARD_ENABLED = True   # 默认关闭 d87 自适应守卫 (变频用户需要, 定频用户可在 time_filters.json 单独开启)

# (A) 软最大平滑
GUARD_SOFTMAX_TEMP_W   = 8.0   # sigmoid 温度参数 (W), gap 小于此值开始平滑
GUARD_SOFTMAX_PIVOT_W  = 15.0  # gap=此值时 weight=0.5 (max 与均值各半)

# (B) 样本量自适应 (基础 + 范围)
GUARD_AF_MIN, GUARD_AF_MAX = 0.75, 0.90   # ALLOW_FACTOR 自适应范围
GUARD_MF_MIN, GUARD_MF_MAX = 1.05, 1.30   # MARGIN_FACTOR 自适应范围
GUARD_AF_N_REF             = 30           # ON 样本充足参考值 (>=30 用 AF_MAX)
GUARD_MF_N_REF             = 1500         # OFF 样本充足参考值 (>=1500 用 MF_MAX)

# (C) 概率融合守卫 (推理侧)
GUARD_PROB_FUSION_ENABLED = True   # 是否启用概率融合
GUARD_PROB_GAMMA          = 0.30   # p_on=1 时阈值乘 (1 - 0.30) = 0.7 (放宽 30%)
GUARD_PROB_MIN_RATIO      = 0.60   # 局部阈值下限 (避免被 p_on 过度压制 -> FP)
RESAMPLE   = "15min"        # 总线重采样周期 (与分路对齐)
RANDOM_SEED = 42

# 目标分路列名 (空调 = p1, 冰箱 = p2, 热水器 = p3, 照明 = p4, 以此类推)
# 多分路 NILM 扩展时只需改此处, 全工程自动适配
TARGET_COL = "p1"           # 当前任务: 空调分路

# ============================================================
# 2.1 数据集划分配置 (v6.5 新增, 全工程统一)
# ============================================================
# 切分策略:
#   "stratified_day" [*] v6.10 默认: 按月分层 + 按"完整天"随机抽样 (整天归 split)
#                                   消除 stratified 同天切分边界泄漏问题
#                                   完整天阈值=80 条/天 (15min 间隔下 96 步的 83%)
#                                   碎片天 (<80 条) 自动归 train, 不污染 val/test
#                                   固定 seed=42 保证可复现
#   "stratified"     按月分层时序切分 (v6.7~6.9 旧默认, 季节覆盖好但有同天泄漏)
#   "time"           纯时序切分 (简单, 但季节工况单一时易过拟合)
SPLIT_STRATEGY = "stratified_day"

# 训练/验证/测试三集占比, 必须三个正数, 和必须 = 1.0
# 业界常用配置:
#   (0.70, 0.15, 0.15)  - 标准三七一五划分 (默认, 兼顾足够训练 + 充分测试)
#   (0.80, 0.10, 0.10)  - 训练偏多, 适合数据量大或模型复杂时
#   (0.60, 0.20, 0.20)  - 评估偏多, 适合小数据集需更稳健评估
#   (0.70, 0.20, 0.10)  - val 偏多, 适合 v6 L4 校正器需更多 val 样本时
SPLIT_RATIOS = (0.60, 0.20, 0.20)


def validate_split_ratios(ratios=None):
    """
    校验切分比例合法性 (在脚本入口调用, 提前发现配置错误)
    返回: tuple(float, float, float)  归一化后的比例
    """
    if ratios is None:
        ratios = SPLIT_RATIOS
    if len(ratios) != 3:
        raise ValueError(f"SPLIT_RATIOS 必须是 3 元组, 实际: {ratios}")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"SPLIT_RATIOS 所有元素必须 > 0, 实际: {ratios}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        # 自动归一化 + 警告
        import warnings
        warnings.warn(
            f"SPLIT_RATIOS 和 = {total:.4f} ≠ 1.0, 已自动归一化为 "
            f"{tuple(round(r/total, 4) for r in ratios)}",
            UserWarning, stacklevel=2,
        )
        ratios = tuple(r / total for r in ratios)
    return tuple(float(r) for r in ratios)

# ============================================================
# 2.1 v5 气象配置 (Open-Meteo API)
# ============================================================
# 用户空调安装地点 (经纬度, WGS84) - 默认: 武汉
WEATHER_LATITUDE  = 30.59
WEATHER_LONGITUDE = 114.31
WEATHER_CITY      = "Wuhan"
# 气象数据本地缓存目录
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"
# 温度驱动季节路由阈值 (°C)
SUMMER_TEMP_THRESHOLD     = 22.0   # 日均 >= 22°C 视为夏季 (制冷主导)
WINTER_TEMP_THRESHOLD     = 12.0   # 日均 <= 12°C 视为冬季 (制热主导)
# 是否启用温度特征 (False = 退回 v4.2 行为)
USE_WEATHER_FEATURES      = True
# 是否使用温度驱动的季节路由 (False = 按月份硬路由)
USE_TEMP_BASED_SEASON     = True


# ============================================================
# 2b. v6.9: L5 ModelSwitcher 决策阈值 (参数化, 便于线上调参)
# ------------------------------------------------------------
# 触发阈值: 同时满足任一即升级到对应级别 (按 ALERT > WARN 优先级)
L5_ALERT_N_ALERT          = 3      # 漂移报告中 ALERT 行数 ≥ 此值 -> ALERT
L5_ALERT_MAX_CONCEPT_DRIFT= 0.50   # 最大概念漂移 ≥ 50% -> ALERT
L5_WARN_N_ALERT           = 1      # ALERT 行数 ≥ 此值 -> WARN
L5_WARN_N_WARN            = 2      # WARN 行数 ≥ 此值 -> WARN
L5_WARN_MAX_CONCEPT_DRIFT = 0.20   # 最大概念漂移 ≥ 20% -> WARN
# 主模型权重: 区分 "L4 启用" 与 "L4 未启用" 两组
# 设计依据 (v6.9): L4 启用时主模型已被校正, 在 OOD 上残余偏差被显著修正,
# 故在 ALERT/WARN 模式下应保留更多主模型权重, 避免 L5 把 L4 的校正收益
# (5 月推理实测 SAE -65%) 全部覆盖。详见 REPORT v6.9 §11。
L5_MAIN_WEIGHT_ALERT_WITH_L4    = 0.5   # L4 启用 + ALERT
L5_MAIN_WEIGHT_ALERT_WITHOUT_L4 = 0.0   # L4 未启用 + ALERT (保持旧行为)
L5_MAIN_WEIGHT_WARN_WITH_L4     = 0.75  # L4 启用 + WARN
L5_MAIN_WEIGHT_WARN_WITHOUT_L4  = 0.6   # L4 未启用 + WARN (保持旧行为)


# ============================================================
# 3. Matplotlib 中文字体 (Windows 优先)
# ============================================================
def setup_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
        "Microsoft JhengHei",
        "WenQuanYi Zen Hei", "Noto Sans CJK SC",
        "PingFang SC", "Heiti SC",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in candidates if n in installed), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return chosen
    return None


# ============================================================
# 4. 统一 Logger (控制台 + 文件)
# ============================================================
_LOGGER_INITIALIZED = {}

def get_logger(name: str = "nilm", log_to_file: bool = True) -> logging.Logger:
    """
    获取全局 logger, 同一 name 仅初始化一次, 避免重复打印。
    控制台 INFO 级别, 文件 DEBUG 级别。

    Windows GBK 控制台兼容: 在第一次创建 logger 时尝试把 stdout/stderr 切到 UTF-8.
    这样 log.info("[OK] ✓ 完成") 这类已有代码不会在 Windows cmd 抛
    UnicodeEncodeError. 文件 handler 本身就用 UTF-8 编码, 无需额外处理.
    """
    if name in _LOGGER_INITIALIZED:
        return logging.getLogger(name)

    # 控制台编码兼容 (Python 3.7+)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    if log_to_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = LOG_DIR / f"{name}_{ts}.log"
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.debug(f"日志文件: {fp}")

    _LOGGER_INITIALIZED[name] = True
    return logger


# ============================================================
# 5. 计时上下文 (含日志输出)
# ============================================================
class Timer:
    def __init__(self, name: str, logger: logging.Logger = None):
        self.name = name
        self.logger = logger or get_logger()

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f">>> [START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.time() - self.t0
        if exc_type is None:
            self.logger.info(f"<<< [DONE ] {self.name}  (耗时 {dt:.2f}s)")
        else:
            self.logger.error(f"<<< [FAIL ] {self.name}  (耗时 {dt:.2f}s) "
                              f"异常: {exc_type.__name__}: {exc_val}")
        return False  # 不吞异常
