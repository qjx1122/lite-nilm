# -*- coding: utf-8 -*-
"""
[v14 方向⑦] 模型卡 (Model Card) 自动生成 + 模型导出 (ONNX/C)

用法:
    # 1. 为已训练模型生成 Markdown 模型卡
    python scripts/export_model_card.py \
        --model models/<uid>/nilm_ac_two_stage.pkl \
        --out   models/<uid>/MODEL_CARD.md

    # 2. 导出 ONNX (分类器+回归器, 可选动态INT8量化)
    python scripts/export_model_card.py \
        --model models/<uid>/nilm_ac_two_stage.pkl \
        --export-onnx models/<uid>/nilm.onnx --quantize

    # 3. 一次性产出模型卡 + ONNX + C 代码
    python scripts/export_model_card.py \
        --model models/<uid>/nilm_ac_two_stage.pkl \
        --out models/<uid>/MODEL_CARD.md \
        --export-onnx models/<uid>/nilm.onnx --quantize \
        --export-c models/<uid>/infer.c
"""
import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path

# 保证 scripts 可 import
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)


def load_model(path: Path):
    import joblib
    return joblib.load(path)


def generate_model_card(bundle: dict, model_path: Path, out_path: Path = None) -> str:
    """生成 Markdown 模型卡, 含模型概要/性能/特征/使用方法/部署信息"""
    feat_names = bundle.get("feat_names", [])
    n_feat = len(feat_names)
    n_train = bundle.get("n_train", 0)
    n_val = bundle.get("n_val", 0)
    version = bundle.get("version", "unknown")
    trained_at = bundle.get("trained_at", "unknown")
    best_thr = bundle.get("best_thr", 0.5)
    on_thr = bundle.get("ON_THR", 10.0)
    post_min = bundle.get("post_min_on", 1)
    post_fill = bundle.get("post_fill_short_off", 3)

    # 特征族归类
    families = {
        "raw": [f for f in feat_names if f.startswith("load_iden_data")
                and "_d" not in f and "_lag" not in f and "_rm" not in f
                and "_rs" not in f and "_ema" not in f and "_range_" not in f
                and "_std_" not in f and "_cv_" not in f and "_p25" not in f
                and "_p75" not in f and "_iqr_" not in f and "_slope_" not in f
                and "_dev_" not in f and "_z_" not in f and "ratio_" not in f
                and not f.endswith(("_max5", "_min5", "_absmax5"))
                and "d87_jump" not in f and "d87_amax" not in f
                and "d87_event" not in f and "d87_spike" not in f],
        "diff": [f for f in feat_names if f.endswith("_d1") or f.endswith("_d3") or f.endswith("_d6")],
        "abs_diff": [f for f in feat_names if "_abs_d" in f],
        "jump_flag": [f for f in feat_names if "_up_d1" in f or "_down_d1" in f],
        "rolling": [f for f in feat_names if "_rm" in f or "_rs" in f],
        "lag": [f for f in feat_names if "_lag" in f],
        "ema": [f for f in feat_names if "_ema" in f and "ratio" not in f],
        "range": [f for f in feat_names if "_range_" in f],
        "spike": [f for f in feat_names if "d87" in f],
        "time": ["hour", "dow", "is_evening", "is_weekend",
                 "sin_hour", "cos_hour", "month", "sin_doy", "cos_doy"],
        "weather": [f for f in feat_names if "temp_" in f or "cooling_" in f
                    or "heating_" in f or "humidity" in f or "apparent" in f],
        "drift": [f for f in feat_names if "power_recent" in f or "temp_power" in f
                  or "is_morning" in f],
        "v14_physics": [f for f in feat_names if any(k in f for k in
                        ["_std_", "_cv_", "_p25_", "_p75_", "_iqr_", "_slope_",
                         "_dev_24h", "_dev_7d", "_z_24h", "ratio_"])],
    }
    fam_counts = {k: len([f for f in v if f in feat_names]) for k, v in families.items()}
    total = sum(fam_counts.values())
    # 未分类
    classified = set()
    for v in families.values():
        classified.update(v)
    n_other = n_feat - len(classified & set(feat_names))

    # 特征重要性 Top-15
    top_imp_rows = []
    clf = bundle.get("clf")
    if clf is not None and hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        order = np.argsort(-imp)[:15]
        for rank, i in enumerate(order, 1):
            top_imp_rows.append(
                f"| {rank} | {feat_names[i]} | {imp[i]:.4f} |")
    else:
        top_imp_rows.append("| - | (模型对象不含 feature_importances_) | - |")

    # d87 守卫信息
    guard = bundle.get("d87_guard_meta", {}) or {}
    guard_enabled = guard.get("enabled", False)
    guard_rows = []
    if guard_enabled:
        guard_rows.append(f"- 守卫状态: **启用** ({guard.get('calibration','v6.15')})")
        guard_rows.append(f"- 训练阈值 |d87|: {guard.get('threshold_abs',0):.1f}")
        guard_rows.append(f"- 自适应缩放: d73_P95={guard.get('train_d73_p95',0):.0f}W")
        guard_rows.append(f"- 样本量: ON={guard.get('n_on_amax',0)}, OFF={guard.get('n_off_amax',0)}")
        guard_rows.append(f"- AF={guard.get('allow_factor',0):.3f}, MF={guard.get('margin_factor',0):.3f}")
    else:
        guard_rows.append("- 守卫状态: **关闭** (变频设备或 d87 尖峰退化)")

    # 模型大小
    try:
        size_kb = model_path.stat().st_size / 1024
        size_str = f"{size_kb/1024:.2f} MB ({size_kb:.0f} KB)"
    except Exception:
        size_str = "N/A"

    md = f"""# NILM 空调负荷分解模型卡 (Model Card)

## 1. 模型概要

| 项 | 值 |
|---|---|
| 任务类型 | 非介入式空调辨识 (ON/OFF 分类 + 功率回归) |
| 算法架构 | 两阶段 GBDT (Stage-1 分类 + Stage-2 MoE 分位回归) + 温度驱动季节路由 + L1-L5 漂移防御 |
| 模型版本 | `{version}` |
| 训练时间 | {trained_at} |
| 训练样本 | n_train={n_train}, n_val={n_val} |
| 特征维度 | **{n_feat}** 维 |
| 模型文件大小 | {size_str} |
| 采样粒度 | 15 min (96 点/天) |

## 2. 后处理与阈值

| 项 | 值 | 说明 |
|---|---:|---|
| ON 判定功率阈值 | {on_thr:.1f} W | 标签二值化阈值 (训练+评估同口径) |
| Stage-1 分类阈值 | {best_thr:.3f} | Val 集 F1 最优 |
| 最小持续 ON 步数 | {post_min} | {post_min*15} min |
| 短 OFF 填充步数 | ≤{post_fill} | 压缩机短歇填充 ({post_fill*15} min) |

## 3. d87 启动尖峰守卫

{chr(10).join(guard_rows)}

## 4. 特征族分布

| 特征族 | 维数 |
|---|---:|
"""
    fam_names = {
        "raw": "原始电参量",
        "diff": "差分(d1/d3/d6)",
        "abs_diff": "绝对差分",
        "jump_flag": "突变方向标志",
        "rolling": "滚动均值/方差",
        "lag": "滞后项",
        "ema": "EMA 跨尺度",
        "range": "窗口极差",
        "spike": "d87 启动尖峰 (极值+事件)",
        "time": "时间编码",
        "weather": "气象温度特征",
        "drift": "L1 漂移感知",
        "v14_physics": "v14 物理指纹 (std/cv/斜率/IQR/偏离)",
    }
    for k, label in fam_names.items():
        c = fam_counts.get(k, 0)
        if c > 0:
            md += f"| {label} | {c} |\n"
    if n_other > 0:
        md += f"| 其他 | {n_other} |\n"
    md += f"| **合计** | **{total}** |\n"

    md += f"""
## 5. Top-15 特征重要性 (Stage-1 分类器)

| 排名 | 特征名 | 重要性 |
|---:|---|---:|
{chr(10).join(top_imp_rows)}

## 6. 预期性能参考

> 注: 性能指标高度依赖用户/季节/工况, 此处仅为参考区间, 以本模型对应
> train/val/test/inference 四张 CSV 指标文件为准。

| 数据集 | F1 期望 | MAE 期望 | SAE 期望 |
|---|---:|---:|---:|
| Train (同分布) | ≥ 0.99 | ≤ 20 W | ≤ 2% |
| Val (同季节) | ≥ 0.95 | ≤ 60 W | ≤ 6% |
| Test (跨季节) | ≥ 0.90 | ≤ 70 W | ≤ 8% |
| Inference (OOD) | ≥ 0.90 | ≤ 80 W | ≤ 15% |

触发重训信号 (任一项):
- L2 漂移 ALERT 数 ≥ 3 持续 3 天
- Inference SAE > 15% 连续 3 天
- L5 切换触发率 > 50% 一周

## 7. 部署要求

| 项 | 要求 |
|---|---|
| 推理延迟 (x86 Python) | < 5 ms/样本 |
| 推理延迟 (ARM ONNX) | ~10 ms/样本 |
| 推理延迟 (STM32H7 C) | ~30 ms/样本 |
| RAM (MCU 部署) | < 32 KB (特征 buffer) + < 200 KB (模型) |
| Flash (MCU 部署) | < 1 MB (主模型 clf+reg) |
| 运行时依赖 | numpy, scikit-learn (Python) / onnxruntime (边缘) / 无 (纯 C) |

## 8. 使用方法

### Python 推理示例

```python
import sys, joblib, numpy as np
sys.path.insert(0, "scripts")
from feature_utils import build_features
from postprocess import apply_postprocess

bundle = joblib.load("nilm_ac_two_stage.pkl")
scaler, clf, moe = bundle["scaler"], bundle["clf"], bundle["moe"]
thr = bundle["best_thr"]

# X: 15min 粒度总线特征 (与 build_features 输出同构)
X_s = scaler.transform(X)
p_on = clf.predict_proba(X_s)[:, 1]
state_raw = (p_on >= thr).astype(int)
p_p50 = np.clip(moe.predict(X_s, season_labels, alpha=0.5), 0, None)
state, y_pred = apply_postprocess(state_raw, p_p50,
                                 bundle["post_min_on"],
                                 bundle["post_fill_short_off"])
```

## 9. 版本演进

本模型基于 v6.12.6+v6.15.0 稳定分支, 包含:
- L1-L5 五层漂移防御
- d87 自适应启动尖峰守卫 (软最大+样本量自适应+概率融合)
- stratified_day 按天分层切分 (零切分泄漏)
- 温度驱动季节 MoE + 12 维气象特征
- v14 物理指纹特征增强 (std/cv/IQR/斜率/偏离/比值)
- 完整的逐日主模型指标 CSV + 训练健康度报告
"""
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"  [模型卡] -> {out_path} ({len(md)} chars)")
    return md


def export_onnx(bundle, out_path: Path, quantize: bool = True):
    """导出 ONNX 模型 (clf + reg), 可选 int8 动态量化"""
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("  [ONNX] skl2onnx 未安装, 跳过 (pip install skl2onnx onnx onnxruntime)")
        return None

    n_features = len(bundle.get("feat_names", []))
    if n_features == 0:
        print("  [ONNX] feat_names 为空, 跳过")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("input", FloatTensorType([None, n_features]))]

    # 1) 分类器
    clf = bundle.get("clf")
    reg = bundle.get("reg")  # fallback P50
    clf_path = out_path.with_name(out_path.stem + "_clf.onnx")
    reg_path = out_path.with_name(out_path.stem + "_reg.onnx")
    try:
        onx_clf = to_onnx(clf, initial_types=initial_type, target_opset=15,
                          options={"zipmap": False})
        with open(clf_path, "wb") as f:
            f.write(onx_clf.SerializeToString())
        print(f"  [ONNX] clf -> {clf_path} ({clf_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  [ONNX] clf 导出失败: {e}")

    try:
        onx_reg = to_onnx(reg, initial_types=initial_type, target_opset=15)
        with open(reg_path, "wb") as f:
            f.write(onx_reg.SerializeToString())
        print(f"  [ONNX] reg -> {reg_path} ({reg_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"  [ONNX] reg 导出失败: {e}")
        reg_path = None

    # 2) 量化 (需 onnxruntime)
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            for src, tag in [(clf_path, "clf"), (reg_path, "reg")]:
                if src is None or not src.exists():
                    continue
                dst = src.with_name(src.stem + "_qint8.onnx")
                quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
                print(f"  [ONNX 量化] {tag} -> {dst} ({dst.stat().st_size/1024:.1f} KB, "
                      f"-{(1-dst.stat().st_size/src.stat().st_size)*100:.0f}%)")
        except ImportError:
            print("  [ONNX 量化] onnxruntime 未安装, 跳过量化 (pip install onnxruntime)")
        except Exception as e:
            print(f"  [ONNX 量化] 失败: {e}")

    return clf_path


def export_c(bundle, out_path: Path):
    """使用 m2cgen 导出纯 C 代码 (仅 clf + reg fallback, 不含 MoE/L4/L5)"""
    try:
        import m2cgen as m2c
    except ImportError:
        print("  [C 导出] m2cgen 未安装, 跳过 (pip install m2cgen)")
        return None

    clf = bundle.get("clf")
    reg = bundle.get("reg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        c_code = "// NILM Stage-1 Classifier (auto-generated by m2cgen, v14)\n"
        c_code += "// Note: 不含 MoE/L4/L5/守卫逻辑, 需要在 MCU 端自行实现\n\n"
        c_code += m2c.export_to_c(clf)
        c_code += "\n\n// ---- Stage-2 P50 Regressor (fallback) ----\n"
        c_code += m2c.export_to_c(reg)
        out_path.write_text(c_code, encoding="utf-8")
        print(f"  [C 导出] -> {out_path} ({len(c_code)/1024:.1f} KB)")
    except Exception as e:
        print(f"  [C 导出] 失败: {e}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="[v14] NILM 模型卡生成 + 模型导出")
    ap.add_argument("--model", type=str, required=True, help="模型 pkl 路径")
    ap.add_argument("--out", type=str, default=None, help="模型卡 Markdown 输出路径")
    ap.add_argument("--export-onnx", type=str, default=None, help="导出 ONNX 路径")
    ap.add_argument("--quantize", action="store_true", help="ONNX 动态 INT8 量化")
    ap.add_argument("--export-c", type=str, default=None, help="导出纯 C 代码路径")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    bundle = load_model(model_path)
    print(f"  版本: {bundle.get('version', '?')}, 特征数: {len(bundle.get('feat_names', []))}")

    # 1. 模型卡
    if args.out:
        out_md = Path(args.out)
        generate_model_card(bundle, model_path, out_md)
    else:
        md = generate_model_card(bundle, model_path)
        print("=" * 60)
        print(md[:2000], "..." if len(md) > 2000 else "")

    # 2. ONNX
    if args.export_onnx:
        export_onnx(bundle, Path(args.export_onnx), quantize=args.quantize)

    # 3. C 代码
    if args.export_c:
        export_c(bundle, Path(args.export_c))


if __name__ == "__main__":
    main()
