# -*- coding: utf-8 -*-
"""
[v14 方向③] 模型轻量化分析 + MCU 部署预算估算

功能:
  1. 解析 sklearn GBDT/MoE bundle, 统计总节点数/树数/最大深度
  2. 估算 Flash/RAM 占用 (C 推理时, INT8/FP32 两种精度)
  3. 给出 STM32/ESP32/RK3568 三档 MCU/MPU 的部署可行性判定
  4. 可选: 模型剪枝 (去除 gain<阈值的树 / 节点), 导出轻量 bundle

用法:
    python scripts/v14_model_analyzer.py \
        --model models/<uid>/nilm_ac_two_stage.pkl \
        --prune --threshold-gain 0.01 \
        --out models/<uid>/nilm_ac_two_stage_lite.pkl
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


def analyze_tree(tree) -> dict:
    """分析单棵 sklearn Tree, 返回节点统计"""
    try:
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        # 叶节点: children_left[i] == -1
        n_leaves = int(np.sum(children_left == -1))
        n_internal = n_nodes - n_leaves
        max_depth = 0
        depths = np.zeros(n_nodes, dtype=int)
        stack = [(0, 0)]
        while stack:
            i, d = stack.pop()
            depths[i] = d
            max_depth = max(max_depth, d)
            if children_left[i] >= 0:
                stack.append((children_left[i], d + 1))
                stack.append((children_right[i], d + 1))
        return {
            "n_nodes": int(n_nodes),
            "n_leaves": int(n_leaves),
            "n_internal": int(n_internal),
            "max_depth": int(max_depth),
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_ensemble(est, name="") -> dict:
    """分析 GBDT/RF/LGBM 集成"""
    stats = {"name": name, "n_estimators": 0, "total_nodes": 0,
             "total_leaves": 0, "max_depth": 0, "trees": []}
    try:
        if hasattr(est, "estimators_"):
            ests = est.estimators_
            # GradientBoosting: est.estimators_ 形状 (n_est, 1) 或 (n_est,)
            for i, e in enumerate(ests):
                if hasattr(e, "tree_"):
                    t = analyze_tree(e.tree_)
                elif hasattr(e, "shape"):  # (n_est, n_classes) 形式
                    t_list = [analyze_tree(ee.tree_) for ee in e if hasattr(ee, "tree_")]
                    if not t_list:
                        continue
                    t = {"n_nodes": sum(x["n_nodes"] for x in t_list),
                         "n_leaves": sum(x["n_leaves"] for x in t_list),
                         "max_depth": max(x["max_depth"] for x in t_list)}
                else:
                    continue
                stats["trees"].append(t)
                stats["n_estimators"] += 1
                stats["total_nodes"] += t.get("n_nodes", 0)
                stats["total_leaves"] += t.get("n_leaves", 0)
                stats["max_depth"] = max(stats["max_depth"], t.get("max_depth", 0))
        elif hasattr(est, "booster_"):  # LightGBM
            dump = est.booster_.dump_model()
            for t in dump.get("tree_info", []):
                struct = t.get("tree_structure", {})

                def _walk(node):
                    if "leaf_value" in node:
                        return 1, 1, 1
                    nn = 1
                    ll, li, ld = _walk(node.get("left_child", {}))
                    rl, ri, rd = _walk(node.get("right_child", {}))
                    return (nn + ll + rl - 1, li + ri, max(ld, rd) + 1)
                tn, tl, td = _walk(struct) if struct else (0, 0, 0)
                stats["trees"].append({"n_nodes": tn, "n_leaves": tl, "max_depth": td})
                stats["n_estimators"] += 1
                stats["total_nodes"] += tn
                stats["total_leaves"] += tl
                stats["max_depth"] = max(stats["max_depth"], td)
    except Exception as e:
        stats["error"] = str(e)
    return stats


def estimate_memory(cls_stats: dict, reg_stats: dict, n_features: int,
                    precision: str = "fp32") -> dict:
    """
    估算 MCU 上纯 C 推理的内存占用

    节点 C 结构体 (fp32):
        typedef struct { int feature; float threshold; int left; int right; float value; } Node;
        单节点 = 4 + 4 + 4 + 4 + 4 = 20 bytes
    节点 C 结构体 (int8 量化):
        typedef struct { int8_t feature; int16_t thr_q; int left; int right; float value; } Node;
        单节点 ≈ 16 bytes (实际按特征索引/阈值紧凑存储, 可达 8-10 bytes; 取保守值)
    叶节点值通常存在内部节点里, 复用 value 字段
    """
    total_nodes = cls_stats.get("total_nodes", 0) + reg_stats.get("total_nodes", 0)
    if precision == "fp32":
        node_bytes = 20
        feat_buf = n_features * 4   # FP32 特征 buffer
        code_ovh = 4096             # 代码段开销
        stack = 1024                # 递归栈/后处理/状态机
    elif precision == "int8":
        node_bytes = 12             # 紧凑存储
        feat_buf = n_features * 4   # 特征仍用 float (输入归一化后)
        code_ovh = 4096
        stack = 1024
    else:
        raise ValueError(precision)
    flash = total_nodes * node_bytes + code_ovh
    ram = feat_buf + stack + 2048   # 2KB 全局变量
    return {
        "total_nodes": total_nodes,
        "node_bytes": node_bytes,
        "flash_bytes": flash,
        "ram_bytes": ram,
        "flash_KB": flash / 1024,
        "ram_KB": ram / 1024,
    }


def deployment_assessment(mem: dict) -> list:
    """三档 MCU/MPU 部署可行性评估"""
    out = []
    targets = [
        ("STM32H743 (Cortex-M7, 2MB Flash / 1MB RAM)", 2*1024*1024, 1024*1024),
        ("ESP32-S3 (Xtensa LX7, 16MB Flash / 512KB SRAM)", 16*1024*1024, 512*1024),
        ("RK3568 (Cortex-A55, Linux)", 64*1024*1024, 4*1024*1024),
        ("ESP32-C3 (RISC-V, 4MB Flash / 400KB SRAM)", 4*1024*1024, 400*1024),
    ]
    flash = mem["flash_bytes"]
    ram = mem["ram_bytes"]
    for name, fmax, rmax in targets:
        f_ok = flash < fmax * 0.7
        r_ok = ram < rmax * 0.5
        status = "✅ 可部署" if (f_ok and r_ok) else ("⚠️ 需剪枝" if f_ok else "❌ 超资源")
        detail = f"Flash {mem['flash_KB']:.0f}KB/{fmax/1024:.0f}KB, RAM {mem['ram_KB']:.0f}KB/{rmax/1024:.0f}KB"
        out.append({"target": name, "status": status, "detail": detail})
    return out


def prune_model(bundle, gain_threshold: float = 0.01) -> dict:
    """
    简单剪枝: 遍历所有 GBR/Classifier 的 estimators_, 重新训练后不能改树,
    此处仅"虚拟剪枝"报告可以去掉的低 gain 树数量, 不实际修改模型.
    (实际剪枝需要重新 fit 或手动改写 sklearn Tree 结构, 工程风险较高;
    推荐用 n_estimators 下调 + max_depth 变浅 + 重训代替)
    """
    # sklearn 没公开 gain 接口, 这里只统计可降配建议
    return {
        "suggestion": (
            f"建议: n_estimators 从默认 300/400 降到 "
            f"150/200 通常可减 50% 体积, MAE 下降 <5%. "
            f"max_depth 从 3 到 2 可再减 25% 体积."
        )
    }


def main():
    ap = argparse.ArgumentParser(description="[v14] NILM 模型轻量化分析")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--threshold-gain", type=float, default=0.01)
    ap.add_argument("--out", type=str, default=None, help="剪枝后模型输出路径 (预留)")
    args = ap.parse_args()

    import joblib
    model_path = Path(args.model)
    print(f"加载模型: {model_path}")
    sys.path.insert(0, str(_SCRIPT_DIR))
    bundle = joblib.load(model_path)
    n_features = len(bundle.get("feat_names", []))
    print(f"  特征数: {n_features}")
    print(f"  版本: {bundle.get('version','?')}")

    # 分析分类器
    clf = bundle.get("clf")
    reg = bundle.get("reg")
    moe = bundle.get("moe")

    print("\n" + "=" * 70)
    print("Stage-1 分类器 (clf) 结构分析")
    print("=" * 70)
    cls_stats = analyze_ensemble(clf, "Stage-1 Classifier")
    print(f"  树数       : {cls_stats['n_estimators']}")
    print(f"  总节点数   : {cls_stats['total_nodes']}")
    print(f"  总叶节点数 : {cls_stats['total_leaves']}")
    print(f"  最大深度   : {cls_stats['max_depth']}")

    print("\n" + "=" * 70)
    print("Stage-2 全局回归器 (fallback P50) 结构分析")
    print("=" * 70)
    reg_stats = analyze_ensemble(reg, "Stage-2 Regressor P50")
    print(f"  树数       : {reg_stats['n_estimators']}")
    print(f"  总节点数   : {reg_stats['total_nodes']}")
    print(f"  总叶节点数 : {reg_stats['total_leaves']}")
    print(f"  最大深度   : {reg_stats['max_depth']}")

    # MoE 分析
    moe_extra_nodes = 0
    moe_extra_trees = 0
    if moe is not None and hasattr(moe, "experts"):
        print("\n" + "=" * 70)
        print("Stage-2 季节 MoE 分析 (Summer/Transition/Winter × P10/P50/P90)")
        print("=" * 70)
        try:
            for s_name, experts in moe.experts.items():
                for alpha, est in experts.items():
                    s = analyze_ensemble(est, f"{s_name}_P{int(alpha*100)}")
                    print(f"  {s_name:11s} P{int(alpha*100)}: "
                          f"trees={s['n_estimators']}, nodes={s['total_nodes']}, "
                          f"depth={s['max_depth']}")
                    moe_extra_nodes += s["total_nodes"]
                    moe_extra_trees += s["n_estimators"]
        except Exception as e:
            print(f"  MoE 遍历失败: {e}")

    # 内存估算
    print("\n" + "=" * 70)
    print(f"MCU 部署内存估算 (仅 clf + fallback, 不含 MoE; precision={args.precision})")
    print("=" * 70)
    mem = estimate_memory(cls_stats, reg_stats, n_features, args.precision)
    print(f"  总节点数   : {mem['total_nodes']}")
    print(f"  Flash 估算 : {mem['flash_KB']:.1f} KB (纯模型, 不含应用代码)")
    print(f"  RAM 估算   : {mem['ram_KB']:.1f} KB (特征buffer + 栈)")
    print(f"  单样本延迟 : ~{mem['total_nodes']*0.01:.0f} ms @Cortex-M7 估算")

    print("\n若部署 MoE (3季节 × 3分位 = 9 专家):")
    mem_moe = estimate_memory(
        cls_stats,
        {"total_nodes": reg_stats["total_nodes"] + moe_extra_nodes},
        n_features, args.precision)
    print(f"  Flash 估算 : {mem_moe['flash_KB']:.1f} KB")
    print(f"  RAM 估算   : {mem_moe['ram_KB']:.1f} KB")

    # 部署评估
    print("\n" + "=" * 70)
    print("目标平台部署可行性")
    print("=" * 70)
    print("\n[仅 clf + fallback P50 (最小部署, 无 MoE/L4/L5/d87 守卫)]")
    for a in deployment_assessment(mem):
        print(f"  {a['status']}  {a['target']:55s}  {a['detail']}")

    print("\n[完整 clf + MoE (含季节路由)]")
    for a in deployment_assessment(mem_moe):
        print(f"  {a['status']}  {a['target']:55s}  {a['detail']}")

    # 剪枝建议
    if args.prune:
        print("\n" + "=" * 70)
        print("轻量化建议")
        print("=" * 70)
        sug = prune_model(bundle, args.threshold_gain)
        print(f"  {sug['suggestion']}")
        print("  注意: v14 不做原地剪枝 (风险高). 推荐通过 common.py 下调 n_estimators 重训.")


if __name__ == "__main__":
    main()
