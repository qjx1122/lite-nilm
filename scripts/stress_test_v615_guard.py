"""
v6.15.0 守卫阈值鲁棒性压力测试
=================================
目的: 模拟 6 个极端场景, 验证 v6.15 自适应公式行为合理 (不崩溃 + 阈值落在物理可解释区间)

场景设计:
  S1: 极小样本 (n_on=3, n_off=100)     ---- 用户上线初期
  S2: OFF 含伪启动尖峰 (n_off=2000, P99=180) ---- 大功率热水器等干扰
  S3: ON/OFF 完美分离 (P10=200, P99=30)  ---- 理想空调
  S4: ON/OFF 严重重叠 (P10=90, P99=85)   ---- 难分离场景
  S5: 极大样本 (n_on=200, n_off=10000)  ---- 长期累积训练
  S6: 零 OFF (n_off=0)                  ---- 训练集异常 (全天开机)
"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/nilm_ac_win/scripts')
from common import (
    GUARD_AF_MIN, GUARD_AF_MAX, GUARD_AF_N_REF,
    GUARD_MF_MIN, GUARD_MF_MAX, GUARD_MF_N_REF,
    GUARD_SOFTMAX_TEMP_W, GUARD_SOFTMAX_PIVOT_W,
    GUARD_PROB_GAMMA, GUARD_PROB_MIN_RATIO,
)

def _sqrt_interp(n, n_ref, vmin, vmax):
    ratio = max(0.0, min(1.0, n / max(1, n_ref)))
    return vmin + (vmax - vmin) * float(np.sqrt(ratio))

def compute_v615(on_p10, off_p99, n_on, n_off):
    """完整 v6.15.0 阈值计算"""
    AF = _sqrt_interp(n_on,  GUARD_AF_N_REF, GUARD_AF_MIN, GUARD_AF_MAX)
    MF = _sqrt_interp(n_off, GUARD_MF_N_REF, GUARD_MF_MIN, GUARD_MF_MAX)
    on_c  = on_p10  * AF
    off_c = off_p99 * MF
    gap = abs(on_c - off_c)
    w = 1.0 / (1.0 + np.exp(-(gap - GUARD_SOFTMAX_PIVOT_W) / GUARD_SOFTMAX_TEMP_W))
    hard_max = max(on_c, off_c)
    mean_two = (on_c + off_c) / 2.0
    th = w * hard_max + (1-w) * mean_two
    bind = "ON" if on_c >= off_c else "OFF"
    return {"AF": AF, "MF": MF, "on_c": on_c, "off_c": off_c,
            "gap": gap, "w": w, "th": th, "bind": bind}

def compute_v6142(on_p10, off_p99):
    """v6.14.2 旧公式"""
    on_c = on_p10 * 0.9
    off_c = off_p99 * 1.25
    th = max(on_c, off_c)
    return {"on_c": on_c, "off_c": off_c, "th": th,
            "bind": "ON" if on_c >= off_c else "OFF"}

def runtime_fusion(th_base, p_on_arr):
    """方案 C 推理时局部阈值范围"""
    local = th_base * np.clip(1.0 - GUARD_PROB_GAMMA * p_on_arr, GUARD_PROB_MIN_RATIO, 1.0)
    return local.min(), local.max()

# 场景定义
scenarios = [
    ("S1 极小样本",      90,  60, 3,   100,  "用户上线初期, 仅 1-2 天数据"),
    ("S2 OFF 伪启动",    120, 180, 50,  2000, "训练集混入大功率电器干扰"),
    ("S3 ON/OFF分离",    200, 30, 80,  4000, "理想空调, 启动信号显著"),
    ("S4 严重重叠",      90,  85, 30,  1500, "弱空调 + 强 OFF 噪声"),
    ("S5 极大样本",      150, 50, 200, 10000,"长期累积训练"),
    ("S6 零 OFF",        100, 0.0, 30, 0,    "训练期全部开机"),
]

print("="*120)
print(f"{'场景':<14} {'描述':<22} | {'v6.14.2 阈值':<14} {'v6.15 阈值':<12} {'AF':<6} {'MF':<6} {'w':<6} {'绑定':<5} {'融合范围':<16}")
print("="*120)

p_on_test = np.array([0.0, 0.3, 0.6, 0.9, 1.0])
for name, on_p10, off_p99, n_on, n_off, desc in scenarios:
    try:
        r15 = compute_v615(on_p10, off_p99, n_on, n_off)
        r14 = compute_v6142(on_p10, off_p99)
        lo, hi = runtime_fusion(r15["th"], p_on_test)
        print(f"{name:<14} {desc:<22} | "
              f"{r14['th']:>6.1f}W {r14['bind']:<5} "
              f"{r15['th']:>6.1f}W   {r15['AF']:<6.3f} {r15['MF']:<6.3f} {r15['w']:<6.3f} "
              f"{r15['bind']:<5} [{lo:.1f},{hi:.1f}]")
    except Exception as e:
        print(f"{name:<14} {desc:<22} | ❌ 异常: {type(e).__name__}: {e}")

print()
print("=== 关键鲁棒性检查 ===")
# S6: 零 OFF
r = compute_v615(100, 0.0, 30, 0)
print(f"S6 零 OFF: 阈值={r['th']:.1f}W (应=on_p10×AF_MIN={100*GUARD_AF_MIN:.0f}=75W, 不应崩)")

# AF/MF 边界
print(f"AF 极限: n=0 -> AF={_sqrt_interp(0, GUARD_AF_N_REF, GUARD_AF_MIN, GUARD_AF_MAX):.3f} "
      f"(应={GUARD_AF_MIN}); n=999 -> AF={_sqrt_interp(999, GUARD_AF_N_REF, GUARD_AF_MIN, GUARD_AF_MAX):.3f} "
      f"(应={GUARD_AF_MAX})")
print(f"MF 极限: n=0 -> MF={_sqrt_interp(0, GUARD_MF_N_REF, GUARD_MF_MIN, GUARD_MF_MAX):.3f} "
      f"(应={GUARD_MF_MIN}); n=99999 -> MF={_sqrt_interp(99999, GUARD_MF_N_REF, GUARD_MF_MIN, GUARD_MF_MAX):.3f} "
      f"(应={GUARD_MF_MAX})")

# 软最大平滑边界
print(f"软最大 w: gap=0 -> w={1/(1+np.exp(-(0-GUARD_SOFTMAX_PIVOT_W)/GUARD_SOFTMAX_TEMP_W)):.3f}; "
      f"gap=100 -> w={1/(1+np.exp(-(100-GUARD_SOFTMAX_PIVOT_W)/GUARD_SOFTMAX_TEMP_W)):.3f}")

# 概率融合边界
print(f"概率融合: p=0 -> 阈值×{1.0 - GUARD_PROB_GAMMA*0:.2f}; "
      f"p=1 -> 阈值×{max(GUARD_PROB_MIN_RATIO, 1.0 - GUARD_PROB_GAMMA*1):.2f} (受 min_ratio={GUARD_PROB_MIN_RATIO} 保护)")
