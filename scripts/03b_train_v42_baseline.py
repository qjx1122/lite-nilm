# -*- coding: utf-8 -*-
"""
Step 3b: v4.2 基线对照训练 (无温度特征)

通过环境变量 NILM_BASELINE_MODE=1 让 03_train.py 自动:
    - 禁用 USE_WEATHER_FEATURES
    - 禁用 USE_TEMP_BASED_SEASON
    - 保存模型到 MODEL_V42_PKL (而非主模型路径)

用法 (Windows):
    python scripts\03b_train_v42_baseline.py
"""
import os
import sys
import subprocess
from pathlib import Path

env = os.environ.copy()
env["NILM_BASELINE_MODE"] = "1"

script_dir = Path(__file__).resolve().parent
train_py   = script_dir / "03_train.py"

print("=" * 72)
print("Step 3b: 启动 v4.2 基线对照训练 (NILM_BASELINE_MODE=1)")
print(f"  调用: python {train_py}")
print(f"  特点: 无温度特征 + 月份硬路由, 输出到 models/nilm_ac_two_stage_v42.pkl")
print("=" * 72)

ret = subprocess.run([sys.executable, str(train_py)], env=env,
                     cwd=str(script_dir.parent))
sys.exit(ret.returncode)
