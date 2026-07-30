@echo off
REM ==============================================================================
REM  NILM-AC v5 一键运行脚本 (Windows CMD / PowerShell 双兼容)
REM  v5 新增: 温度特征 (Open-Meteo Archive API) + 温度驱动季节路由
REM
REM  使用前提:
REM      1) 已创建 conda 环境:  conda env create -f environment.yml
REM      2) 已激活 conda 环境:  conda activate nilm_ac
REM      3) data\ 目录下已放入原始 CSV (总线 + 分路)
REM      4) 首次运行需联网拉取气象数据 (后续自动走本地缓存)
REM
REM  本脚本将依次执行:
REM      Step 1/8  数据合并  (如有多份历史数据)
REM      Step 2/8  数据勘察
REM      Step 3/8  时间对齐 + 特征筛选
REM      Step 4/8  [v5] 拉取气象数据并缓存
REM      Step 5/8  模型训练 v5 (含温度特征) + 保存
REM      Step 6/8  模型训练 v4.2 (无温度对照基线) + 保存
REM      Step 7/8  测试集评估 + 可视化
REM      Step 8/8  独立推理 Demo
REM ==============================================================================
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 切换到脚本所在目录 (项目根目录)
cd /d "%~dp0"

REM ---------- 环境检查 ----------
if "%CONDA_DEFAULT_ENV%"=="" (
    echo [WARN] 当前未检测到激活的 conda 环境
    echo        建议先执行: conda activate nilm_ac
    echo.
    choice /c YN /m "是否继续运行 (Y=继续 / N=退出)"
    if errorlevel 2 exit /b 0
)

REM ---------- 数据文件检查 ----------
if not exist "data\merged_bus.csv" (
    if not exist "data\*.csv" (
        echo [ERROR] data\ 目录下找不到任何 CSV 文件
        echo         请先将原始总线/分路 CSV 放入 data\ 目录
        exit /b 1
    )
    echo [INFO] 未检测到 merged_bus.csv, 将执行数据合并
    set NEED_MERGE=1
) else (
    echo [INFO] 已检测到 merged_bus.csv, 跳过合并步骤
    set NEED_MERGE=0
)

REM ============================================================
echo.
echo ============================================================
echo  NILM-AC v5 训练 + 评估 + 推理 全流程
echo  开始时间: %DATE% %TIME%
echo ============================================================

REM ---------- Step 1: 数据合并 (条件执行) ----------
if "%NEED_MERGE%"=="1" (
    echo.
    echo ============================================================
    echo  Step 1/8  数据合并
    echo ============================================================
    python scripts\merge_data.py
    if errorlevel 1 goto :error
) else (
    echo.
    echo ============================================================
    echo  Step 1/8  数据合并 [跳过]
    echo ============================================================
)

REM ---------- Step 2: 数据勘察 ----------
echo.
echo ============================================================
echo  Step 2/8  数据勘察
echo ============================================================
python scripts\01_audit.py
if errorlevel 1 goto :error

REM ---------- Step 3: 对齐 + 特征 ----------
echo.
echo ============================================================
echo  Step 3/8  时间对齐 + 特征相关性
echo ============================================================
python scripts\02_align_and_feat.py
if errorlevel 1 goto :error

REM ---------- Step 4: [v5] 拉取气象数据 ----------
echo.
echo ============================================================
echo  Step 4/8  [v5] 拉取气象数据 (首次联网, 后续走缓存)
echo ============================================================
python scripts\fetch_weather.py
if errorlevel 1 (
    echo [WARN] 气象数据拉取失败, v5 训练将启用降级模式 (经验气温)
)

REM ---------- Step 5: 训练 v5 ----------
echo.
echo ============================================================
echo  Step 5/8  模型训练 v5 (含温度特征 + 温度路由) + 保存
echo ============================================================
python scripts\03_train.py
if errorlevel 1 goto :error

REM ---------- Step 6: 训练 v4.2 对照基线 ----------
echo.
echo ============================================================
echo  Step 6/8  模型训练 v4.2 基线对照 (无温度特征)
echo ============================================================
python scripts\03b_train_v42_baseline.py
if errorlevel 1 (
    echo [WARN] v4.2 基线训练失败, 跳过对照评估
)

REM ---------- Step 7: 测试集评估 ----------
echo.
echo ============================================================
echo  Step 7/8  加载模型 + 测试集评估 + 可视化
echo ============================================================
python scripts\04_evaluate.py
if errorlevel 1 goto :error

REM ---------- Step 8: 独立推理 ----------
echo.
echo ============================================================
echo  Step 8/8  独立推理 Demo (默认使用 data\ 下的 CSV)
echo ============================================================
python scripts\05_inference.py
if errorlevel 1 goto :error

REM ============================================================
echo.
echo ============================================================
echo  [SUCCESS] 全流程完成
echo  完成时间: %DATE% %TIME%
echo ============================================================
echo.
echo  关键产物:
echo    - v5 主模型      : models\nilm_ac_two_stage.pkl       (含温度特征)
echo    - v4.2 对照模型  : models\nilm_ac_two_stage_v42.pkl   (无温度, 基线)
echo    - 训练备份       : models\nilm_ac_two_stage_*.pkl
echo    - 模型元数据     : models\model_meta.json
echo    - 气象缓存       : data\weather_cache\*.csv
echo    - 评估指标       : artifacts\metrics\metrics_pivot.csv
echo    - 阈值曲线       : artifacts\metrics\threshold_curve_val.csv
echo    - MoE 专家摘要   : artifacts\metrics\expert_summary.csv
echo    - 预测明细       : artifacts\predictions\*.csv
echo    - 可视化         : artifacts\test_prediction.png
echo    - 特征重要性     : artifacts\feat_importance.png
echo    - 详细日志       : logs\*.log
echo.
echo  下一步建议:
echo    1) 查看指标:  notepad artifacts\metrics\metrics_pivot.csv
echo    2) 查看可视化: start artifacts\test_prediction.png
echo    3) 用新数据推理:
echo       python scripts\05_inference.py --bus 您的总线.csv --branch 您的分路.csv
echo ============================================================
goto :eof

:error
echo.
echo ============================================================
echo  [ERROR] 运行失败, 错误码=%errorlevel%
echo  请查看 logs\ 目录下最新的 .log 文件定位问题
echo ============================================================
exit /b %errorlevel%
