@echo off
REM APT归因图神经网络训练脚本 (Windows)

REM 设置路径
set PROCESSED_DIR=processed
set OUTPUT_DIR=models\gnn_output

REM 创建输出目录
if not exist "%OUTPUT_DIR%\gat" mkdir "%OUTPUT_DIR%\gat"

REM 训练GAT模型（推荐）
echo 开始训练GAT模型...
python train_gnn.py ^
    --graphs-path "%PROCESSED_DIR%\graphs.pt" ^
    --label-mapping-path "%PROCESSED_DIR%\label_mapping.json" ^
    --output-dir "%OUTPUT_DIR%\gat" ^
    --model-type GAT ^
    --hidden-dim 128 ^
    --num-layers 2 ^
    --heads 8 ^
    --dropout 0.6 ^
    --batch-size 32 ^
    --epochs 100 ^
    --lr 0.001 ^
    --patience 20 ^
    --device auto

echo 训练完成！结果保存在: %OUTPUT_DIR%\gat
pause









