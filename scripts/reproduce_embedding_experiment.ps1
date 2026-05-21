
# 1. 环境设置
# 使用相对路径以避免中文编码问题
$DATASET_PDF = "dataset_PDF"
$DATASET_TXT = "dataset_TXT"
$OUTPUT_DIR = "processed_repro"
$MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 3. 运行预处理
Write-Host "Starting Preprocessing..." -ForegroundColor Green
if (Test-Path $OUTPUT_DIR) {
    Write-Host "Cleaning old data directory..."
    Remove-Item -Path $OUTPUT_DIR -Recurse -Force
}

python preprocess_apt_dataset.py `
    --pdf-root $DATASET_PDF `
    --txt-root $DATASET_TXT `
    --output-dir $OUTPUT_DIR `
    --only-txt `
    --embed-text `
    --embed-model $MODEL_NAME `
    --use-tfidf `
    --tfidf-dim 300 `
    --min-entities 3

if ($LASTEXITCODE -ne 0) {
    Write-Error "Preprocessing failed!"
    exit 1
}

# 4. 验证生成的特征维度
Write-Host "Verifying Feature Dimensions..." -ForegroundColor Green
$verify_script = @"
import torch
import os
try:
    data = torch.load(os.path.join('$OUTPUT_DIR', 'graphs.pt'), weights_only=False)
    if len(data) > 0:
        feat_dim = data[0].x.shape[1]
        print(f'Feature Dimension: {feat_dim}')
        # 期望维度: ~911-917 (31 Type + 64*3 Hash + 4 Stats + 384 Emb + 300 TFIDF)
        # 如果是 525-531，说明 Embedding 没加上 (为0或被跳过)
        if feat_dim > 600:
            print('SUCCESS: High dimensional features detected (Embedding likely present).')
        else:
            print('WARNING: Low dimensional features detected. Embedding might be missing!')
    else:
        print('Error: No graphs found.')
except Exception as e:
    print(f'Verification failed: {e}')
"@

python -c $verify_script

# 5. 运行训练 (如果预处理成功)
# 使用之前的超参数配置 (假设是 GraphSAGE)
Write-Host "Starting Training..." -ForegroundColor Green
python train_gnn.py `
    --graphs-path "$OUTPUT_DIR/graphs.pt" `
    --label-mapping-path "$OUTPUT_DIR/label_mapping.json" `
    --model-type GraphSAGE `
    --hidden-dim 128 `
    --epochs 200 `
    --batch-size 32 `
    --lr 0.001 `
    --weight-decay 5e-4 `
    --dropout 0.3 `
    --cv-folds 5 `
    --output-dir "results_repro_sage"

Write-Host "Reproduction Pipeline Completed." -ForegroundColor Green
