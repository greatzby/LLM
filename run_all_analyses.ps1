# PowerShell脚本 - 运行所有分析
# 使用方法：在PowerShell中运行 .\run_all_analyses.ps1

# 设置参数
$NUM_NODES = 100
$CONFIG = "1_1_120"
$OUTPUT_DIR = "analysis_results"

# 创建输出目录
Write-Host "Creating output directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host "`nStarting comprehensive analysis pipeline..." -ForegroundColor Cyan

# 1. 运行综合分析
Write-Host "`n1. Running comprehensive analysis..." -ForegroundColor Yellow
python comprehensive_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR

# 检查是否成功
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in comprehensive analysis!" -ForegroundColor Red
    exit 1
}

# 2. 运行位置敏感度分析
Write-Host "`n2. Running position sensitivity analysis..." -ForegroundColor Yellow
python position_sensitivity_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 1000 5000 10000 20000 30000 40000 50000 70000 100000

# 3. 运行context降级测试
Write-Host "`n3. Running context degradation test..." -ForegroundColor Yellow
python context_degradation_test.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 10000 30000 50000 70000 100000

# 4. 运行错误模式分析
Write-Host "`n4. Running error pattern analysis..." -ForegroundColor Yellow
python error_pattern_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 1000 5000 10000 20000 30000 40000 50000 70000 100000

# 5. 运行分布分析
Write-Host "`n5. Running distribution analysis..." -ForegroundColor Yellow
python distribution_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 10000 30000 50000 70000 100000

# 6. 创建最终可视化
Write-Host "`n6. Creating final visualizations..." -ForegroundColor Yellow
python create_final_visualizations.py --output_dir $OUTPUT_DIR

# 完成
Write-Host "`nAll analyses complete!" -ForegroundColor Green
Write-Host "Results saved in $OUTPUT_DIR/" -ForegroundColor Green

# 显示输出文件
Write-Host "`nGenerated files:" -ForegroundColor Cyan
Get-ChildItem -Path $OUTPUT_DIR -Recurse -File | Select-Object FullName | Format-Table -AutoSize

# 暂停以查看结果
Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")