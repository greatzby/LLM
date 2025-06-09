#!/bin/bash
# Bash脚本 - 运行所有分析
# 使用方法：chmod +x run_all_analyses.sh && ./run_all_analyses.sh

# 设置参数
NUM_NODES=100
CONFIG="1_1_120"
OUTPUT_DIR="analysis_results"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 创建输出目录
echo -e "${GREEN}Creating output directory...${NC}"
mkdir -p $OUTPUT_DIR

echo -e "\n${CYAN}Starting comprehensive analysis pipeline...${NC}"

# 1. 运行综合分析
echo -e "\n${YELLOW}1. Running comprehensive analysis...${NC}"
python comprehensive_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR

# 检查是否成功
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in comprehensive analysis!${NC}"
    exit 1
fi

# 2. 运行位置敏感度分析
echo -e "\n${YELLOW}2. Running position sensitivity analysis...${NC}"
python position_sensitivity_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 1000 5000 10000 20000 30000 40000 50000 70000 100000

# 3. 运行context降级测试
echo -e "\n${YELLOW}3. Running context degradation test...${NC}"
python context_degradation_test.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 10000 30000 50000 70000 100000

# 4. 运行错误模式分析
echo -e "\n${YELLOW}4. Running error pattern analysis...${NC}"
python error_pattern_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 1000 5000 10000 20000 30000 40000 50000 70000 100000

# 5. 运行分布分析
echo -e "\n${YELLOW}5. Running distribution analysis...${NC}"
python distribution_analysis.py --num_nodes $NUM_NODES --config $CONFIG --output_dir $OUTPUT_DIR --checkpoints 10000 30000 50000 70000 100000

# 6. 创建最终可视化
echo -e "\n${YELLOW}6. Creating final visualizations...${NC}"
python create_final_visualizations.py --output_dir $OUTPUT_DIR

# 完成
echo -e "\n${GREEN}All analyses complete!${NC}"
echo -e "${GREEN}Results saved in $OUTPUT_DIR/${NC}"

# 显示输出文件
echo -e "\n${CYAN}Generated files:${NC}"
find $OUTPUT_DIR -type f -name "*.png" -o -name "*.json" -o -name "*.csv" -o -name "*.tex" | sort

echo -e "\n${GREEN}Analysis pipeline finished successfully!${NC}"