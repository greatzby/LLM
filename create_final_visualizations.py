"""
综合可视化脚本 - 生成所有分析结果的综合可视化
运行方式: python create_final_visualizations.py --output_dir analysis_results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.gridspec import GridSpec

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    return parser.parse_args()

def load_all_results(output_dir):
    """加载所有分析结果"""
    results = {}
    
    # 加载各个分析的结果
    files = {
        'comprehensive': 'comprehensive_analysis.json',
        'position': 'position_analysis/position_sensitivity_results.json',
        'degradation': 'context_degradation/context_degradation_results.json',
        'errors': 'error_patterns/error_pattern_results.json',
        'distribution': 'distribution_analysis/distribution_metrics.json'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
            print(f"Loaded {key} results")
        else:
            print(f"Warning: {filepath} not found")
    
    return results

def create_summary_dashboard(results, output_dir):
    """创建综合仪表板"""
    # 设置样式
    plt.style.use('seaborn-v0_8')
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Weight evolution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'comprehensive' in results:
        data = results['comprehensive']
        iterations = sorted([int(k) for k in data.keys()])
        
        edge_weights = []
        non_edge_weights = []
        weight_gaps = []
        
        for i in iterations:
            if str(i) in data:
                edge_weights.append(data[str(i)]['weight_stats']['edge_mean'])
                non_edge_weights.append(data[str(i)]['weight_stats']['non_edge_mean'])
                weight_gaps.append(data[str(i)]['weight_stats']['weight_gap'])
        
        ax1.plot(iterations, edge_weights, 'b-', label='Edge weights', linewidth=2)
        ax1.plot(iterations, non_edge_weights, 'r-', label='Non-edge weights', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Weight')
        ax1.set_title('Weight Evolution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加转换点标记
        transition_point = None
        for i, (iter_num, gap) in enumerate(zip(iterations[:-1], weight_gaps[:-1])):
            if weight_gaps[i] > 0 and weight_gaps[i+1] < weight_gaps[i]:
                transition_point = iter_num
                ax1.axvline(x=iter_num, color='green', linestyle=':', alpha=0.7, label='Transition')
                break
    
    # 2. Context sensitivity
    ax2 = fig.add_subplot(gs[0, 1])
    if 'degradation' in results:
        data = results['degradation']
        
        for ckpt in sorted([int(k) for k in data.keys()]):
            noise_levels = []
            accuracies = []
            
            for noise_key in sorted(data[str(ckpt)].keys()):
                if noise_key.startswith('noise_'):
                    noise_level = float(noise_key.split('_')[1])
                    noise_levels.append(noise_level)
                    accuracies.append(data[str(ckpt)][noise_key]['mean_accuracy'])
            
            ax2.plot(noise_levels, accuracies, marker='o', label=f'Iter {ckpt//1000}k', linewidth=2)
        
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Context Noise Sensitivity', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
    
    # 3. Error evolution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'errors' in results:
        data = results['errors']
        checkpoints = sorted([int(k) for k in data.keys()])
        
        # 只显示正确率
        correct_pcts = [data[str(ckpt)].get('correct', 0) for ckpt in checkpoints]
        
        ax3.plot(checkpoints, correct_pcts, 'g-', marker='o', linewidth=3, markersize=8)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Correct %')
        ax3.set_title('Model Accuracy Evolution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
        
        # 标注关键点
        max_acc_idx = np.argmax(correct_pcts)
        ax3.annotate(f'Peak: {correct_pcts[max_acc_idx]:.1f}%', 
                    xy=(checkpoints[max_acc_idx], correct_pcts[max_acc_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 4. Embedding norm evolution
    ax4 = fig.add_subplot(gs[1, 0])
    if 'comprehensive' in results:
        data = results['comprehensive']
        iterations = sorted([int(k) for k in data.keys()])
        
        norms = []
        norm_stds = []
        
        for i in iterations:
            if str(i) in data:
                norms.append(data[str(i)]['embedding_stats']['norm_mean'])
                norm_stds.append(data[str(i)]['embedding_stats']['norm_std'])
        
        ax4.plot(iterations, norms, 'purple', marker='o', linewidth=2)
        ax4.fill_between(iterations, 
                        np.array(norms) - np.array(norm_stds),
                        np.array(norms) + np.array(norm_stds),
                        alpha=0.3, color='purple')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Mean Embedding Norm')
        ax4.set_title('Embedding Scale Evolution', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # 添加增长率标注
        if len(norms) > 1:
            growth_rate = (norms[-1] / norms[0] - 1) * 100
            ax4.text(0.95, 0.05, f'Growth: {growth_rate:.1f}%', 
                    transform=ax4.transAxes, ha='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # 5. Position sensitivity heatmap
    ax5 = fig.add_subplot(gs[1, 1:])
    if 'position' in results:
        data = results['position']
        checkpoints = sorted([int(k) for k in data.keys()])
        positions = [3, 4, 5, 6, 7, 8]
        
        entropy_matrix = []
        for ckpt in checkpoints:
            row = []
            for pos in positions:
                key = f'position_{pos}'
                if str(ckpt) in data and key in data[str(ckpt)]:
                    row.append(data[str(ckpt)][key]['entropy_mean'])
                else:
                    row.append(np.nan)
            entropy_matrix.append(row)
        
        im = ax5.imshow(entropy_matrix, aspect='auto', cmap='viridis')
        ax5.set_xticks(range(len(positions)))
        ax5.set_xticklabels(positions)
        ax5.set_yticks(range(len(checkpoints)))
        ax5.set_yticklabels([f'{ckpt//1000}k' for ckpt in checkpoints])
        ax5.set_xlabel('Position')
        ax5.set_ylabel('Checkpoint')
        ax5.set_title('Prediction Entropy by Position', fontsize=14)
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Entropy')
    
    # 6. Distribution metrics comparison
    ax6 = fig.add_subplot(gs[2, :])
    if 'distribution' in results:
        data = results['distribution']
        checkpoints = sorted([int(k) for k in data.keys()])
        
        metrics_to_plot = {
            'entropy': 'Entropy',
            'top1_prob': 'Top-1 Prob',
            'effective_choices': 'Eff. Choices'
        }
        
        x = np.arange(len(checkpoints))
        width = 0.25
        
        for i, (metric, label) in enumerate(metrics_to_plot.items()):
            values = []
            for ckpt in checkpoints:
                if str(ckpt) in data and metric in data[str(ckpt)]:
                    values.append(data[str(ckpt)][metric]['mean'])
                else:
                    values.append(0)
            
            ax6.bar(x + i*width, values, width, label=label)
        
        ax6.set_xlabel('Checkpoint')
        ax6.set_ylabel('Value')
        ax6.set_title('Distribution Metrics Evolution', fontsize=14)
        ax6.set_xticks(x + width)
        ax6.set_xticklabels([f'{ckpt//1000}k' for ckpt in checkpoints])
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Error type breakdown (最后一个checkpoint)
    ax7 = fig.add_subplot(gs[3, 0])
    if 'errors' in results:
        data = results['errors']
        last_checkpoint = max([int(k) for k in data.keys()])
        
        if str(last_checkpoint) in data:
            error_data = data[str(last_checkpoint)]
            
            # 过滤掉'correct'和值为0的错误类型
            labels = []
            sizes = []
            for error_type, percentage in error_data.items():
                if error_type != 'correct' and percentage > 0:
                    labels.append(error_type.replace('_', ' ').title())
                    sizes.append(percentage)
            
            if sizes:
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                ax7.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax7.set_title(f'Error Distribution at {last_checkpoint//1000}k', fontsize=14)
            else:
                ax7.text(0.5, 0.5, 'No errors', ha='center', va='center', fontsize=16)
                ax7.set_xlim(0, 1)
                ax7.set_ylim(0, 1)
    
    # 8. Key metrics summary table
    ax8 = fig.add_subplot(gs[3, 1:])
    ax8.axis('tight')
    ax8.axis('off')
    
    # 创建汇总表
    summary_data = []
    
    if 'comprehensive' in results:
        last_iter = max([int(k) for k in results['comprehensive'].keys()])
        data = results['comprehensive'][str(last_iter)]
        
        summary_data.append(['Final Iteration', f'{last_iter:,}'])
        summary_data.append(['Embedding Norm', f"{data['embedding_stats']['norm_mean']:.3f}"])
        summary_data.append(['Weight Gap', f"{data['weight_stats']['weight_gap']:.3f}"])
    
    if 'errors' in results:
        last_checkpoint = max([int(k) for k in results['errors'].keys()])
        correct_pct = results['errors'][str(last_checkpoint)].get('correct', 0)
        summary_data.append(['Final Accuracy', f'{correct_pct:.1f}%'])
    
    if 'distribution' in results:
        last_checkpoint = max([int(k) for k in results['distribution'].keys()])
        if str(last_checkpoint) in results['distribution']:
            entropy = results['distribution'][str(last_checkpoint)]['entropy']['mean']
            summary_data.append(['Final Entropy', f'{entropy:.3f}'])
    
    if summary_data:
        table = ax8.table(cellText=summary_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.suptitle('Comprehensive Analysis Dashboard', fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_latex_tables(results, output_dir):
    """创建LaTeX格式的表格"""
    
    # 1. 综合统计表
    if 'comprehensive' in results:
        data = results['comprehensive']
        iterations = sorted([int(k) for k in data.keys()])
        
        # 选择关键checkpoint
        key_checkpoints = [1000, 10000, 30000, 50000, 70000, 100000]
        key_checkpoints = [ckpt for ckpt in key_checkpoints if ckpt in iterations]
        
        rows = []
        for i in key_checkpoints:
            if str(i) in data:
                row = {
                    'Iteration': f'{i:,}',
                    'Edge Weight': f"{data[str(i)]['weight_stats']['edge_mean']:.4f}",
                    'Non-edge Weight': f"{data[str(i)]['weight_stats']['non_edge_mean']:.4f}",
                    'Weight Gap': f"{data[str(i)]['weight_stats']['weight_gap']:.4f}",
                    'Embedding Norm': f"{data[str(i)]['embedding_stats']['norm_mean']:.3f}"
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        latex_table = df.to_latex(index=False, escape=False, column_format='l' + 'r'*4)
        
        with open(os.path.join(output_dir, 'weight_evolution_table.tex'), 'w') as f:
            f.write("% Weight and Embedding Evolution\n")
            f.write(latex_table)
    
    # 2. 性能指标表
    perf_rows = []
    
    if 'errors' in results and 'distribution' in results:
        checkpoints = sorted(list(set(
            [int(k) for k in results['errors'].keys()] + 
            [int(k) for k in results['distribution'].keys()]
        )))
        
        for ckpt in checkpoints:
            row = {'Iteration': f'{ckpt:,}'}
            
            if str(ckpt) in results['errors']:
                row['Accuracy (\\%)'] = f"{results['errors'][str(ckpt)].get('correct', 0):.1f}"
            else:
                row['Accuracy (\\%)'] = '-'
            
            if str(ckpt) in results['distribution']:
                dist_data = results['distribution'][str(ckpt)]
                row['Entropy'] = f"{dist_data['entropy']['mean']:.3f}"
                row['Top-1 Prob'] = f"{dist_data['top1_prob']['mean']:.3f}"
            else:
                row['Entropy'] = '-'
                row['Top-1 Prob'] = '-'
            
            perf_rows.append(row)
    
    if perf_rows:
        df = pd.DataFrame(perf_rows)
        latex_table = df.to_latex(index=False, escape=False, column_format='l' + 'r'*3)
        
        with open(os.path.join(output_dir, 'performance_metrics_table.tex'), 'w') as f:
            f.write("% Performance Metrics Evolution\n")
            f.write(latex_table)

def create_final_plots(results, output_dir):
    """创建最终的专业图表"""
    
    # 设置出版质量的参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    # 1. 三阶段演化图
    if 'comprehensive' in results and 'errors' in results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # 获取数据
        comp_data = results['comprehensive']
        error_data = results['errors']
        
        iterations = sorted([int(k) for k in comp_data.keys()])
        
        # Weight gap
        weight_gaps = [comp_data[str(i)]['weight_stats']['weight_gap'] for i in iterations]
        ax1.plot(iterations, weight_gaps, 'b-', linewidth=2, label='Weight Gap')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Weight Gap')
        ax1.set_title('Model Evolution: Three Stages')
        ax1.grid(True, alpha=0.3)
        
        # 添加阶段标记
        stage1_end = 20000
        stage2_end = 50000
        
        ax1.axvspan(0, stage1_end, alpha=0.1, color='green', label='Stage 1')
        ax1.axvspan(stage1_end, stage2_end, alpha=0.1, color='yellow', label='Stage 2')
        ax1.axvspan(stage2_end, max(iterations), alpha=0.1, color='red', label='Stage 3')
        ax1.legend()
        
        # Accuracy
        acc_iterations = sorted([int(k) for k in error_data.keys()])
        accuracies = [error_data[str(i)].get('correct', 0) for i in acc_iterations]
        
        ax2.plot(acc_iterations, accuracies, 'g-', linewidth=2, marker='o')
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        
        # 添加阶段标记
        ax2.axvspan(0, stage1_end, alpha=0.1, color='green')
        ax2.axvspan(stage1_end, stage2_end, alpha=0.1, color='yellow')
        ax2.axvspan(stage2_end, max(iterations), alpha=0.1, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'three_stages_evolution.png'), dpi=300)
        plt.close()

def main():
    args = parse_args()
    
    print("Loading all analysis results...")
    results = load_all_results(args.output_dir)
    
    if not results:
        print("No results found! Please run the analysis scripts first.")
        return
    
    print("Creating summary dashboard...")
    create_summary_dashboard(results, args.output_dir)
    
    print("Creating LaTeX tables...")
    create_latex_tables(results, args.output_dir)
    
    print("Creating final plots...")
    create_final_plots(results, args.output_dir)
    
    print(f"\nAll visualizations created! Check {args.output_dir}/ for outputs:")
    print(f"  - summary_dashboard.png: Comprehensive overview")
    print(f"  - three_stages_evolution.png: Three-stage model evolution")
    print(f"  - *.tex: LaTeX tables for paper")

if __name__ == "__main__":
    main()