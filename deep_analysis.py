"""
deep_analysis.py - 深入分析logit分布和softmax效应
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import networkx as nx
import re
from collections import defaultdict
import seaborn as sns

# 设置matplotlib不显示图形（只保存）
plt.ioff()
sns.set_style("whitegrid")

# 设置参数
dataset = 'simple_graph'
num_nodes = 100
n_layer = 1
n_head = 1
n_embd = 120
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 路径设置
data_dir = f'data/{dataset}/{num_nodes}'
out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'

# 创建深度分析图像保存目录
deep_analysis_dir = os.path.join(out_dir, 'deep_analysis')
os.makedirs(deep_analysis_dir, exist_ok=True)

# 加载meta信息
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
vocab_size = meta['vocab_size']

# 加载图
graph_path = os.path.join(data_dir, "path_graph.graphml")
G = nx.read_graphml(graph_path)

# 重要：建立node到token的映射
NODE_OFFSET = 2

def node_to_token(node_id):
    return node_id + NODE_OFFSET

def token_to_node(token_id):
    if token_id < NODE_OFFSET:
        return None
    return token_id - NODE_OFFSET

def get_successors(node):
    return list(G.successors(str(node)))

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def encode(s):
    ss = s.split(" ")
    return [stoi[token] for token in ss if token in stoi]

def load_test_samples():
    test_file = os.path.join(data_dir, 'test.txt')
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(("", line))
    return samples

def analyze_logit_distributions(models_dict, test_samples, num_samples=200):
    """分析不同checkpoint的logit分布"""
    
    results = {ckpt: {
        'edge_logits': [],
        'non_edge_logits': [],
        'correct_logits': [],
        'other_successor_logits': [],
        'all_logits': [],
        'softmax_effects': []
    } for ckpt in models_dict.keys()}
    
    analyzed = 0
    
    for idx, (prompt, full_path) in enumerate(test_samples[:num_samples]):
        path_nodes = re.findall(r'\d+', full_path)
        if len(path_nodes) < 4:
            continue
            
        for i in range(2, len(path_nodes) - 1):
            current_node = int(path_nodes[i])
            correct_next = int(path_nodes[i + 1])
            
            input_seq = encode(" ".join(path_nodes[:i+1]))
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            
            successors = get_successors(current_node)
            successors = [int(s) for s in successors]
            
            if correct_next not in successors:
                continue
                
            for ckpt, model in models_dict.items():
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                    logits = logits[0, -1, :].cpu().numpy()  # 原始logits
                    
                    # 只考虑节点部分
                    node_logits = logits[NODE_OFFSET:NODE_OFFSET+num_nodes]
                    
                    # 分类logits
                    correct_logit = node_logits[correct_next]
                    results[ckpt]['correct_logits'].append(correct_logit)
                    
                    # 后继节点的logits
                    for s in successors:
                        if s != correct_next:
                            results[ckpt]['other_successor_logits'].append(node_logits[s])
                    
                    # 边（所有后继）的logits
                    for s in successors:
                        results[ckpt]['edge_logits'].append(node_logits[s])
                    
                    # 非边的logits
                    for n in range(num_nodes):
                        if n not in successors and n != current_node:
                            results[ckpt]['non_edge_logits'].append(node_logits[n])
                    
                    # 所有logits（用于计算整体分布）
                    results[ckpt]['all_logits'].extend(node_logits)
                    
                    # 计算softmax效应
                    probs = F.softmax(torch.tensor(node_logits), dim=-1).numpy()
                    max_logit_idx = np.argmax(node_logits)
                    max_prob = probs[max_logit_idx]
                    second_logit_idx = np.argsort(node_logits)[-2]
                    second_prob = probs[second_logit_idx]
                    
                    # softmax放大效应：最大和第二大概率的比率
                    amplification = max_prob / (second_prob + 1e-10)
                    results[ckpt]['softmax_effects'].append(amplification)
            
            analyzed += 1
            if analyzed >= num_samples:
                break
                
        if analyzed >= num_samples:
            break
    
    print(f"Analyzed {analyzed} predictions")
    return results

def plot_logit_distributions(results):
    """绘制logit分布对比图"""
    
    checkpoints = list(results.keys())
    n_ckpts = len(checkpoints)
    
    # 1. 边vs非边的logit分布
    fig, axes = plt.subplots(2, n_ckpts//2 + n_ckpts%2, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, ckpt in enumerate(checkpoints):
        ax = axes[idx]
        
        edge_logits = results[ckpt]['edge_logits']
        non_edge_logits = results[ckpt]['non_edge_logits']
        
        # 绘制直方图
        bins = np.linspace(-2, 2, 50)
        ax.hist(edge_logits, bins=bins, alpha=0.6, label=f'Edges (n={len(edge_logits)})', 
                color='green', density=True)
        ax.hist(non_edge_logits, bins=bins, alpha=0.6, label=f'Non-edges (n={len(non_edge_logits)})', 
                color='red', density=True)
        
        # 添加均值线
        ax.axvline(np.mean(edge_logits), color='darkgreen', linestyle='--', 
                  label=f'Edge mean: {np.mean(edge_logits):.3f}')
        ax.axvline(np.mean(non_edge_logits), color='darkred', linestyle='--', 
                  label=f'Non-edge mean: {np.mean(non_edge_logits):.3f}')
        
        ax.set_title(f'Checkpoint {ckpt//1000}k')
        ax.set_xlabel('Logit Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'logit_distributions_edge_vs_nonedge.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()
    
    # 2. 正确答案vs其他后继的logit分布
    fig, axes = plt.subplots(1, n_ckpts, figsize=(20, 5))
    if n_ckpts == 1:
        axes = [axes]
    
    for idx, ckpt in enumerate(checkpoints):
        ax = axes[idx]
        
        correct_logits = results[ckpt]['correct_logits']
        other_successor_logits = results[ckpt]['other_successor_logits']
        
        # 箱线图
        data_to_plot = [correct_logits, other_successor_logits]
        labels = ['Correct', 'Other Successors']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('orange')
        
        ax.set_title(f'Checkpoint {ckpt//1000}k')
        ax.set_ylabel('Logit Value')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax.text(0.5, 0.95, f'Correct mean: {np.mean(correct_logits):.3f}', 
                transform=ax.transAxes, ha='center', fontsize=9)
        ax.text(0.5, 0.90, f'Others mean: {np.mean(other_successor_logits):.3f}', 
                transform=ax.transAxes, ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'logit_distributions_correct_vs_others.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def analyze_softmax_effect(results):
    """分析softmax的放大效应"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    checkpoints = list(results.keys())
    
    # 1. Softmax放大倍数的分布
    for ckpt in checkpoints:
        effects = results[ckpt]['softmax_effects']
        ax1.hist(np.log10(effects), bins=50, alpha=0.6, label=f'{ckpt//1000}k', density=True)
    
    ax1.set_xlabel('Log10(Max Prob / Second Prob)')
    ax1.set_ylabel('Density')
    ax1.set_title('Softmax Amplification Effect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均放大倍数随checkpoint的变化
    mean_effects = [np.median(results[ckpt]['softmax_effects']) for ckpt in checkpoints]
    ax2.plot(checkpoints, mean_effects, marker='o', markersize=8)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Median Amplification')
    ax2.set_title('Softmax Amplification Over Training')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=40000, color='r', linestyle='--', alpha=0.5, label='Weights turn positive')
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'softmax_amplification_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def compare_weight_matrices(checkpoints_to_compare):
    """直接比较权重矩阵"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, ckpt in enumerate(checkpoints_to_compare[:4]):
        ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            continue
            
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 获取输出权重矩阵
        output_weights = checkpoint['model']['lm_head.weight'].numpy()
        
        # 只看节点部分的权重
        node_weights = output_weights[NODE_OFFSET:NODE_OFFSET+num_nodes, :]
        
        # 计算每个节点的平均输出权重
        mean_weights = np.mean(node_weights, axis=1)
        
        ax = axes[idx//2, idx%2]
        
        # 为每个节点标记是否是常见的目标节点
        # 这需要从图结构中统计
        in_degrees = dict(G.in_degree())
        high_in_degree_nodes = sorted(in_degrees.items(), key=lambda x: int(x[1]), reverse=True)[:10]
        high_in_degree_nodes = [int(n[0]) for n in high_in_degree_nodes]
        
        colors = ['red' if i in high_in_degree_nodes else 'blue' for i in range(num_nodes)]
        
        ax.scatter(range(num_nodes), mean_weights, c=colors, alpha=0.6)
        ax.set_title(f'Checkpoint {ckpt//1000}k - Mean Output Weights')
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Mean Weight')
        ax.grid(True, alpha=0.3)
        
        # 添加统计
        ax.text(0.02, 0.98, f'Min: {mean_weights.min():.4f}', transform=ax.transAxes, 
                va='top', fontsize=9)
        ax.text(0.02, 0.93, f'Max: {mean_weights.max():.4f}', transform=ax.transAxes, 
                va='top', fontsize=9)
        ax.text(0.02, 0.88, f'Range: {mean_weights.max()-mean_weights.min():.4f}', 
                transform=ax.transAxes, va='top', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'weight_matrix_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def analyze_specific_examples_with_logits(models_dict, test_samples, num_examples=3):
    """分析具体例子的logit和概率变化"""
    
    fig, axes = plt.subplots(num_examples, len(models_dict), figsize=(20, 5*num_examples))
    
    for ex_idx in range(num_examples):
        sample = test_samples[ex_idx]
        path_nodes = re.findall(r'\d+', sample[1])
        
        if len(path_nodes) < 5:
            continue
            
        analyze_pos = 3
        current_node = int(path_nodes[analyze_pos])
        correct_next = int(path_nodes[analyze_pos + 1])
        
        input_seq = encode(" ".join(path_nodes[:analyze_pos+1]))
        input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
        
        successors = get_successors(current_node)
        successors = [int(s) for s in successors]
        
        for model_idx, (ckpt, model) in enumerate(models_dict.items()):
            ax = axes[ex_idx, model_idx] if num_examples > 1 else axes[model_idx]
            
            with torch.no_grad():
                logits, _ = model(input_tensor)
                logits = logits[0, -1, NODE_OFFSET:NODE_OFFSET+num_nodes].cpu().numpy()
                probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
            
            # 准备数据
            categories = []
            logit_values = []
            prob_values = []
            colors = []
            
            # 正确答案
            categories.append(f'Correct\n({correct_next})')
            logit_values.append(logits[correct_next])
            prob_values.append(probs[correct_next])
            colors.append('green')
            
            # 其他后继
            for s in successors[:3]:  # 最多显示3个
                if s != correct_next:
                    categories.append(f'Succ\n({s})')
                    logit_values.append(logits[s])
                    prob_values.append(probs[s])
                    colors.append('orange')
            
            # 非后继平均
            non_succ_logits = [logits[i] for i in range(num_nodes) 
                              if i not in successors and i != current_node]
            categories.append('Non-succ\n(avg)')
            logit_values.append(np.mean(non_succ_logits))
            prob_values.append(np.mean([probs[i] for i in range(num_nodes) 
                                       if i not in successors and i != current_node]))
            colors.append('red')
            
            # 绘制
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, logit_values, width, label='Logits', alpha=0.7)
            bars2 = ax.bar(x + width/2, prob_values, width, label='Probs', alpha=0.7)
            
            # 设置颜色
            for bar1, bar2, color in zip(bars1, bars2, colors):
                bar1.set_color(color)
                bar2.set_color(color)
            
            ax.set_xlabel('Category')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_title(f'Example {ex_idx+1}, Ckpt {ckpt//1000}k\n{current_node}→{correct_next}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                       f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
                ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                       f'{height2:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'specific_examples_logit_prob_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def save_detailed_report(results, models_dict):
    """保存详细的数值报告"""
    
    report_path = os.path.join(deep_analysis_dir, 'deep_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEP ANALYSIS REPORT: LOGIT DISTRIBUTIONS AND SOFTMAX EFFECTS\n")
        f.write("="*80 + "\n\n")
        
        for ckpt in results.keys():
            f.write(f"\nCheckpoint {ckpt}:\n")
            f.write("-"*40 + "\n")
            
            # Logit统计
            edge_logits = results[ckpt]['edge_logits']
            non_edge_logits = results[ckpt]['non_edge_logits']
            correct_logits = results[ckpt]['correct_logits']
            other_successor_logits = results[ckpt]['other_successor_logits']
            
            f.write("LOGIT STATISTICS:\n")
            f.write(f"  Edge logits: mean={np.mean(edge_logits):.4f}, std={np.std(edge_logits):.4f}\n")
            f.write(f"  Non-edge logits: mean={np.mean(non_edge_logits):.4f}, std={np.std(non_edge_logits):.4f}\n")
            f.write(f"  Logit gap (edge - non-edge): {np.mean(edge_logits) - np.mean(non_edge_logits):.4f}\n")
            f.write(f"  Correct answer logits: mean={np.mean(correct_logits):.4f}\n")
            f.write(f"  Other successor logits: mean={np.mean(other_successor_logits):.4f}\n")
            
            # Softmax效应
            softmax_effects = results[ckpt]['softmax_effects']
            f.write(f"\nSOFTMAX AMPLIFICATION:\n")
            f.write(f"  Median amplification: {np.median(softmax_effects):.1f}x\n")
            f.write(f"  Mean amplification: {np.mean(softmax_effects):.1f}x\n")
            f.write(f"  Max amplification: {np.max(softmax_effects):.1f}x\n")
            
            # 转换到概率后的差异
            edge_probs = [np.exp(l) / (np.exp(l) + np.exp(np.mean(non_edge_logits))) for l in edge_logits]
            non_edge_probs = [np.exp(l) / (np.exp(l) + np.exp(np.mean(edge_logits))) for l in non_edge_logits]
            
            f.write(f"\nAFTER SOFTMAX:\n")
            f.write(f"  Approx edge prob: {np.mean(edge_probs):.6f}\n")
            f.write(f"  Approx non-edge prob: {np.mean(non_edge_probs):.6f}\n")
            f.write(f"  Probability ratio: {np.mean(edge_probs) / (np.mean(non_edge_probs) + 1e-10):.1f}x\n")
        
        # 关键发现
        f.write("\n\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        
        # 比较40k和100k
        if 40000 in results and 100000 in results:
            logit_gap_40k = np.mean(results[40000]['edge_logits']) - np.mean(results[40000]['non_edge_logits'])
            logit_gap_100k = np.mean(results[100000]['edge_logits']) - np.mean(results[100000]['non_edge_logits'])
            
            f.write(f"\nLogit gap change from 40k to 100k:\n")
            f.write(f"  40k: {logit_gap_40k:.4f}\n")
            f.write(f"  100k: {logit_gap_100k:.4f}\n")
            f.write(f"  Change: {logit_gap_100k - logit_gap_40k:.4f} ({(logit_gap_100k/logit_gap_40k - 1)*100:.1f}%)\n")
            
            f.write(f"\nSoftmax amplification change:\n")
            f.write(f"  40k median: {np.median(results[40000]['softmax_effects']):.1f}x\n")
            f.write(f"  100k median: {np.median(results[100000]['softmax_effects']):.1f}x\n")
    
    print(f"Saved report: {report_path}")

def main():
    # 要分析的checkpoints
    checkpoints_to_analyze = [40000, 50000, 70000, 100000]
    
    # 加载测试样本
    print("Loading test samples...")
    test_samples = load_test_samples()
    print(f"Loaded {len(test_samples)} test samples")
    
    # 加载模型
    print("\nLoading models...")
    models_dict = {}
    for ckpt in checkpoints_to_analyze:
        ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
        model = load_model(ckpt_path)
        if model is not None:
            models_dict[ckpt] = model
    
    # 1. 分析logit分布
    print("\n" + "="*80)
    print("ANALYZING LOGIT DISTRIBUTIONS")
    print("="*80)
    results = analyze_logit_distributions(models_dict, test_samples, num_samples=300)
    
    # 2. 绘制logit分布图
    print("\nPlotting logit distributions...")
    plot_logit_distributions(results)
    
    # 3. 分析softmax效应
    print("\nAnalyzing softmax amplification effects...")
    analyze_softmax_effect(results)
    
    # 4. 比较权重矩阵
    print("\nComparing weight matrices...")
    compare_weight_matrices(checkpoints_to_analyze)
    
    # 5. 分析具体例子
    print("\nAnalyzing specific examples with logits...")
    analyze_specific_examples_with_logits(models_dict, test_samples, num_examples=3)
    
    # 6. 保存详细报告
    print("\nSaving detailed report...")
    save_detailed_report(results, models_dict)
    
    print(f"\nAll analysis completed! Results saved to: {deep_analysis_dir}")
    print("\nPlease check the following files:")
    print("1. logit_distributions_edge_vs_nonedge.png - Shows how edge and non-edge logits are distributed")
    print("2. logit_distributions_correct_vs_others.png - Compares correct answer vs other successors")
    print("3. softmax_amplification_analysis.png - Shows how softmax amplifies differences")
    print("4. weight_matrix_analysis.png - Direct visualization of weight matrices")
    print("5. specific_examples_logit_prob_comparison.png - Concrete examples")
    print("6. deep_analysis_report.txt - Detailed numerical analysis")

if __name__ == "__main__":
    main()