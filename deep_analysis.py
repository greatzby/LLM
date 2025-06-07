"""
deep_analysis.py - 深入分析logit分布和softmax效应
修正版：仔细处理所有索引问题
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
NODE_OFFSET = 2  # 节点i对应token i+2

print(f"Vocab size: {vocab_size}")
print(f"Number of nodes: {num_nodes}")
print(f"Node tokens range: {NODE_OFFSET} to {NODE_OFFSET + num_nodes - 1}")

def node_to_token(node_id):
    """将节点ID转换为token ID"""
    if not (0 <= node_id < num_nodes):
        raise ValueError(f"Invalid node ID: {node_id}")
    return node_id + NODE_OFFSET

def token_to_node(token_id):
    """将token ID转换为节点ID"""
    if token_id < NODE_OFFSET or token_id >= NODE_OFFSET + num_nodes:
        return None
    return token_id - NODE_OFFSET

def get_successors(node):
    """获取节点的所有后继节点（有向图）"""
    # GraphML使用字符串作为节点ID
    successors = list(G.successors(str(node)))
    # 转换回整数
    return [int(s) for s in successors]

def load_model(checkpoint_path):
    """加载模型checkpoint"""
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
    """编码字符串为token序列"""
    ss = s.split(" ")
    encoded = []
    for token in ss:
        if token in stoi:
            encoded.append(stoi[token])
        else:
            print(f"Warning: token '{token}' not in vocabulary")
    return encoded

def load_test_samples():
    """加载测试样本"""
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
        'all_node_logits': [],
        'softmax_effects': [],
        'raw_logits_examples': []  # 保存一些原始例子用于验证
    } for ckpt in models_dict.keys()}
    
    analyzed = 0
    
    for idx, (prompt, full_path) in enumerate(test_samples):
        if analyzed >= num_samples:
            break
            
        path_nodes = re.findall(r'\d+', full_path)
        if len(path_nodes) < 4:
            continue
            
        # 对路径中的每一步进行分析（除了最后一步）
        for i in range(2, min(len(path_nodes) - 1, 8)):  # 限制分析深度避免过长序列
            current_node = int(path_nodes[i])
            correct_next = int(path_nodes[i + 1])
            
            # 验证节点ID有效性
            if not (0 <= current_node < num_nodes and 0 <= correct_next < num_nodes):
                print(f"Warning: Invalid node IDs: {current_node}, {correct_next}")
                continue
            
            # 构建输入序列
            input_seq = encode(" ".join(path_nodes[:i+1]))
            if not input_seq:
                continue
                
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            
            # 获取后继节点
            try:
                successors = get_successors(current_node)
            except:
                print(f"Warning: Failed to get successors for node {current_node}")
                continue
                
            if correct_next not in successors:
                continue
                
            # 对每个模型进行分析
            for ckpt, model in models_dict.items():
                with torch.no_grad():
                    try:
                        logits, _ = model(input_tensor)
                        all_logits = logits[0, -1, :].cpu().numpy()  # 完整的logits向量
                        
                        # 验证维度
                        if len(all_logits) != vocab_size:
                            print(f"Warning: Unexpected logits size: {len(all_logits)}")
                            continue
                        
                        # 提取节点部分的logits
                        node_logits = all_logits[NODE_OFFSET:NODE_OFFSET+num_nodes]
                        
                        # 保存一个例子用于验证
                        if len(results[ckpt]['raw_logits_examples']) < 5:
                            results[ckpt]['raw_logits_examples'].append({
                                'current': current_node,
                                'correct': correct_next,
                                'successors': successors,
                                'logits': node_logits.copy(),
                                'all_logits_shape': all_logits.shape
                            })
                        
                        # 正确答案的logit
                        correct_logit = node_logits[correct_next]
                        results[ckpt]['correct_logits'].append(correct_logit)
                        
                        # 后继节点的logits
                        for s in successors:
                            if 0 <= s < num_nodes:
                                results[ckpt]['edge_logits'].append(node_logits[s])
                                if s != correct_next:
                                    results[ckpt]['other_successor_logits'].append(node_logits[s])
                        
                        # 非后继节点的logits
                        for n in range(num_nodes):
                            if n not in successors and n != current_node:
                                results[ckpt]['non_edge_logits'].append(node_logits[n])
                        
                        # 所有节点logits（用于整体分布）
                        results[ckpt]['all_node_logits'].extend(node_logits)
                        
                        # 计算softmax效应
                        probs = F.softmax(torch.tensor(node_logits), dim=-1).numpy()
                        
                        # 找出概率最大的两个节点
                        sorted_indices = np.argsort(probs)[::-1]
                        max_prob = probs[sorted_indices[0]]
                        second_prob = probs[sorted_indices[1]]
                        
                        # softmax放大效应
                        amplification = max_prob / (second_prob + 1e-10)
                        results[ckpt]['softmax_effects'].append(amplification)
                        
                    except Exception as e:
                        print(f"Error processing checkpoint {ckpt}: {e}")
                        continue
            
            analyzed += 1
            if analyzed >= num_samples:
                break
    
    print(f"Successfully analyzed {analyzed} predictions")
    
    # 打印一些验证信息
    for ckpt in results.keys():
        if results[ckpt]['raw_logits_examples']:
            example = results[ckpt]['raw_logits_examples'][0]
            print(f"\nCheckpoint {ckpt} example:")
            print(f"  Current node: {example['current']}")
            print(f"  Correct next: {example['correct']}")
            print(f"  Successors: {example['successors']}")
            print(f"  Correct logit: {example['logits'][example['correct']]:.4f}")
            print(f"  All logits shape: {example['all_logits_shape']}")
    
    return results

def plot_logit_distributions(results):
    """绘制logit分布对比图"""
    
    checkpoints = sorted(list(results.keys()))
    n_ckpts = len(checkpoints)
    
    # 1. 边vs非边的logit分布
    fig, axes = plt.subplots(2, (n_ckpts + 1) // 2, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, ckpt in enumerate(checkpoints):
        ax = axes[idx]
        
        edge_logits = results[ckpt]['edge_logits']
        non_edge_logits = results[ckpt]['non_edge_logits']
        
        if not edge_logits or not non_edge_logits:
            print(f"Warning: No data for checkpoint {ckpt}")
            continue
        
        # 确定合理的bin范围
        all_values = edge_logits + non_edge_logits
        min_val, max_val = np.percentile(all_values, [1, 99])
        bins = np.linspace(min_val, max_val, 50)
        
        # 绘制直方图
        ax.hist(edge_logits, bins=bins, alpha=0.6, label=f'Edges (n={len(edge_logits)})', 
                color='green', density=True)
        ax.hist(non_edge_logits, bins=bins, alpha=0.6, label=f'Non-edges (n={len(non_edge_logits)})', 
                color='red', density=True)
        
        # 添加均值线
        edge_mean = np.mean(edge_logits)
        non_edge_mean = np.mean(non_edge_logits)
        ax.axvline(edge_mean, color='darkgreen', linestyle='--', 
                  label=f'Edge mean: {edge_mean:.3f}')
        ax.axvline(non_edge_mean, color='darkred', linestyle='--', 
                  label=f'Non-edge mean: {non_edge_mean:.3f}')
        
        ax.set_title(f'Checkpoint {ckpt//1000}k')
        ax.set_xlabel('Logit Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 添加差值信息
        ax.text(0.02, 0.98, f'Gap: {edge_mean - non_edge_mean:.3f}', 
                transform=ax.transAxes, va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(len(checkpoints), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'logit_distributions_edge_vs_nonedge.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()
    
    # 2. 正确答案vs其他后继的logit分布（箱线图）
    fig, axes = plt.subplots(1, n_ckpts, figsize=(5*n_ckpts, 5))
    if n_ckpts == 1:
        axes = [axes]
    
    for idx, ckpt in enumerate(checkpoints):
        ax = axes[idx]
        
        correct_logits = results[ckpt]['correct_logits']
        other_successor_logits = results[ckpt]['other_successor_logits']
        
        if not correct_logits or not other_successor_logits:
            print(f"Warning: Insufficient data for checkpoint {ckpt}")
            continue
        
        # 准备数据
        data_to_plot = [correct_logits, other_successor_logits]
        labels = ['Correct', 'Other Successors']
        
        # 绘制箱线图
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
        ax.text(0.5, 0.85, f'Difference: {np.mean(correct_logits) - np.mean(other_successor_logits):.3f}', 
                transform=ax.transAxes, ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'logit_distributions_correct_vs_others.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def analyze_softmax_effect(results):
    """分析softmax的放大效应"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    checkpoints = sorted(list(results.keys()))
    
    # 1. Softmax放大倍数的分布
    for ckpt in checkpoints:
        effects = results[ckpt]['softmax_effects']
        if not effects:
            continue
            
        # 使用log scale，过滤掉极端值
        log_effects = np.log10(np.clip(effects, 1, 1e6))
        ax1.hist(log_effects, bins=50, alpha=0.6, label=f'{ckpt//1000}k', density=True)
    
    ax1.set_xlabel('Log10(Max Prob / Second Prob)')
    ax1.set_ylabel('Density')
    ax1.set_title('Softmax Amplification Effect Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均放大倍数随checkpoint的变化
    mean_effects = []
    median_effects = []
    for ckpt in checkpoints:
        if results[ckpt]['softmax_effects']:
            effects = np.clip(results[ckpt]['softmax_effects'], 1, 1e6)
            mean_effects.append(np.mean(effects))
            median_effects.append(np.median(effects))
        else:
            mean_effects.append(np.nan)
            median_effects.append(np.nan)
    
    ax2.plot(checkpoints, median_effects, marker='o', markersize=8, label='Median', linewidth=2)
    ax2.plot(checkpoints, mean_effects, marker='s', markersize=8, label='Mean', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Amplification Factor')
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
    axes = axes.flatten()
    
    for idx, ckpt in enumerate(checkpoints_to_compare[:4]):
        ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            continue
            
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 获取输出权重矩阵
        if 'lm_head.weight' in checkpoint['model']:
            output_weights = checkpoint['model']['lm_head.weight'].numpy()
        else:
            print(f"Warning: lm_head.weight not found in checkpoint {ckpt}")
            continue
        
        # 验证维度
        print(f"Checkpoint {ckpt} weight matrix shape: {output_weights.shape}")
        
        # 只看节点部分的权重
        if output_weights.shape[0] < NODE_OFFSET + num_nodes:
            print(f"Warning: Weight matrix too small for checkpoint {ckpt}")
            continue
            
        node_weights = output_weights[NODE_OFFSET:NODE_OFFSET+num_nodes, :]
        
        # 计算每个节点的平均输出权重
        mean_weights = np.mean(node_weights, axis=1)
        
        ax = axes[idx]
        
        # 获取入度信息
        in_degrees = dict(G.in_degree())
        high_in_degree_nodes = sorted([(int(k), v) for k, v in in_degrees.items()], 
                                    key=lambda x: x[1], reverse=True)[:10]
        high_in_degree_node_ids = [n[0] for n in high_in_degree_nodes]
        
        # 设置颜色
        colors = ['red' if i in high_in_degree_node_ids else 'blue' for i in range(num_nodes)]
        
        # 绘制散点图
        scatter = ax.scatter(range(num_nodes), mean_weights, c=colors, alpha=0.6)
        ax.set_title(f'Checkpoint {ckpt//1000}k - Mean Output Weights')
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Mean Weight')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax.text(0.02, 0.98, f'Min: {mean_weights.min():.4f}', transform=ax.transAxes, 
                va='top', fontsize=9)
        ax.text(0.02, 0.93, f'Max: {mean_weights.max():.4f}', transform=ax.transAxes, 
                va='top', fontsize=9)
        ax.text(0.02, 0.88, f'Range: {mean_weights.max()-mean_weights.min():.4f}', 
                transform=ax.transAxes, va='top', fontsize=9)
        ax.text(0.02, 0.83, f'Std: {mean_weights.std():.4f}', 
                transform=ax.transAxes, va='top', fontsize=9)
        
        # 标记一些高入度节点
        for node_id in high_in_degree_node_ids[:3]:
            ax.annotate(f'{node_id}', (node_id, mean_weights[node_id]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 添加图例
    axes[0].text(0.02, 0.02, 'Red: High in-degree nodes\nBlue: Other nodes', 
                transform=axes[0].transAxes, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(deep_analysis_dir, 'weight_matrix_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def analyze_specific_examples_with_logits(models_dict, test_samples, num_examples=3):
    """分析具体例子的logit和概率变化"""
    
    # 确保有足够的样本
    valid_samples = []
    for sample in test_samples[:50]:
        path_nodes = re.findall(r'\d+', sample[1])
        if len(path_nodes) >= 5:
            valid_samples.append(sample)
        if len(valid_samples) >= num_examples:
            break
    
    if not valid_samples:
        print("No valid samples found for specific example analysis")
        return
    
    fig, axes = plt.subplots(len(valid_samples), len(models_dict), 
                            figsize=(5*len(models_dict), 5*len(valid_samples)))
    if len(valid_samples) == 1:
        axes = axes.reshape(1, -1)
    if len(models_dict) == 1:
        axes = axes.reshape(-1, 1)
    
    for ex_idx, sample in enumerate(valid_samples):
        path_nodes = re.findall(r'\d+', sample[1])
        
        analyze_pos = min(3, len(path_nodes) - 2)
        current_node = int(path_nodes[analyze_pos])
        correct_next = int(path_nodes[analyze_pos + 1])
        
        # 验证节点有效性
        if not (0 <= current_node < num_nodes and 0 <= correct_next < num_nodes):
            continue
        
        input_seq = encode(" ".join(path_nodes[:analyze_pos+1]))
        if not input_seq:
            continue
            
        input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
        
        try:
            successors = get_successors(current_node)
        except:
            continue
        
        for model_idx, (ckpt, model) in enumerate(models_dict.items()):
            ax = axes[ex_idx, model_idx]
            
            with torch.no_grad():
                try:
                    logits, _ = model(input_tensor)
                    all_logits = logits[0, -1, :].cpu().numpy()
                    node_logits = all_logits[NODE_OFFSET:NODE_OFFSET+num_nodes]
                    probs = F.softmax(torch.tensor(node_logits), dim=-1).numpy()
                except Exception as e:
                    print(f"Error in example analysis: {e}")
                    continue
            
            # 准备数据
            categories = []
            logit_values = []
            prob_values = []
            colors = []
            
            # 正确答案
            categories.append(f'Correct\n({correct_next})')
            logit_values.append(node_logits[correct_next])
            prob_values.append(probs[correct_next])
            colors.append('green')
            
            # 其他后继（最多3个）
            other_successors = [s for s in successors if s != correct_next][:3]
            for s in other_successors:
                if 0 <= s < num_nodes:
                    categories.append(f'Succ\n({s})')
                    logit_values.append(node_logits[s])
                    prob_values.append(probs[s])
                    colors.append('orange')
            
            # 非后继平均
            non_succ_indices = [i for i in range(num_nodes) 
                               if i not in successors and i != current_node]
            if non_succ_indices:
                non_succ_logits = [node_logits[i] for i in non_succ_indices]
                non_succ_probs = [probs[i] for i in non_succ_indices]
                categories.append('Non-succ\n(avg)')
                logit_values.append(np.mean(non_succ_logits))
                prob_values.append(np.mean(non_succ_probs))
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
                       f'{height1:.2f}', ha='center', va='bottom', fontsize=8)
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
        
        for ckpt in sorted(results.keys()):
            f.write(f"\nCheckpoint {ckpt}:\n")
            f.write("-"*40 + "\n")
            
            # 检查数据完整性
            if not results[ckpt]['edge_logits']:
                f.write("  No data available for this checkpoint\n")
                continue
            
            # Logit统计
            edge_logits = results[ckpt]['edge_logits']
            non_edge_logits = results[ckpt]['non_edge_logits']
            correct_logits = results[ckpt]['correct_logits']
            other_successor_logits = results[ckpt]['other_successor_logits']
            
            f.write("LOGIT STATISTICS:\n")
            f.write(f"  Number of samples: {len(correct_logits)}\n")
            f.write(f"  Edge logits: mean={np.mean(edge_logits):.4f}, std={np.std(edge_logits):.4f}, "
                   f"min={np.min(edge_logits):.4f}, max={np.max(edge_logits):.4f}\n")
            f.write(f"  Non-edge logits: mean={np.mean(non_edge_logits):.4f}, std={np.std(non_edge_logits):.4f}, "
                   f"min={np.min(non_edge_logits):.4f}, max={np.max(non_edge_logits):.4f}\n")
            f.write(f"  Logit gap (edge - non-edge): {np.mean(edge_logits) - np.mean(non_edge_logits):.4f}\n")
            f.write(f"  Correct answer logits: mean={np.mean(correct_logits):.4f}, std={np.std(correct_logits):.4f}\n")
            if other_successor_logits:
                f.write(f"  Other successor logits: mean={np.mean(other_successor_logits):.4f}, "
                       f"std={np.std(other_successor_logits):.4f}\n")
            
            # Softmax效应
            if results[ckpt]['softmax_effects']:
                softmax_effects = np.clip(results[ckpt]['softmax_effects'], 1, 1e6)
                f.write(f"\nSOFTMAX AMPLIFICATION:\n")
                f.write(f"  Median amplification: {np.median(softmax_effects):.1f}x\n")
                f.write(f"  Mean amplification: {np.mean(softmax_effects):.1f}x\n")
                f.write(f"  90th percentile: {np.percentile(softmax_effects, 90):.1f}x\n")
                f.write(f"  99th percentile: {np.percentile(softmax_effects, 99):.1f}x\n")
            
            # 简化的概率估计
            edge_mean = np.mean(edge_logits)
            non_edge_mean = np.mean(non_edge_logits)
            gap = edge_mean - non_edge_mean
            
            f.write(f"\nSIMPLIFIED PROBABILITY ESTIMATION:\n")
            f.write(f"  If we had only 2 choices with gap={gap:.4f}:\n")
            f.write(f"    Edge probability ≈ {1/(1+np.exp(-gap)):.6f}\n")
            f.write(f"    Non-edge probability ≈ {1/(1+np.exp(gap)):.6f}\n")
            f.write(f"    Ratio ≈ {np.exp(gap):.1f}x\n")
        
        # 关键发现
        f.write("\n\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        
        # 比较第一个和最后一个checkpoint
        checkpoints = sorted([c for c in results.keys() if results[c]['edge_logits']])
        if len(checkpoints) >= 2:
            first_ckpt = checkpoints[0]
            last_ckpt = checkpoints[-1]
            
            logit_gap_first = np.mean(results[first_ckpt]['edge_logits']) - np.mean(results[first_ckpt]['non_edge_logits'])
            logit_gap_last = np.mean(results[last_ckpt]['edge_logits']) - np.mean(results[last_ckpt]['non_edge_logits'])
            
            f.write(f"\nLogit gap change from {first_ckpt} to {last_ckpt}:\n")
            f.write(f"  {first_ckpt}: {logit_gap_first:.4f}\n")
            f.write(f"  {last_ckpt}: {logit_gap_last:.4f}\n")
            f.write(f"  Change: {logit_gap_last - logit_gap_first:.4f} "
                   f"({(logit_gap_last/logit_gap_first - 1)*100:+.1f}%)\n")
            
            if results[first_ckpt]['softmax_effects'] and results[last_ckpt]['softmax_effects']:
                f.write(f"\nSoftmax amplification change:\n")
                f.write(f"  {first_ckpt} median: {np.median(results[first_ckpt]['softmax_effects']):.1f}x\n")
                f.write(f"  {last_ckpt} median: {np.median(results[last_ckpt]['softmax_effects']):.1f}x\n")
            
            # 分析权重变化的影响
            f.write(f"\nIMPACT OF WEIGHT CHANGES:\n")
            f.write(f"  Despite non-edge weights becoming positive after 40k,\n")
            f.write(f"  the logit gap decreased by only {abs((logit_gap_last/logit_gap_first - 1)*100):.1f}%\n")
            f.write(f"  This explains why the probability distribution remains highly concentrated.\n")
    
    print(f"Saved report: {report_path}")

def main():
    # 要分析的checkpoints
    checkpoints_to_analyze = [1000,7000,8000,10000,20000,30000,40000, 50000,60000, 70000,80000,90000, 100000]
    
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
    
    if not models_dict:
        print("No models loaded successfully!")
        return
    
    # 1. 分析logit分布
    print("\n" + "="*80)
    print("ANALYZING LOGIT DISTRIBUTIONS")
    print("="*80)
    results = analyze_logit_distributions(models_dict, test_samples, num_samples=300)
    
    # 验证结果
    for ckpt in results.keys():
        print(f"\nCheckpoint {ckpt} data sizes:")
        print(f"  Edge logits: {len(results[ckpt]['edge_logits'])}")
        print(f"  Non-edge logits: {len(results[ckpt]['non_edge_logits'])}")
        print(f"  Correct logits: {len(results[ckpt]['correct_logits'])}")
    
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
    print("1. logit_distributions_edge_vs_nonedge.png - Edge vs non-edge logit distributions")
    print("2. logit_distributions_correct_vs_others.png - Correct answer vs other successors")
    print("3. softmax_amplification_analysis.png - Softmax amplification effects")
    print("4. weight_matrix_analysis.png - Weight matrix visualization")
    print("5. specific_examples_logit_prob_comparison.png - Concrete examples")
    print("6. deep_analysis_report.txt - Detailed numerical analysis")

if __name__ == "__main__":
    main()