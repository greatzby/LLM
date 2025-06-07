"""
analyze_predictions.py - 分析模型在不同训练阶段的预测分布变化
重点关注Exclusion到Selection的转变
所有图像都会保存到输出目录
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

# 设置matplotlib不显示图形（只保存）
plt.ioff()

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

# 创建图像保存目录
img_dir = os.path.join(out_dir, 'analysis_plots')
os.makedirs(img_dir, exist_ok=True)

# 加载meta信息
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
vocab_size = meta['vocab_size']

# 加载图（注意：这是有向图）
graph_path = os.path.join(data_dir, "path_graph.graphml")
G = nx.read_graphml(graph_path)

# 重要：建立node到token的映射
NODE_OFFSET = 2  # 节点i对应token i+2

def node_to_token(node_id):
    """将节点ID转换为token ID"""
    return node_id + NODE_OFFSET

def token_to_node(token_id):
    """将token ID转换为节点ID"""
    if token_id < NODE_OFFSET:
        return None  # PAD或\n
    return token_id - NODE_OFFSET

def get_successors(node):
    """获取节点的所有后继节点（有向图）"""
    return list(G.successors(str(node)))  # GraphML使用字符串作为节点ID

def load_model(checkpoint_path):
    """加载模型checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # 加载权重
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
    return [stoi[token] for token in ss if token in stoi]

def decode(l):
    """解码token序列为字符串"""
    return " ".join([itos[i] for i in l])

def analyze_prediction_distribution(model, test_samples, num_samples=100):
    """分析模型的预测分布特征（增强版）"""
    results = {
        'correct_token_prob': [],      
        'max_successor_prob': [],       
        'min_successor_prob': [],       
        'successor_prob_std': [],       
        'avg_successor_prob': [],       
        'avg_non_successor_prob': [],   
        'entropy': [],                 
        'correct_successor_rank': [],   
        'successor_vs_non_successor_ratio': [],
        # 新增指标
        'top5_concentration': [],      # top-5概率之和
        'effective_choices': [],       # 概率>0.01的选项数
        'other_successor_probs': [],  # 其他后继节点的概率列表
        'distribution_smoothness': []  # 分布平滑度
    }
    
    analyzed = 0
    with torch.no_grad():
        for idx, (prompt, full_path) in enumerate(test_samples[:num_samples]):
            # 解析路径
            path_nodes = re.findall(r'\d+', full_path)
            if len(path_nodes) < 4:
                continue
                
            # 对路径中的每一步进行分析（除了最后一步）
            for i in range(2, len(path_nodes) - 1):
                current_node = int(path_nodes[i])
                correct_next = int(path_nodes[i + 1])
                
                # 构建输入序列
                input_seq = encode(" ".join(path_nodes[:i+1]))
                input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
                
                # 获取模型预测
                logits, _ = model(input_tensor)
                logits = logits[0, -1, :]  # 取最后一个位置的logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                
                # 获取后继节点（有向图）
                successors = get_successors(current_node)
                successors = [int(s) for s in successors]  # 转换为整数
                
                if correct_next not in successors:
                    continue  # 跳过错误的ground truth
                
                # 转换到token空间
                successor_tokens = [node_to_token(n) for n in successors]
                non_successor_nodes = [i for i in range(num_nodes) 
                                      if i not in successors and i != current_node]
                non_successor_tokens = [node_to_token(n) for n in non_successor_nodes]
                
                # 计算统计量
                successor_probs = probs[successor_tokens]
                non_successor_probs = probs[non_successor_tokens]
                
                correct_token = node_to_token(correct_next)
                results['correct_token_prob'].append(probs[correct_token])
                results['max_successor_prob'].append(successor_probs.max())
                results['min_successor_prob'].append(successor_probs.min())
                results['successor_prob_std'].append(successor_probs.std())
                results['avg_successor_prob'].append(successor_probs.mean())
                results['avg_non_successor_prob'].append(non_successor_probs.mean())
                
                # 计算熵（只考虑有效的节点token）
                node_tokens = list(range(NODE_OFFSET, NODE_OFFSET + num_nodes))
                node_probs = probs[node_tokens]
                node_probs = node_probs / node_probs.sum()  # 重新归一化
                entropy = -(node_probs * np.log(node_probs + 1e-10)).sum()
                results['entropy'].append(entropy)
                
                # 正确答案在后继节点中的排名
                correct_prob = probs[correct_token]
                rank = sum(1 for p in successor_probs if p > correct_prob) + 1
                results['correct_successor_rank'].append(rank)
                
                # 后继vs非后继比率
                ratio = successor_probs.mean() / (non_successor_probs.mean() + 1e-10)
                results['successor_vs_non_successor_ratio'].append(ratio)
                
                # 新增统计
                # Top-5概率集中度
                top5_probs = sorted(node_probs)[-5:]
                results['top5_concentration'].append(sum(top5_probs))
                
                # 有效选择数（概率>0.01）
                effective = sum(1 for p in node_probs if p > 0.01)
                results['effective_choices'].append(effective)
                
                # 其他后继节点的概率
                other_successor_probs = [probs[node_to_token(s)] for s in successors if s != correct_next]
                results['other_successor_probs'].extend(other_successor_probs)
                
                # 分布平滑度（使用基尼系数的反向）
                sorted_probs = sorted(node_probs)
                n = len(sorted_probs)
                gini = sum((2*i - n + 1) * p for i, p in enumerate(sorted_probs)) / (n * sum(sorted_probs))
                results['distribution_smoothness'].append(1 - gini)
                
                analyzed += 1
                if analyzed >= num_samples:
                    break
                    
            if analyzed >= num_samples:
                break
    
    print(f"Analyzed {analyzed} predictions")
    return results

def visualize_probability_distributions(test_samples, checkpoints_to_compare):
    """可视化同一个预测在不同checkpoint的概率分布"""
    
    # 选择一个代表性的测试样本
    selected_sample = None
    for sample in test_samples[:50]:
        path_nodes = re.findall(r'\d+', sample[1])
        if len(path_nodes) >= 6:  # 确保路径足够长
            selected_sample = sample
            break
    
    if not selected_sample:
        print("No suitable sample found")
        return
    
    path_nodes = re.findall(r'\d+', selected_sample[1])
    analyze_pos = 4  # 分析第4步
    current_node = int(path_nodes[analyze_pos])
    correct_next = int(path_nodes[analyze_pos + 1])
    
    successors = get_successors(current_node)
    successors = [int(s) for s in successors]
    
    print(f"Analyzing: {current_node} -> {correct_next}")
    print(f"Successors: {successors}")
    
    # 构建输入
    input_seq = encode(" ".join(path_nodes[:analyze_pos+1]))
    input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, ckpt in enumerate(checkpoints_to_compare[:8]):
        ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
        model = load_model(ckpt_path)
        if model is None:
            continue
        
        with torch.no_grad():
            logits, _ = model(input_tensor)
            probs = F.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
        
        # 获取节点概率
        node_probs = probs[NODE_OFFSET:NODE_OFFSET+num_nodes]
        
        # 准备数据
        correct_prob = node_probs[correct_next]
        other_successor_probs = [node_probs[s] for s in successors if s != correct_next]
        non_successor_probs = [node_probs[i] for i in range(num_nodes) 
                              if i not in successors and i != current_node]
        
        # 绘制
        ax = axes[idx]
        
        # 绘制所有后继节点的概率
        x_pos = 0
        colors = []
        heights = []
        labels = []
        
        # 正确答案
        heights.append(correct_prob)
        colors.append('green')
        labels.append(f'Correct ({correct_next})')
        
        # 其他后继
        for i, (s, p) in enumerate(zip([s for s in successors if s != correct_next], other_successor_probs)):
            heights.append(p)
            colors.append('orange')
            labels.append(f'Succ {s}')
        
        # 非后继平均
        heights.append(np.mean(non_successor_probs))
        colors.append('red')
        labels.append('Non-succ\n(avg)')
        
        bars = ax.bar(range(len(heights)), heights, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Checkpoint {ckpt//1000}k')
        
        # 添加统计信息
        entropy = -sum(p * np.log(p + 1e-10) for p in node_probs if p > 0)
        effective_choices = sum(1 for p in node_probs if p > 0.01)
        ax.text(0.5, 0.95, f'Entropy: {entropy:.2f}', transform=ax.transAxes, ha='center')
        ax.text(0.5, 0.90, f'Effective choices: {effective_choices}', transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(img_dir, 'distribution_evolution.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def plot_distribution_metrics(metrics):
    """绘制分布特征的演变图"""
    
    fig = plt.figure(figsize=(16, 12))
    checkpoints = metrics['checkpoints']
    
    # 1. 概率集中度
    plt.subplot(3, 2, 1)
    plt.plot(checkpoints, metrics['top1_concentration'], label='Top-1 (Correct)', marker='o', markersize=8)
    plt.plot(checkpoints, metrics['top5_concentration'], label='Top-5 Sum', marker='s', markersize=8)
    plt.axvline(x=40000, color='r', linestyle='--', alpha=0.5, label='Non-edge weights turn positive')
    plt.xlabel('Training Steps')
    plt.ylabel('Probability')
    plt.title('Probability Concentration Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 有效选择数
    plt.subplot(3, 2, 2)
    plt.plot(checkpoints, metrics['effective_choices'], marker='o', markersize=8, color='purple')
    plt.axvline(x=40000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Number of Choices')
    plt.title('Effective Choices (p > 0.01)')
    plt.grid(True, alpha=0.3)
    
    # 3. 探索性分数
    plt.subplot(3, 2, 3)
    plt.plot(checkpoints, metrics['exploration_score'], marker='o', markersize=8, color='green')
    plt.axvline(x=40000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Exploration Score')
    plt.title('Exploration Capability (Entropy/Max Entropy)')
    plt.grid(True, alpha=0.3)
    
    # 4. 分布平滑度
    plt.subplot(3, 2, 4)
    plt.plot(checkpoints, metrics['distribution_smoothness'], marker='o', markersize=8, color='orange')
    plt.axvline(x=40000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Smoothness Score')
    plt.title('Distribution Smoothness (1 - Gini)')
    plt.grid(True, alpha=0.3)
    
    # 5. 其他后继节点的概率
    plt.subplot(3, 2, 5)
    plt.plot(checkpoints, metrics['avg_other_successor_prob'], marker='o', markersize=8, color='brown')
    plt.axvline(x=40000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Average Probability')
    plt.title('Average Probability of Other Successors')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 6. Exclusion vs Selection对比
    plt.subplot(3, 2, 6)
    exclusion_phase = [i for i, c in enumerate(checkpoints) if c <= 40000]
    selection_phase = [i for i, c in enumerate(checkpoints) if c > 40000]
    
    if exclusion_phase and selection_phase:
        exclusion_entropy = np.mean([metrics['exploration_score'][i] for i in exclusion_phase])
        selection_entropy = np.mean([metrics['exploration_score'][i] for i in selection_phase])
        
        exclusion_choices = np.mean([metrics['effective_choices'][i] for i in exclusion_phase])
        selection_choices = np.mean([metrics['effective_choices'][i] for i in selection_phase])
        
        x = ['Exclusion\n(≤40k)', 'Selection\n(>40k)']
        y1 = [exclusion_entropy, selection_entropy]
        y2 = [exclusion_choices/10, selection_choices/10]  # 缩放以便在同一图中显示
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        plt.bar(x_pos - width/2, y1, width, label='Exploration Score', color='green', alpha=0.7)
        plt.bar(x_pos + width/2, y2, width, label='Effective Choices/10', color='purple', alpha=0.7)
        plt.xticks(x_pos, x)
        plt.ylabel('Score')
        plt.title('Exclusion vs Selection Phase Comparison')
        plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(img_dir, 'distribution_metrics_evolution.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def plot_key_metrics_summary(all_results, checkpoints):
    """绘制关键指标的总结图"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 准备数据
    ckpt_list = sorted([c for c in checkpoints if c in all_results])
    
    # 1. 正确答案的概率
    plt.subplot(2, 3, 1)
    values = [np.mean(all_results[c]['correct_token_prob']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5, label='TF accuracy drop')
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5, label='Weights turn positive')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title('Average Probability of Correct Token')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 后继节点间概率的标准差
    plt.subplot(2, 3, 2)
    values = [np.mean(all_results[c]['successor_prob_std']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Std Dev')
    plt.title('Std of Successor Probabilities\n(Lower = more uniform)')
    plt.grid(True, alpha=0.3)
    
    # 3. 后继vs非后继的概率比
    plt.subplot(2, 3, 3)
    values = [np.mean(all_results[c]['successor_vs_non_successor_ratio']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Successor/Non-successor Probability Ratio')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 4. 预测分布的熵
    plt.subplot(2, 3, 4)
    values = [np.mean(all_results[c]['entropy']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Prediction Entropy')
    plt.grid(True, alpha=0.3)
    
    # 5. 正确答案的平均排名
    plt.subplot(2, 3, 5)
    values = [np.mean(all_results[c]['correct_successor_rank']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Average Rank')
    plt.title('Average Rank of Correct Token\n(among successors)')
    plt.grid(True, alpha=0.3)
    
    # 6. 有效选择数
    plt.subplot(2, 3, 6)
    values = [np.mean(all_results[c]['effective_choices']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Number of Choices')
    plt.title('Effective Choices (p > 0.01)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(img_dir, 'key_metrics_summary.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def analyze_distribution_characteristics(all_results, checkpoints):
    """分析分布特征的演变"""
    
    metrics = {
        'checkpoints': [],
        'top1_concentration': [],
        'top5_concentration': [],
        'effective_choices': [],
        'distribution_smoothness': [],
        'exploration_score': [],
        'avg_other_successor_prob': []
    }
    
    for ckpt in checkpoints:
        if ckpt not in all_results:
            continue
            
        results = all_results[ckpt]
        metrics['checkpoints'].append(ckpt)
        
        # Top-1集中度（正确答案的概率）
        metrics['top1_concentration'].append(np.mean(results['correct_token_prob']))
        
        # Top-5集中度
        metrics['top5_concentration'].append(np.mean(results['top5_concentration']))
        
        # 有效选择数
        metrics['effective_choices'].append(np.mean(results['effective_choices']))
        
        # 分布平滑度
        metrics['distribution_smoothness'].append(np.mean(results['distribution_smoothness']))
        
        # 探索性分数（熵/最大可能熵）
        avg_entropy = np.mean(results['entropy'])
        max_possible_entropy = np.log(num_nodes)
        metrics['exploration_score'].append(avg_entropy / max_possible_entropy)
        
        # 其他后继节点的平均概率
        if results['other_successor_probs']:
            metrics['avg_other_successor_prob'].append(np.mean(results['other_successor_probs']))
        else:
            metrics['avg_other_successor_prob'].append(0)
    
    return metrics

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

def save_summary_report(all_results, metrics, checkpoints):
    """保存文本格式的分析报告"""
    
    report_path = os.path.join(img_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PREDICTION DISTRIBUTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 总体趋势
        f.write("1. OVERALL TRENDS\n")
        f.write("-"*40 + "\n")
        
        for ckpt in checkpoints:
            if ckpt in all_results:
                f.write(f"\nCheckpoint {ckpt}:\n")
                f.write(f"  Correct token prob: {np.mean(all_results[ckpt]['correct_token_prob']):.4f}\n")
                f.write(f"  Effective choices: {np.mean(all_results[ckpt]['effective_choices']):.2f}\n")
                f.write(f"  Entropy: {np.mean(all_results[ckpt]['entropy']):.4f}\n")
                f.write(f"  Successor/Non-successor ratio: {np.mean(all_results[ckpt]['successor_vs_non_successor_ratio']):.2f}\n")
        
        # Exclusion vs Selection对比
        f.write("\n\n2. EXCLUSION VS SELECTION PHASE COMPARISON\n")
        f.write("-"*40 + "\n")
        
        exclusion_checkpoints = [c for c in checkpoints if c <= 40000 and c in all_results]
        selection_checkpoints = [c for c in checkpoints if c > 40000 and c in all_results]
        
        if exclusion_checkpoints:
            f.write("\nExclusion Phase (≤40k):\n")
            for metric in ['correct_token_prob', 'effective_choices', 'entropy']:
                avg_value = np.mean([np.mean(all_results[c][metric]) for c in exclusion_checkpoints])
                f.write(f"  Avg {metric}: {avg_value:.4f}\n")
        
        if selection_checkpoints:
            f.write("\nSelection Phase (>40k):\n")
            for metric in ['correct_token_prob', 'effective_choices', 'entropy']:
                avg_value = np.mean([np.mean(all_results[c][metric]) for c in selection_checkpoints])
                f.write(f"  Avg {metric}: {avg_value:.4f}\n")
        
        # 关键发现
        f.write("\n\n3. KEY FINDINGS\n")
        f.write("-"*40 + "\n")
        
        # 检查是否有明显的转变
        if exclusion_checkpoints and selection_checkpoints:
            exclusion_choices = np.mean([np.mean(all_results[c]['effective_choices']) for c in exclusion_checkpoints])
            selection_choices = np.mean([np.mean(all_results[c]['effective_choices']) for c in selection_checkpoints])
            
            if selection_choices > exclusion_choices * 1.5:
                f.write("\n✓ Evidence of transition from Exclusion to Selection:\n")
                f.write(f"  Effective choices increased from {exclusion_choices:.2f} to {selection_choices:.2f}\n")
            else:
                f.write("\n✗ No clear evidence of Exclusion to Selection transition:\n")
                f.write(f"  Effective choices remained similar: {exclusion_choices:.2f} vs {selection_choices:.2f}\n")
    
    print(f"Saved report: {report_path}")

def main():
    # 扩展要分析的checkpoint范围
    checkpoints = [7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 
                   50000, 60000, 70000, 80000, 90000, 100000]
    
    # 加载测试样本
    print("Loading test samples...")
    test_samples = load_test_samples()
    print(f"Loaded {len(test_samples)} test samples")
    
    # 1. 分析每个checkpoint的预测分布
    all_results = {}
    for ckpt in checkpoints:
        ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt} not found, skipping...")
            continue
            
        model = load_model(ckpt_path)
        if model is None:
            continue
            
        print(f"\nAnalyzing checkpoint {ckpt}...")
        results = analyze_prediction_distribution(model, test_samples, num_samples=200)
        all_results[ckpt] = results
        
        # 打印关键统计
        print(f"Checkpoint {ckpt} summary:")
        print(f"  Avg correct token prob: {np.mean(results['correct_token_prob']):.4f}")
        print(f"  Avg successor prob std: {np.mean(results['successor_prob_std']):.4f}")
        print(f"  Avg successor/non-successor ratio: {np.mean(results['successor_vs_non_successor_ratio']):.2f}")
        print(f"  Avg entropy: {np.mean(results['entropy']):.4f}")
        print(f"  Avg correct token rank: {np.mean(results['correct_successor_rank']):.2f}")
        print(f"  Avg effective choices: {np.mean(results['effective_choices']):.2f}")
        print(f"  Top-5 concentration: {np.mean(results['top5_concentration']):.4f}")
    
    # 2. 绘制并保存关键指标总结图
    print("\n" + "="*80)
    print("PLOTTING KEY METRICS SUMMARY")
    print("="*80)
    
    plot_key_metrics_summary(all_results, checkpoints)
    
    # 3. 分析分布特征
    print("\n" + "="*80)
    print("ANALYZING DISTRIBUTION CHARACTERISTICS")
    print("="*80)
    
    metrics = analyze_distribution_characteristics(all_results, checkpoints)
    plot_distribution_metrics(metrics)
    
    # 4. 可视化具体的概率分布演变
    print("\n" + "="*80)
    print("VISUALIZING PROBABILITY DISTRIBUTIONS")
    print("="*80)
    
    visualize_probability_distributions(test_samples, 
                                      [10000, 20000, 30000, 40000, 50000, 70000, 90000, 100000])
    
    # 5. 保存分析报告
    print("\n" + "="*80)
    print("SAVING ANALYSIS REPORT")
    print("="*80)
    
    save_summary_report(all_results, metrics, checkpoints)
    
    print(f"\nAll plots and report saved to: {img_dir}")

if __name__ == "__main__":
    main()