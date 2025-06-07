"""
analyze_predictions.py - 分析模型在不同训练阶段的预测分布变化
重点关注Exclusion到Selection的转变
所有图像都会保存到输出目录
修改版：使用验证集(val.bin)而非测试集，与TF accuracy保持一致
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
val_batch_size = 64  # 与train.py保持一致

# 路径设置
data_dir = f'data/{dataset}/{num_nodes}'
out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'

# 创建图像保存目录
img_dir = os.path.join(out_dir, 'analysis_plots_val')
os.makedirs(img_dir, exist_ok=True)

# 加载meta信息
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
vocab_size = meta['vocab_size']

# 加载验证集数据（与train.py保持一致）
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# 加载图（注意：这是有向图）
graph_path = os.path.join(data_dir, "path_graph.graphml")
G = nx.read_graphml(graph_path)

# 加载邻接矩阵用于快速查询
adjacency_path = os.path.join(data_dir, "adjacency.npy")
adjacency_matrix = np.load(adjacency_path)

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

def get_successors_from_adjacency(node):
    """从邻接矩阵获取节点的所有后继节点"""
    return [i for i in range(num_nodes) if adjacency_matrix[node, i] > 0]

def get_batch_val(batch_size=val_batch_size):
    """获取验证集批次数据（与train.py中的get_batch保持一致）"""
    data = val_data
    data_size = block_size + 1
    ix = torch.randint((len(data) - data_size) // data_size, (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

def decode_tokens(tokens):
    """解码token序列，处理特殊情况"""
    result = []
    for t in tokens:
        if t < len(itos):
            result.append(itos[t])
    return result

def analyze_prediction_distribution_val(model, num_batches=10):
    """分析模型在验证集上的预测分布特征"""
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
        'top5_concentration': [],      
        'effective_choices': [],       
        'other_successor_probs': [],  
        'distribution_smoothness': [],
        'tf_accuracy_samples': []  # 新增：记录每个样本的准确率
    }
    
    total_analyzed = 0
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            X, Y = get_batch_val()
            logits, _ = model(X, Y)
            probs = F.softmax(logits, dim=-1)
            
            batch_size, seq_len, _ = logits.shape
            
            for i in range(batch_size):
                for j in range(seq_len):
                    # 获取当前token和目标token
                    if j > 0:  # 跳过第一个位置
                        current_token = X[i, j].item()
                        target_token = Y[i, j].item()
                        
                        # 转换为节点
                        current_node = token_to_node(current_token)
                        target_node = token_to_node(target_token)
                        
                        # 跳过非节点token
                        if current_node is None or target_node is None:
                            continue
                        
                        # 获取后继节点
                        successors = get_successors_from_adjacency(current_node)
                        
                        if not successors:
                            continue
                        
                        # 获取当前位置的概率分布
                        current_probs = probs[i, j].cpu().numpy()
                        
                        # 转换到token空间
                        successor_tokens = [node_to_token(n) for n in successors]
                        non_successor_nodes = [n for n in range(num_nodes) 
                                             if n not in successors and n != current_node]
                        non_successor_tokens = [node_to_token(n) for n in non_successor_nodes]
                        
                        # 计算统计量
                        successor_probs = current_probs[successor_tokens]
                        non_successor_probs = current_probs[non_successor_tokens]
                        
                        # 记录是否预测正确（用于TF accuracy）
                        predicted_token = torch.argmax(logits[i, j]).item()
                        results['tf_accuracy_samples'].append(predicted_token == target_token)
                        
                        # 目标token的概率
                        results['correct_token_prob'].append(current_probs[target_token])
                        
                        # 其他统计量
                        if len(successor_probs) > 0:
                            results['max_successor_prob'].append(successor_probs.max())
                            results['min_successor_prob'].append(successor_probs.min())
                            results['successor_prob_std'].append(successor_probs.std())
                            results['avg_successor_prob'].append(successor_probs.mean())
                        
                        if len(non_successor_probs) > 0:
                            results['avg_non_successor_prob'].append(non_successor_probs.mean())
                            ratio = successor_probs.mean() / (non_successor_probs.mean() + 1e-10)
                            results['successor_vs_non_successor_ratio'].append(ratio)
                        
                        # 计算熵（只考虑节点token）
                        node_tokens = list(range(NODE_OFFSET, NODE_OFFSET + num_nodes))
                        node_probs = current_probs[node_tokens]
                        node_probs = node_probs / (node_probs.sum() + 1e-10)
                        entropy = -(node_probs * np.log(node_probs + 1e-10)).sum()
                        results['entropy'].append(entropy)
                        
                        # 正确答案在后继节点中的排名
                        if target_node in successors:
                            target_prob = current_probs[target_token]
                            rank = sum(1 for p in successor_probs if p > target_prob) + 1
                            results['correct_successor_rank'].append(rank)
                        
                        # Top-5概率集中度
                        top5_probs = sorted(node_probs)[-5:]
                        results['top5_concentration'].append(sum(top5_probs))
                        
                        # 有效选择数
                        effective = sum(1 for p in node_probs if p > 0.01)
                        results['effective_choices'].append(effective)
                        
                        # 其他后继节点的概率
                        if target_node in successors:
                            other_successor_probs = [current_probs[node_to_token(s)] 
                                                   for s in successors if s != target_node]
                            results['other_successor_probs'].extend(other_successor_probs)
                        
                        # 分布平滑度
                        sorted_probs = sorted(node_probs)
                        n = len(sorted_probs)
                        if n > 0 and sum(sorted_probs) > 0:
                            gini = sum((2*i - n + 1) * p for i, p in enumerate(sorted_probs)) / (n * sum(sorted_probs))
                            results['distribution_smoothness'].append(1 - gini)
                        
                        total_analyzed += 1
    
    print(f"Analyzed {total_analyzed} predictions from validation set")
    
    # 计算TF accuracy
    if results['tf_accuracy_samples']:
        tf_accuracy = sum(results['tf_accuracy_samples']) / len(results['tf_accuracy_samples'])
        print(f"Teacher Forcing Accuracy on analyzed samples: {tf_accuracy:.4f}")
    
    return results

def visualize_val_sample_distribution(model, checkpoints_to_compare):
    """可视化验证集样本在不同checkpoint的概率分布"""
    
    # 获取一个验证集批次
    X, Y = get_batch_val(batch_size=1)
    
    # 找一个有意义的位置进行分析
    analyze_pos = 5  # 分析序列的第5个位置
    current_token = X[0, analyze_pos].item()
    target_token = Y[0, analyze_pos].item()
    
    current_node = token_to_node(current_token)
    target_node = token_to_node(target_token)
    
    if current_node is None or target_node is None:
        print("Selected position contains non-node tokens, trying another...")
        return
    
    successors = get_successors_from_adjacency(current_node)
    
    print(f"Analyzing validation sample: {current_node} -> {target_node}")
    print(f"Successors: {successors}")
    print(f"Is target a successor? {target_node in successors}")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, ckpt in enumerate(checkpoints_to_compare[:8]):
        ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
        model = load_model(ckpt_path)
        if model is None:
            continue
        
        with torch.no_grad():
            logits, _ = model(X)
            probs = F.softmax(logits[0, analyze_pos, :], dim=-1).cpu().numpy()
        
        # 获取节点概率
        node_probs = probs[NODE_OFFSET:NODE_OFFSET+num_nodes]
        
        # 准备数据
        target_prob = node_probs[target_node] if target_node < num_nodes else 0
        
        # 绘制
        ax = axes[idx]
        
        # 绘制概率分布
        heights = []
        colors = []
        labels = []
        
        # 目标答案
        if target_node in successors:
            heights.append(target_prob)
            colors.append('green')
            labels.append(f'Target ({target_node})')
        else:
            heights.append(target_prob)
            colors.append('red')
            labels.append(f'Target ({target_node})\n(not successor)')
        
        # 其他后继
        for s in successors[:5]:  # 最多显示5个
            if s != target_node:
                heights.append(node_probs[s])
                colors.append('orange')
                labels.append(f'Succ {s}')
        
        # 非后继平均
        non_successor_nodes = [n for n in range(num_nodes) if n not in successors and n != current_node]
        if non_successor_nodes:
            avg_non_succ = np.mean([node_probs[n] for n in non_successor_nodes])
            heights.append(avg_non_succ)
            colors.append('gray')
            labels.append('Non-succ\n(avg)')
        
        bars = ax.bar(range(len(heights)), heights, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Checkpoint {ckpt//1000}k')
        
        # 添加统计信息
        entropy = -sum(p * np.log(p + 1e-10) for p in node_probs if p > 0)
        effective_choices = sum(1 for p in node_probs if p > 0.01)
        predicted = np.argmax(node_probs)
        
        ax.text(0.5, 0.95, f'Entropy: {entropy:.2f}', transform=ax.transAxes, ha='center', fontsize=8)
        ax.text(0.5, 0.90, f'Effective: {effective_choices}', transform=ax.transAxes, ha='center', fontsize=8)
        ax.text(0.5, 0.85, f'Predicted: {predicted}', transform=ax.transAxes, ha='center', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(img_dir, 'val_distribution_evolution.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def plot_val_metrics_comparison(all_results, checkpoints):
    """绘制验证集上的关键指标，并与TF accuracy对比"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 准备数据
    ckpt_list = sorted([c for c in checkpoints if c in all_results])
    
    # 1. 正确答案的概率 vs TF Accuracy
    plt.subplot(2, 3, 1)
    correct_probs = [np.mean(all_results[c]['correct_token_prob']) for c in ckpt_list]
    tf_accuracies = [sum(all_results[c]['tf_accuracy_samples']) / len(all_results[c]['tf_accuracy_samples']) 
                     for c in ckpt_list]
    
    ax1 = plt.gca()
    ax1.plot(ckpt_list, correct_probs, 'b-', marker='o', label='Avg Correct Token Prob')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Probability', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(ckpt_list, tf_accuracies, 'r--', marker='s', label='TF Accuracy')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.axvline(x=8000, color='orange', linestyle='--', alpha=0.5)
    ax1.axvline(x=40000, color='green', linestyle='--', alpha=0.5)
    plt.title('Correct Token Probability vs TF Accuracy')
    
    # 2. 熵的变化
    plt.subplot(2, 3, 2)
    values = [np.mean(all_results[c]['entropy']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5, label='TF drop')
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5, label='Weights positive')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Prediction Entropy on Validation Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 有效选择数
    plt.subplot(2, 3, 3)
    values = [np.mean(all_results[c]['effective_choices']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Number of Choices')
    plt.title('Effective Choices (p > 0.01)')
    plt.grid(True, alpha=0.3)
    
    # 4. 后继vs非后继比率
    plt.subplot(2, 3, 4)
    values = [np.mean(all_results[c]['successor_vs_non_successor_ratio']) for c in ckpt_list if all_results[c]['successor_vs_non_successor_ratio']]
    valid_ckpts = [c for c in ckpt_list if all_results[c]['successor_vs_non_successor_ratio']]
    if values:
        plt.plot(valid_ckpts, values, marker='o')
        plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Ratio (log scale)')
        plt.title('Successor/Non-successor Ratio')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # 5. Top-5概率集中度
    plt.subplot(2, 3, 5)
    values = [np.mean(all_results[c]['top5_concentration']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Probability Sum')
    plt.title('Top-5 Probability Concentration')
    plt.grid(True, alpha=0.3)
    
    # 6. 分布平滑度
    plt.subplot(2, 3, 6)
    values = [np.mean(all_results[c]['distribution_smoothness']) for c in ckpt_list if all_results[c]['distribution_smoothness']]
    valid_ckpts = [c for c in ckpt_list if all_results[c]['distribution_smoothness']]
    if values:
        plt.plot(valid_ckpts, values, marker='o')
        plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=40000, color='g', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Smoothness (1-Gini)')
        plt.title('Distribution Smoothness')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(img_dir, 'val_metrics_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def save_val_analysis_report(all_results, checkpoints):
    """保存验证集分析报告"""
    
    report_path = os.path.join(img_dir, 'val_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VALIDATION SET PREDICTION DISTRIBUTION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("IMPORTANT: This analysis uses the validation set (val.bin),\n")
        f.write("which is the same dataset used for Teacher Forcing accuracy evaluation.\n\n")
        
        # 总体趋势
        f.write("1. CHECKPOINT ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        for ckpt in checkpoints:
            if ckpt in all_results:
                results = all_results[ckpt]
                
                # 计算TF accuracy
                tf_acc = 0
                if results['tf_accuracy_samples']:
                    tf_acc = sum(results['tf_accuracy_samples']) / len(results['tf_accuracy_samples'])
                
                f.write(f"\nCheckpoint {ckpt}:\n")
                f.write(f"  Teacher Forcing Accuracy: {tf_acc:.4f}\n")
                f.write(f"  Avg Correct Token Prob: {np.mean(results['correct_token_prob']):.4f}\n")
                f.write(f"  Avg Entropy: {np.mean(results['entropy']):.4f}\n")
                f.write(f"  Avg Effective Choices: {np.mean(results['effective_choices']):.2f}\n")
                if results['successor_vs_non_successor_ratio']:
                    f.write(f"  Successor/Non-successor Ratio: {np.mean(results['successor_vs_non_successor_ratio']):.2f}\n")
        
        # 8k-10k详细分析
        f.write("\n\n2. 8K-10K CRITICAL PERIOD ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        critical_checkpoints = [7000, 8000, 9000, 10000]
        for ckpt in critical_checkpoints:
            if ckpt in all_results:
                results = all_results[ckpt]
                tf_acc = sum(results['tf_accuracy_samples']) / len(results['tf_accuracy_samples']) if results['tf_accuracy_samples'] else 0
                
                f.write(f"\nCheckpoint {ckpt}:\n")
                f.write(f"  TF Accuracy: {tf_acc:.4f}\n")
                f.write(f"  Correct Token Prob: {np.mean(results['correct_token_prob']):.4f}\n")
                f.write(f"  Entropy: {np.mean(results['entropy']):.4f}\n")
                
                # 检查是否存在矛盾
                if tf_acc < 0.3 and np.mean(results['correct_token_prob']) > 0.8:
                    f.write("  ⚠️ ANOMALY: Low accuracy despite high correct token probability!\n")
        
        # 关键发现
        f.write("\n\n3. KEY FINDINGS\n")
        f.write("-"*40 + "\n")
        
        # 检查correct token prob和TF accuracy的关系
        if 8000 in all_results and 10000 in all_results:
            prob_8k = np.mean(all_results[8000]['correct_token_prob'])
            prob_10k = np.mean(all_results[10000]['correct_token_prob'])
            
            tf_8k = sum(all_results[8000]['tf_accuracy_samples']) / len(all_results[8000]['tf_accuracy_samples'])
            tf_10k = sum(all_results[10000]['tf_accuracy_samples']) / len(all_results[10000]['tf_accuracy_samples'])
            
            f.write(f"\n8k->10k changes:\n")
            f.write(f"  Correct token prob: {prob_8k:.4f} -> {prob_10k:.4f} (Δ={prob_10k-prob_8k:+.4f})\n")
            f.write(f"  TF accuracy: {tf_8k:.4f} -> {tf_10k:.4f} (Δ={tf_10k-tf_8k:+.4f})\n")
            
            if abs(tf_10k - tf_8k) > 0.5 and abs(prob_10k - prob_8k) < 0.1:
                f.write("\n⚠️ Large TF accuracy drop without corresponding probability change!\n")
                f.write("This suggests the issue is not simple confidence collapse.\n")
    
    print(f"Saved report: {report_path}")

def main():
    # 要分析的checkpoints
    checkpoints = [1000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 
                   50000, 60000, 70000, 80000, 90000, 100000]
    
    print("="*80)
    print("VALIDATION SET ANALYSIS")
    print("="*80)
    print("This analysis uses val.bin - the same dataset used for Teacher Forcing accuracy")
    print("="*80 + "\n")
    
    # 1. 分析每个checkpoint在验证集上的表现
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
        results = analyze_prediction_distribution_val(model, num_batches=20)
        all_results[ckpt] = results
        
        # 打印关键统计
        tf_acc = sum(results['tf_accuracy_samples']) / len(results['tf_accuracy_samples']) if results['tf_accuracy_samples'] else 0
        print(f"Checkpoint {ckpt} summary:")
        print(f"  Teacher Forcing Accuracy: {tf_acc:.4f}")
        print(f"  Avg correct token prob: {np.mean(results['correct_token_prob']):.4f}")
        print(f"  Avg entropy: {np.mean(results['entropy']):.4f}")
        print(f"  Avg effective choices: {np.mean(results['effective_choices']):.2f}")
        
        # 特别关注8k-10k
        if ckpt in [8000, 9000, 10000]:
            print(f"  *** Critical checkpoint - TF collapse period ***")
    
    # 2. 绘制验证集指标对比图
    print("\n" + "="*80)
    print("PLOTTING VALIDATION METRICS")
    print("="*80)
    
    plot_val_metrics_comparison(all_results, checkpoints)
    
    # 3. 可视化具体样本的概率分布
    print("\n" + "="*80)
    print("VISUALIZING SAMPLE DISTRIBUTIONS")
    print("="*80)
    
    # 加载任意一个模型用于可视化
    if checkpoints:
        model = load_model(os.path.join(out_dir, f'{checkpoints[0]}_ckpt_20.pt'))
        if model:
            visualize_val_sample_distribution(model, 
                                            [1000, 8000, 10000, 20000, 40000, 60000, 80000, 100000])
    
    # 4. 保存分析报告
    print("\n" + "="*80)
    print("SAVING ANALYSIS REPORT")
    print("="*80)
    
    save_val_analysis_report(all_results, checkpoints)
    
    print(f"\nAll plots and report saved to: {img_dir}")
    print("\nIMPORTANT: This analysis now uses the validation set,")
    print("ensuring consistency with Teacher Forcing accuracy measurements.")

if __name__ == "__main__":
    main()