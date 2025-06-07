"""
analyze_predictions.py - 分析模型在不同训练阶段的预测分布变化
特别关注8k-10k步的异常现象
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

# 加载meta信息
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
vocab_size = meta['vocab_size']

# 加载邻接矩阵
adjacency = np.load(os.path.join(data_dir, 'adjacency.npy'))

# 加载图
graph_path = os.path.join(data_dir, "path_graph.graphml")
if os.path.exists(graph_path):
    G = nx.read_graphml(graph_path)
else:
    print("使用邻接矩阵构建图")
    G = nx.from_numpy_array(adjacency)

def load_model(checkpoint_path):
    """加载模型checkpoint"""
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

def get_neighbors(node):
    """获取节点的所有邻居"""
    return list(np.where(adjacency[node] == 1)[0])

def analyze_prediction_distribution(model, test_samples, num_samples=100):
    """分析模型的预测分布特征"""
    results = {
        'correct_token_prob': [],      # 正确答案的概率
        'max_neighbor_prob': [],       # 所有邻居中的最大概率
        'min_neighbor_prob': [],       # 所有邻居中的最小概率
        'neighbor_prob_std': [],       # 邻居间概率的标准差
        'avg_neighbor_prob': [],       # 邻居的平均概率
        'avg_non_neighbor_prob': [],   # 非邻居的平均概率
        'entropy': [],                 # 整个分布的熵
        'correct_neighbor_rank': [],   # 正确答案在邻居中的排名
        'neighbor_vs_non_neighbor_ratio': []  # 邻居/非邻居平均概率比
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
                
                # 获取邻居和非邻居
                neighbors = get_neighbors(current_node)
                non_neighbors = [i for i in range(num_nodes) if i not in neighbors and i != current_node]
                
                if correct_next not in neighbors:
                    continue  # 跳过错误的ground truth
                
                # 计算统计量
                neighbor_probs = probs[neighbors]
                non_neighbor_probs = probs[non_neighbors]
                
                results['correct_token_prob'].append(probs[correct_next])
                results['max_neighbor_prob'].append(neighbor_probs.max())
                results['min_neighbor_prob'].append(neighbor_probs.min())
                results['neighbor_prob_std'].append(neighbor_probs.std())
                results['avg_neighbor_prob'].append(neighbor_probs.mean())
                results['avg_non_neighbor_prob'].append(non_neighbor_probs.mean())
                
                # 计算熵
                entropy = -(probs * np.log(probs + 1e-10)).sum()
                results['entropy'].append(entropy)
                
                # 正确答案在邻居中的排名
                correct_prob = probs[correct_next]
                rank = sum(1 for p in neighbor_probs if p > correct_prob) + 1
                results['correct_neighbor_rank'].append(rank)
                
                # 邻居vs非邻居比率
                ratio = neighbor_probs.mean() / (non_neighbor_probs.mean() + 1e-10)
                results['neighbor_vs_non_neighbor_ratio'].append(ratio)
                
                analyzed += 1
                if analyzed >= num_samples:
                    break
                    
            if analyzed >= num_samples:
                break
    
    print(f"Analyzed {analyzed} predictions")
    return results

def analyze_specific_examples(checkpoints, test_samples, num_examples=5):
    """分析具体例子在不同checkpoint的预测变化"""
    # 选择固定的测试样本
    selected_samples = test_samples[:num_examples]
    
    for sample_idx, (prompt, full_path) in enumerate(selected_samples):
        print(f"\n{'='*80}")
        print(f"Example {sample_idx + 1}")
        print(f"Full path: {full_path}")
        
        # 解析路径
        path_nodes = re.findall(r'\d+', full_path)
        if len(path_nodes) < 4:
            continue
            
        # 选择路径中间的一个预测点
        analyze_pos = min(5, len(path_nodes) - 2)  # 分析第5步或倒数第二步
        current_node = int(path_nodes[analyze_pos])
        correct_next = int(path_nodes[analyze_pos + 1])
        
        print(f"Analyzing step {analyze_pos}: {current_node} -> ?")
        print(f"Correct next: {correct_next}")
        
        neighbors = get_neighbors(current_node)
        print(f"Neighbors of {current_node}: {neighbors}")
        print(f"Is correct next a neighbor? {correct_next in neighbors}")
        
        # 构建输入
        input_seq = encode(" ".join(path_nodes[:analyze_pos+1]))
        input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
        
        # 对每个checkpoint分析
        for ckpt in checkpoints:
            ckpt_path = os.path.join(out_dir, f'{ckpt}_ckpt_20.pt')
            if not os.path.exists(ckpt_path):
                continue
                
            model = load_model(ckpt_path)
            
            with torch.no_grad():
                logits, _ = model(input_tensor)
                logits = logits[0, -1, :]
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            print(f"\n--- Checkpoint {ckpt} ---")
            
            # 显示top 10预测
            top_k_indices = probs.argsort()[-10:][::-1]
            for rank, idx in enumerate(top_k_indices):
                if idx >= num_nodes:
                    continue
                is_neighbor = "✓" if idx in neighbors else "✗"
                is_correct = "★" if idx == correct_next else " "
                print(f"{rank+1}. {is_correct}{is_neighbor} Node {idx}: {probs[idx]:.4f}")
            
            # 显示邻居概率统计
            neighbor_probs = probs[neighbors]
            print(f"\nNeighbor statistics:")
            print(f"  Max: {neighbor_probs.max():.4f}, Min: {neighbor_probs.min():.4f}")
            print(f"  Mean: {neighbor_probs.mean():.4f}, Std: {neighbor_probs.std():.4f}")
            print(f"  Correct token prob: {probs[correct_next]:.4f}")
            print(f"  Correct token rank among neighbors: {sum(1 for p in neighbor_probs if p > probs[correct_next]) + 1}/{len(neighbors)}")

def load_test_samples():
    """加载测试样本"""
    test_file = os.path.join(data_dir, 'test.txt')
    samples = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设格式是完整路径
            prompt = line.split(':')[0] + ':' if ':' in line else line
            samples.append((prompt, line))
    
    return samples

def main():
    # 要分析的checkpoint
    checkpoints = [7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000]
    
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
        print(f"\nAnalyzing checkpoint {ckpt}...")
        results = analyze_prediction_distribution(model, test_samples, num_samples=200)
        all_results[ckpt] = results
        
        # 打印关键统计
        print(f"Checkpoint {ckpt} summary:")
        print(f"  Avg correct token prob: {np.mean(results['correct_token_prob']):.4f}")
        print(f"  Avg neighbor prob std: {np.mean(results['neighbor_prob_std']):.4f}")
        print(f"  Avg neighbor/non-neighbor ratio: {np.mean(results['neighbor_vs_non_neighbor_ratio']):.2f}")
        print(f"  Avg entropy: {np.mean(results['entropy']):.4f}")
        print(f"  Avg correct token rank: {np.mean(results['correct_neighbor_rank']):.2f}")
    
    # 2. 绘制关键指标随训练步数的变化
    plt.figure(figsize=(15, 10))
    
    # 准备数据
    ckpt_list = sorted([c for c in checkpoints if c in all_results])
    
    # 2.1 正确答案的概率
    plt.subplot(2, 3, 1)
    values = [np.mean(all_results[c]['correct_token_prob']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5, label='TF accuracy drop')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title('Average Probability of Correct Token')
    plt.legend()
    
    # 2.2 邻居间概率的标准差
    plt.subplot(2, 3, 2)
    values = [np.mean(all_results[c]['neighbor_prob_std']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Std Dev')
    plt.title('Std of Neighbor Probabilities\n(Lower = more uniform)')
    
    # 2.3 邻居vs非邻居的概率比
    plt.subplot(2, 3, 3)
    values = [np.mean(all_results[c]['neighbor_vs_non_neighbor_ratio']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Ratio')
    plt.title('Neighbor/Non-neighbor Probability Ratio')
    plt.yscale('log')
    
    # 2.4 预测分布的熵
    plt.subplot(2, 3, 4)
    values = [np.mean(all_results[c]['entropy']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Prediction Entropy')
    
    # 2.5 正确答案的平均排名
    plt.subplot(2, 3, 5)
    values = [np.mean(all_results[c]['correct_neighbor_rank']) for c in ckpt_list]
    plt.plot(ckpt_list, values, marker='o')
    plt.axvline(x=8000, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Average Rank')
    plt.title('Average Rank of Correct Token\n(among neighbors)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prediction_distribution_analysis.png'), dpi=150)
    plt.show()
    
    # 3. 分析具体例子
    print("\n" + "="*80)
    print("SPECIFIC EXAMPLE ANALYSIS")
    print("="*80)
    analyze_specific_examples([7000, 8000, 9000, 10000, 20000, 40000], test_samples, num_examples=3)
    
    # 4. 特别分析8k-10k的变化
    print("\n" + "="*80)
    print("SPECIAL ANALYSIS: 8k-10k transition")
    print("="*80)
    
    if 8000 in all_results and 10000 in all_results:
        print("\nComparing 8k vs 10k:")
        print(f"Correct token prob: {np.mean(all_results[8000]['correct_token_prob']):.4f} -> {np.mean(all_results[10000]['correct_token_prob']):.4f}")
        print(f"Neighbor prob std: {np.mean(all_results[8000]['neighbor_prob_std']):.4f} -> {np.mean(all_results[10000]['neighbor_prob_std']):.4f}")
        print(f"Neighbor/non-neighbor ratio: {np.mean(all_results[8000]['neighbor_vs_non_neighbor_ratio']):.2f} -> {np.mean(all_results[10000]['neighbor_vs_non_neighbor_ratio']):.2f}")
        
        # 检查是否邻居概率变得更均匀
        std_8k = np.mean(all_results[8000]['neighbor_prob_std'])
        std_10k = np.mean(all_results[10000]['neighbor_prob_std'])
        if std_10k < std_8k * 0.5:
            print("\n⚠️ 邻居间概率标准差大幅下降，表明模型失去了区分不同邻居的能力！")

if __name__ == "__main__":
    main()