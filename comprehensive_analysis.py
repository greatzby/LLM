"""
综合分析脚本 - 对所有checkpoint进行全面分析
运行方式: python comprehensive_analysis.py --num_nodes 100 --config 1_1_120
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    parser.add_argument('--checkpoints', type=int, nargs='+', 
                       default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 
                               10000, 15000, 20000, 25000, 30000, 35000, 40000, 
                               45000, 50000, 60000, 70000, 80000, 90000, 100000])
    return parser.parse_args()

def load_model_from_checkpoint(checkpoint_path, device):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint['iter_num']

def analyze_checkpoint(model, test_data, vocab_size, device):
    """分析单个checkpoint的各项指标"""
    results = {
        'weight_stats': {},
        'embedding_stats': {},
        'logit_stats': {},
        'hidden_stats': {}
    }
    
    # 1. 分析权重矩阵
    lm_head_weight = model.lm_head.weight.data.cpu().numpy()
    
    # 获取所有权重的统计
    results['weight_stats']['all_mean'] = float(np.mean(lm_head_weight))
    results['weight_stats']['all_std'] = float(np.std(lm_head_weight))
    results['weight_stats']['all_max'] = float(np.max(lm_head_weight))
    results['weight_stats']['all_min'] = float(np.min(lm_head_weight))
    
    # 简化的edge/non-edge分析（根据实际情况调整）
    # 假设对角线元素代表自循环边
    diagonal_weights = np.diagonal(lm_head_weight[2:, 2:])  # 跳过特殊token
    off_diagonal = lm_head_weight[2:, 2:].copy()
    np.fill_diagonal(off_diagonal, 0)
    off_diagonal_weights = off_diagonal[off_diagonal != 0]
    
    results['weight_stats']['edge_mean'] = float(np.mean(diagonal_weights))
    results['weight_stats']['non_edge_mean'] = float(np.mean(off_diagonal_weights))
    results['weight_stats']['weight_gap'] = float(results['weight_stats']['edge_mean'] - results['weight_stats']['non_edge_mean'])
    
    # 2. 分析embedding
    embeddings = model.transformer.wte.weight.data.cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1)
    results['embedding_stats']['norm_mean'] = float(np.mean(norms))
    results['embedding_stats']['norm_std'] = float(np.std(norms))
    results['embedding_stats']['norm_max'] = float(np.max(norms))
    results['embedding_stats']['norm_min'] = float(np.min(norms))
    
    # 3. 计算embedding之间的相似度
    # 归一化embeddings
    normalized_embeddings = embeddings / (norms[:, np.newaxis] + 1e-8)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # 获取相似度统计
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    results['embedding_stats']['similarity_mean'] = float(np.mean(upper_triangle))
    results['embedding_stats']['similarity_std'] = float(np.std(upper_triangle))
    
    # 4. 分析transformer内部的权重
    # 分析attention权重
    if hasattr(model.transformer.h[0].attn, 'c_attn'):
        attn_weight = model.transformer.h[0].attn.c_attn.weight.data.cpu().numpy()
        results['hidden_stats']['attn_weight_norm'] = float(np.linalg.norm(attn_weight))
        results['hidden_stats']['attn_weight_mean'] = float(np.mean(attn_weight))
    
    # 分析MLP权重
    if hasattr(model.transformer.h[0].mlp, 'c_fc'):
        mlp_weight = model.transformer.h[0].mlp.c_fc.weight.data.cpu().numpy()
        results['hidden_stats']['mlp_weight_norm'] = float(np.linalg.norm(mlp_weight))
        results['hidden_stats']['mlp_weight_mean'] = float(np.mean(mlp_weight))
    
    return results

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置路径
    dataset = 'simple_graph'
    out_dir = f'out/{dataset}_{args.config}_{args.num_nodes}'
    
    # 使用指定的checkpoints
    checkpoints = []
    for iter_num in args.checkpoints:
        ckpt_path = os.path.join(out_dir, f'{iter_num}_ckpt_20.pt')
        if os.path.exists(ckpt_path):
            checkpoints.append((iter_num, ckpt_path))
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found")
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints to analyze")
    
    # 加载meta信息
    meta_path = f'data/{dataset}/{args.num_nodes}/meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    
    # 检查是否有已保存的进度
    progress_file = os.path.join(args.output_dir, 'analysis_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            all_results = json.load(f)
        print(f"Resuming from previous run, {len(all_results)} checkpoints already analyzed")
    else:
        all_results = {}
    
    # 分析所有checkpoint
    for iter_num, ckpt_path in tqdm(checkpoints):
        if str(iter_num) in all_results:
            print(f"Skipping checkpoint {iter_num} (already analyzed)")
            continue
            
        print(f"\nAnalyzing checkpoint at iteration {iter_num}")
        
        try:
            model, _ = load_model_from_checkpoint(ckpt_path, args.device)
            results = analyze_checkpoint(model, None, vocab_size, args.device)
            results['iteration'] = iter_num
            all_results[str(iter_num)] = results
            
            # 保存进度
            with open(progress_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error analyzing checkpoint {iter_num}: {e}")
            continue
    
    # 保存最终结果
    with open(os.path.join(args.output_dir, 'comprehensive_analysis.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成汇总表格
    df_data = []
    for iter_num, res in all_results.items():
        row = {
            'iteration': int(iter_num),
            'edge_weight_mean': res['weight_stats']['edge_mean'],
            'non_edge_weight_mean': res['weight_stats']['non_edge_mean'],
            'weight_gap': res['weight_stats']['weight_gap'],
            'embedding_norm_mean': res['embedding_stats']['norm_mean'],
            'embedding_norm_std': res['embedding_stats']['norm_std'],
            'embedding_similarity_mean': res['embedding_stats']['similarity_mean']
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('iteration')
    df.to_csv(os.path.join(args.output_dir, 'summary_statistics.csv'), index=False)
    
    # 创建基本可视化
    create_basic_plots(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")

def create_basic_plots(df, output_dir):
    """创建基本的可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Weight evolution
    ax = axes[0, 0]
    ax.plot(df['iteration'], df['edge_weight_mean'], 'b-', label='Edge weights', marker='o', markersize=4)
    ax.plot(df['iteration'], df['non_edge_weight_mean'], 'r-', label='Non-edge weights', marker='s', markersize=4)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Weight Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Weight gap
    ax = axes[0, 1]
    ax.plot(df['iteration'], df['weight_gap'], 'g-', marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight Gap')
    ax.set_title('Edge vs Non-edge Weight Gap')
    ax.grid(True, alpha=0.3)
    
    # 3. Embedding norm
    ax = axes[1, 0]
    ax.plot(df['iteration'], df['embedding_norm_mean'], 'purple', marker='o', markersize=4)
    ax.fill_between(df['iteration'], 
                     df['embedding_norm_mean'] - df['embedding_norm_std'],
                     df['embedding_norm_mean'] + df['embedding_norm_std'],
                     alpha=0.3, color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Embedding Norm')
    ax.set_title('Embedding Norm Evolution')
    ax.grid(True, alpha=0.3)
    
    # 4. Embedding similarity
    ax = axes[1, 1]
    ax.plot(df['iteration'], df['embedding_similarity_mean'], 'orange', marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Similarity')
    ax.set_title('Embedding Similarity Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis_plots.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()