"""
analyze_hidden_weight_interaction.py
分析为什么positive weights仍然产生negative logits
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse
from model import GPT, GPTConfig

def load_model_and_data(checkpoint_path, data_dir):
    """加载模型和数据"""
    # 加载meta信息
    with open(Path(data_dir) / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # 模型配置
    model_args = {
        'n_layer': 1,
        'n_head': 1, 
        'n_embd': 120,
        'block_size': meta['block_size'],
        'vocab_size': meta['vocab_size'],
        'bias': False,
        'dropout': 0.0
    }
    
    # 加载模型
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    # 加载验证数据
    val_data = np.memmap(Path(data_dir) / 'val.bin', dtype=np.uint16, mode='r')
    
    return model, meta, val_data

def get_batch(val_data, block_size, batch_size=32):
    """获取一个batch的数据"""
    data_size = block_size + 1
    ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.cuda(), y.cuda()

def get_hidden_states(model, x):
    """手动forward以获取hidden states"""
    with torch.no_grad():
        # 1. Embeddings
        tok_emb = model.transformer.wte(x)
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = model.transformer.wpe(pos)
        x_emb = tok_emb + pos_emb
        
        # 2. 通过唯一的Block
        # 由于只有1层，直接处理
        h = x_emb
        block = model.transformer.h[0]
        
        # Attention部分
        h_ln1 = block.ln_1(h)
        h_attn = block.attn(h_ln1)
        h = h + h_attn
        
        # MLP部分
        h_ln2 = block.ln_2(h)
        h_mlp = block.mlp(h_ln2)
        h = h + h_mlp
        
        # 3. Final LayerNorm
        hidden_states = model.transformer.ln_f(h)
        
        # 4. 计算logits（用于验证）
        logits = model.lm_head(hidden_states)
        
    return {
        'embeddings': x_emb,
        'after_attn': h_attn,
        'after_block': h,
        'after_ln_f': hidden_states,
        'logits': logits
    }

def analyze_logit_computation(model, meta, val_data):
    """详细分析logit计算"""
    # 获取数据
    x, y = get_batch(val_data, meta['block_size'], batch_size=64)
    
    # 获取所有中间状态
    states = get_hidden_states(model, x)
    hidden_states = states['after_ln_f']  # [batch, seq, n_embd]
    logits = states['logits']  # [batch, seq, vocab]
    
    # 获取output weights
    W_output = model.lm_head.weight.T  # [n_embd, vocab]
    
    # 识别edge和non-edge indices
    # 假设：节点编号从0到99，边的token从100开始
    node_tokens = list(range(100))
    edge_tokens = [i for i in range(len(meta['itos'])) if i >= 100]
    
    # 分析不同位置的hidden states
    positions_to_analyze = [5, 10, 15]  # 选择几个代表性位置
    results = {}
    
    for pos in positions_to_analyze:
        h_at_pos = hidden_states[:, pos, :]  # [batch, n_embd]
        
        # 计算与edge/non-edge weights的点积
        edge_logits = h_at_pos @ W_output[:, edge_tokens]  # [batch, n_edge]
        non_edge_logits = h_at_pos @ W_output[:, node_tokens]  # [batch, n_nodes]
        
        # 详细分析一个样本
        sample_idx = 0
        h_sample = h_at_pos[sample_idx]  # [n_embd]
        
        # 分析hidden state的特性
        h_positive = (h_sample > 0).float().mean().item()
        h_negative = (h_sample < 0).float().mean().item()
        h_mean = h_sample.mean().item()
        h_std = h_sample.std().item()
        h_norm = h_sample.norm().item()
        
        # 分析weights
        w_non_edge_mean = W_output[:, node_tokens].mean().item()
        w_edge_mean = W_output[:, edge_tokens].mean().item()
        
        # 逐维度分析贡献
        # 选择一个典型的non-edge node
        node_idx = node_tokens[50]  # 节点50
        w_node = W_output[:, node_idx]  # [n_embd]
        
        # 计算每个维度的贡献
        contributions = h_sample * w_node  # [n_embd]
        pos_contrib = contributions[contributions > 0].sum().item()
        neg_contrib = contributions[contributions < 0].sum().item()
        total_logit = contributions.sum().item()
        
        # 验证计算
        computed_logit = (h_sample @ w_node).item()
        actual_logit = logits[sample_idx, pos, node_idx].item()
        
        results[f'position_{pos}'] = {
            'hidden_state': {
                'percent_positive': h_positive,
                'percent_negative': h_negative,
                'mean': h_mean,
                'std': h_std,
                'norm': h_norm
            },
            'logits': {
                'edge_mean': edge_logits.mean().item(),
                'non_edge_mean': non_edge_logits.mean().item(),
                'edge_std': edge_logits.std().item(),
                'non_edge_std': non_edge_logits.std().item()
            },
            'detailed_analysis': {
                'weight_mean': w_node.mean().item(),
                'positive_contribution': pos_contrib,
                'negative_contribution': neg_contrib,
                'total_logit': total_logit,
                'computed_logit': computed_logit,
                'actual_logit': actual_logit,
                'verification': abs(computed_logit - actual_logit) < 1e-5
            }
        }
    
    return results, hidden_states, W_output

def visualize_analysis(results, hidden_states, W_output, output_dir):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Hidden state statistics across positions
    positions = []
    h_means = []
    h_pos_percent = []
    logit_means = []
    
    for key, data in results.items():
        pos = int(key.split('_')[1])
        positions.append(pos)
        h_means.append(data['hidden_state']['mean'])
        h_pos_percent.append(data['hidden_state']['percent_positive'])
        logit_means.append(data['logits']['non_edge_mean'])
    
    ax = axes[0, 0]
    ax.plot(positions, h_means, 'b-o', label='Hidden mean')
    ax.plot(positions, logit_means, 'r-s', label='Non-edge logit mean')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Hidden State vs Logit Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Contribution breakdown
    ax = axes[0, 1]
    pos_contribs = [results[f'position_{p}']['detailed_analysis']['positive_contribution'] 
                    for p in [5, 10, 15]]
    neg_contribs = [results[f'position_{p}']['detailed_analysis']['negative_contribution']
                    for p in [5, 10, 15]]
    
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, pos_contribs, width, label='Positive', color='green', alpha=0.7)
    ax.bar(x + width/2, neg_contribs, width, label='Negative', color='red', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Contribution')
    ax.set_title('Positive vs Negative Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels(['5', '10', '15'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Hidden state distribution (position 10)
    ax = axes[0, 2]
    h_sample = hidden_states[0, 10, :].cpu().numpy()
    ax.hist(h_sample, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Hidden State Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Hidden State Distribution (pos=10)\nMean={h_sample.mean():.3f}')
    ax.grid(True, alpha=0.3)
    
    # 4. Weight distribution
    ax = axes[1, 0]
    node_tokens = list(range(100))
    w_non_edge = W_output[:, node_tokens].mean(dim=1).cpu().numpy()
    ax.hist(w_non_edge, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Non-edge Weight (averaged)')
    ax.set_ylabel('Count')
    ax.set_title(f'Non-edge Weight Distribution\nMean={w_non_edge.mean():.4f}')
    ax.grid(True, alpha=0.3)
    
    # 5. Dimension-wise contribution heatmap
    ax = axes[1, 1]
    # 选择一个样本和位置
    h = hidden_states[0, 10, :].cpu().numpy()
    w = W_output[:, 50].cpu().numpy()  # node 50的weights
    contributions = h * w
    
    # 创建贡献矩阵用于可视化
    contrib_matrix = contributions.reshape(12, 10)  # 120 = 12x10
    im = ax.imshow(contrib_matrix, cmap='RdBu_r', center=0)
    ax.set_title('Dimension-wise Contributions\n(Hidden * Weight)')
    ax.set_xlabel('Dimension (mod 10)')
    ax.set_ylabel('Dimension (div 10)')
    plt.colorbar(im, ax=ax)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "Summary Analysis:\n\n"
    
    for pos in [5, 10, 15]:
        data = results[f'position_{pos}']
        summary_text += f"Position {pos}:\n"
        summary_text += f"  Hidden mean: {data['hidden_state']['mean']:.4f}\n"
        summary_text += f"  Positive dims: {data['hidden_state']['percent_positive']:.1%}\n"
        summary_text += f"  Non-edge logit: {data['logits']['non_edge_mean']:.4f}\n"
        summary_text += f"  Pos contrib: {data['detailed_analysis']['positive_contribution']:.2f}\n"
        summary_text += f"  Neg contrib: {data['detailed_analysis']['negative_contribution']:.2f}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'hidden_weight_analysis.png', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='out/simple_graph_1_1_120_100/40000_ckpt_20.pt')
    parser.add_argument('--data_dir', type=str, 
                       default='data/simple_graph/100')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis_results')
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("Loading model and data...")
    model, meta, val_data = load_model_and_data(args.checkpoint, args.data_dir)
    
    print("Analyzing logit computation...")
    results, hidden_states, W_output = analyze_logit_computation(model, meta, val_data)
    
    print("\nDetailed Results:")
    for pos_key, data in results.items():
        print(f"\n{pos_key}:")
        print(f"  Hidden state mean: {data['hidden_state']['mean']:.4f}")
        print(f"  Non-edge logit mean: {data['logits']['non_edge_mean']:.4f}")
        print(f"  Weight mean: {data['detailed_analysis']['weight_mean']:.4f}")
        print(f"  Positive contribution: {data['detailed_analysis']['positive_contribution']:.4f}")
        print(f"  Negative contribution: {data['detailed_analysis']['negative_contribution']:.4f}")
        print(f"  Total logit: {data['detailed_analysis']['total_logit']:.4f}")
        print(f"  Verification passed: {data['detailed_analysis']['verification']}")
    
    print("\nCreating visualizations...")
    visualize_analysis(results, hidden_states, W_output, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()