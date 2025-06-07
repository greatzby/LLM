"""
analyze_hidden_weight_interaction_v3.py
修复logits计算问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from model import GPT, GPTConfig

def load_model_and_data(checkpoint_path, data_dir):
    """加载模型和数据"""
    with open(Path(data_dir) / 'meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    model_args = {
        'n_layer': 1,
        'n_head': 1, 
        'n_embd': 120,
        'block_size': meta['block_size'],
        'vocab_size': meta['vocab_size'],
        'bias': False,
        'dropout': 0.0
    }
    
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    val_data = np.memmap(Path(data_dir) / 'val.bin', dtype=np.uint16, mode='r')
    
    return model, meta, val_data

def get_full_forward(model, x):
    """手动执行完整forward以获取所有位置的logits和hidden states"""
    with torch.no_grad():
        # 1. Embeddings
        tok_emb = model.transformer.wte(x)
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = model.transformer.wpe(pos)
        h = tok_emb + pos_emb
        
        # 2. 通过Block
        h = model.transformer.drop(h)
        for block in model.transformer.h:
            h = block(h)
        
        # 3. Final LayerNorm
        hidden_states = model.transformer.ln_f(h)
        
        # 4. 计算所有位置的logits（不只是最后一个）
        logits = model.lm_head(hidden_states)  # [batch, seq, vocab]
        
    return logits, hidden_states

def analyze_logits_correctly(model, meta, val_data):
    """正确分析logits"""
    # 获取一批数据
    block_size = meta['block_size']
    batch_size = 32
    
    # 随机采样
    data_size = block_size + 1
    num_batches = (len(val_data) - data_size) // data_size
    if num_batches < batch_size:
        batch_size = num_batches
        
    ix = torch.randint(num_batches, (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.cuda(), y.cuda()
    
    # 获取完整的logits和hidden states
    logits, hidden_states = get_full_forward(model, x)
    
    # 获取output weights
    W_output = model.lm_head.weight.T.detach()  # [n_embd, vocab]
    
    # 分析每个位置
    results = {
        'successor_logits': [],
        'non_successor_logits': [],
        'positions': [],
        'hidden_means': [],
        'weight_stats': []
    }
    
    # 节点对应的token indices（2-101对应节点0-99）
    node_tokens = list(range(2, min(102, meta['vocab_size'])))
    
    # 分析位置5-20（或更少，如果序列较短）
    start_pos = 5
    end_pos = min(20, x.shape[1] - 1)
    
    for pos in range(start_pos, end_pos):
        # 获取这个位置的正确答案
        correct_next = y[:, pos]  # [batch]
        
        # 只分析有效的节点预测（忽略PAD和换行）
        valid_mask = (correct_next >= 2) & (correct_next < 102)
        if not valid_mask.any():
            continue
            
        # 获取这个位置的logits
        pos_logits = logits[valid_mask, pos, :]  # [valid_batch, vocab]
        pos_hidden = hidden_states[valid_mask, pos, :]  # [valid_batch, n_embd]
        correct_next_valid = correct_next[valid_mask]
        
        # 记录hidden state统计
        results['hidden_means'].append(pos_hidden.mean().item())
        
        # 分离successor和non-successor logits
        for b in range(pos_logits.shape[0]):
            correct_token = correct_next_valid[b].item()
            
            # Successor logit（正确的下一个节点）
            successor_logit = pos_logits[b, correct_token].item()
            results['successor_logits'].append(successor_logit)
            
            # Non-successor logits（其他所有节点）
            for node_token in node_tokens:
                if node_token != correct_token:
                    results['non_successor_logits'].append(pos_logits[b, node_token].item())
            
        results['positions'].append(pos)
    
    # 分析节点对应的weights
    node_weights = W_output[:, node_tokens]  # [n_embd, num_nodes]
    
    # 详细分析一个样本
    sample_pos = 10 if 10 < end_pos else start_pos
    sample_hidden = hidden_states[0, sample_pos, :].detach()
    sample_correct = y[0, sample_pos].item()
    
    # 计算贡献
    contributions = {}
    if 2 <= sample_correct < 102:
        w_correct = W_output[:, sample_correct].detach()
        contrib = sample_hidden * w_correct
        contributions['positive'] = contrib[contrib > 0].sum().item()
        contributions['negative'] = contrib[contrib < 0].sum().item()
        contributions['total'] = contrib.sum().item()
    
    analysis = {
        'logits': {
            'successor_mean': np.mean(results['successor_logits']) if results['successor_logits'] else 0,
            'non_successor_mean': np.mean(results['non_successor_logits']) if results['non_successor_logits'] else 0,
            'successor_std': np.std(results['successor_logits']) if results['successor_logits'] else 0,
            'non_successor_std': np.std(results['non_successor_logits']) if results['non_successor_logits'] else 0,
            'num_samples': len(results['successor_logits'])
        },
        'weights': {
            'mean': node_weights.mean().item(),
            'std': node_weights.std().item(),
            'min': node_weights.min().item(),
            'max': node_weights.max().item()
        },
        'hidden_states': hidden_states,
        'W_output': W_output,
        'raw_results': results,
        'contributions': contributions
    }
    
    return analysis

def visualize_correct_analysis(analysis, output_dir):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Logit分布对比
    ax = axes[0, 0]
    if analysis['raw_results']['successor_logits']:
        bins = 30
        ax.hist(analysis['raw_results']['successor_logits'], bins=bins, alpha=0.5, 
                label=f"Successor (n={len(analysis['raw_results']['successor_logits'])})", 
                color='green', density=True)
        ax.hist(analysis['raw_results']['non_successor_logits'], bins=bins, alpha=0.5, 
                label=f"Non-successor (n={len(analysis['raw_results']['non_successor_logits'])})", 
                color='red', density=True)
        
        ax.axvline(analysis['logits']['successor_mean'], color='green', 
                   linestyle='--', linewidth=2)
        ax.axvline(analysis['logits']['non_successor_mean'], color='red', 
                   linestyle='--', linewidth=2)
    
    ax.set_xlabel('Logit Value')
    ax.set_ylabel('Density')
    ax.set_title('Successor vs Non-successor Logits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Weight分布
    ax = axes[0, 1]
    W = analysis['W_output'][:, 2:102].detach().cpu().numpy()
    ax.hist(W.flatten(), bins=50, alpha=0.7, color='blue')
    ax.axvline(W.mean(), color='red', linestyle='--', linewidth=2)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Node Weight Distribution\nMean: {W.mean():.4f}')
    ax.grid(True, alpha=0.3)
    
    # 3. Hidden state统计
    ax = axes[0, 2]
    if analysis['raw_results']['positions']:
        ax.plot(analysis['raw_results']['positions'], 
                analysis['raw_results']['hidden_means'], 'b-o')
        ax.set_xlabel('Position')
        ax.set_ylabel('Hidden State Mean')
        ax.set_title('Hidden State Mean by Position')
        ax.grid(True, alpha=0.3)
    
    # 4. Logit gap evolution
    ax = axes[1, 0]
    gap = analysis['logits']['successor_mean'] - analysis['logits']['non_successor_mean']
    ax.bar(['Gap'], [gap], color='purple', alpha=0.7)
    ax.set_ylabel('Logit Difference')
    ax.set_title(f'Successor - Non-successor Gap\n{gap:.4f}')
    ax.grid(True, alpha=0.3)
    
    # 5. Sample contribution分析
    ax = axes[1, 1]
    if analysis['contributions']:
        contrib_data = [
            analysis['contributions']['positive'],
            analysis['contributions']['negative'],
            analysis['contributions']['total']
        ]
        labels = ['Positive', 'Negative', 'Total']
        colors = ['green', 'red', 'blue']
        ax.bar(labels, contrib_data, color=colors, alpha=0.7)
        ax.set_ylabel('Contribution')
        ax.set_title('Sample Logit Contribution Breakdown')
        ax.grid(True, alpha=0.3)
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
Analysis Summary:

Samples analyzed: {analysis['logits']['num_samples']}

Logits:
  Successor: {analysis['logits']['successor_mean']:.4f} ± {analysis['logits']['successor_std']:.4f}
  Non-successor: {analysis['logits']['non_successor_mean']:.4f} ± {analysis['logits']['non_successor_std']:.4f}
  Gap: {analysis['logits']['successor_mean'] - analysis['logits']['non_successor_mean']:.4f}

Node Weights:
  Mean: {analysis['weights']['mean']:.4f}
  Range: [{analysis['weights']['min']:.4f}, {analysis['weights']['max']:.4f}]

Mechanism: {"Selection" if analysis['weights']['mean'] > 0 else "Exclusion"}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'analysis.png', dpi=150)
    plt.close()
    
    print(f"Visualization saved to {output_dir}/analysis.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/100')
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("Loading model and data...")
    model, meta, val_data = load_model_and_data(args.checkpoint, args.data_dir)
    
    print("\nAnalyzing with correct understanding...")
    analysis = analyze_logits_correctly(model, meta, val_data)
    
    if analysis['logits']['num_samples'] > 0:
        print(f"\nResults:")
        print(f"Samples analyzed: {analysis['logits']['num_samples']}")
        print(f"Successor logit mean: {analysis['logits']['successor_mean']:.4f}")
        print(f"Non-successor logit mean: {analysis['logits']['non_successor_mean']:.4f}")
        print(f"Logit gap: {analysis['logits']['successor_mean'] - analysis['logits']['non_successor_mean']:.4f}")
        print(f"\nNode weights mean: {analysis['weights']['mean']:.4f}")
        print(f"Mechanism type: {'Selection' if analysis['weights']['mean'] > 0 else 'Exclusion'}")
        
        print("\nCreating visualizations...")
        visualize_correct_analysis(analysis, args.output_dir)
    else:
        print("No valid samples found for analysis!")
    
    print(f"\nDone!")

if __name__ == "__main__":
    main()