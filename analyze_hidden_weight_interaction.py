"""
analyze_hidden_weight_interaction_v2.py
正确理解数据格式后的分析
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

def get_node_tokens(meta):
    """获取所有节点对应的token indices"""
    # 根据数据生成代码，节点0-99对应token 2-101
    node_tokens = list(range(2, 102))  # tokens for nodes 0-99
    return node_tokens

def analyze_logits_correctly(model, meta, val_data):
    """正确分析logits"""
    # 获取一批数据
    block_size = meta['block_size']
    batch_size = 32
    
    # 随机采样
    data_size = block_size + 1
    ix = torch.randint((len(val_data) - data_size) // data_size, (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.cuda(), y.cuda()
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(x)  # [batch, seq, vocab]
        
        # 获取hidden states
        tok_emb = model.transformer.wte(x)
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = model.transformer.wpe(pos)
        h = tok_emb + pos_emb
        h = model.transformer.h[0](h)
        hidden_states = model.transformer.ln_f(h)
    
    # 获取output weights
    W_output = model.lm_head.weight.T.detach()  # [n_embd, vocab]
    
    # 节点tokens
    node_tokens = get_node_tokens(meta)
    
    # 分析每个位置
    results = {
        'successor_logits': [],
        'non_successor_logits': [],
        'positions': []
    }
    
    for pos in range(5, min(20, x.shape[1])):  # 分析位置5-20
        # 获取这个位置的正确答案（下一个节点）
        correct_next = y[:, pos]  # [batch]
        
        # 只分析节点tokens（忽略PAD和换行）
        valid_mask = (correct_next >= 2) & (correct_next < 102)
        if not valid_mask.any():
            continue
            
        # 获取logits
        pos_logits = logits[valid_mask, pos, :]  # [valid_batch, vocab]
        correct_next_valid = correct_next[valid_mask]
        
        # 分离successor（正确的下一个节点）和non-successor logits
        for b in range(pos_logits.shape[0]):
            correct_token = correct_next_valid[b].item()
            
            # Successor logit
            successor_logit = pos_logits[b, correct_token].item()
            results['successor_logits'].append(successor_logit)
            
            # Non-successor logits (其他所有节点)
            for node_token in node_tokens:
                if node_token != correct_token:
                    results['non_successor_logits'].append(pos_logits[b, node_token].item())
            
        results['positions'].append(pos)
    
    # 分析weights
    node_weights = W_output[:, node_tokens]  # [n_embd, 100]
    
    analysis = {
        'logits': {
            'successor_mean': np.mean(results['successor_logits']),
            'non_successor_mean': np.mean(results['non_successor_logits']),
            'successor_std': np.std(results['successor_logits']),
            'non_successor_std': np.std(results['non_successor_logits'])
        },
        'weights': {
            'mean': node_weights.mean().item(),
            'std': node_weights.std().item(),
            'min': node_weights.min().item(),
            'max': node_weights.max().item()
        },
        'hidden_states': hidden_states,
        'W_output': W_output,
        'raw_results': results
    }
    
    return analysis

def visualize_correct_analysis(analysis, output_dir):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Logit分布对比
    ax = axes[0, 0]
    
    # 绘制直方图
    bins = 50
    ax.hist(analysis['raw_results']['successor_logits'], bins=bins, alpha=0.5, 
            label=f"Successor (n={len(analysis['raw_results']['successor_logits'])})", 
            color='green', density=True)
    ax.hist(analysis['raw_results']['non_successor_logits'], bins=bins, alpha=0.5, 
            label=f"Non-successor (n={len(analysis['raw_results']['non_successor_logits'])})", 
            color='red', density=True)
    
    ax.axvline(analysis['logits']['successor_mean'], color='green', 
               linestyle='--', linewidth=2, label=f"Successor mean: {analysis['logits']['successor_mean']:.2f}")
    ax.axvline(analysis['logits']['non_successor_mean'], color='red', 
               linestyle='--', linewidth=2, label=f"Non-successor mean: {analysis['logits']['non_successor_mean']:.2f}")
    
    ax.set_xlabel('Logit Value')
    ax.set_ylabel('Density')
    ax.set_title('Successor vs Non-successor Logit Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Weight分布
    ax = axes[0, 1]
    W = analysis['W_output'][:, 2:102].detach().cpu().numpy()  # 只看节点weights
    ax.hist(W.flatten(), bins=50, alpha=0.7, color='blue')
    ax.axvline(W.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Node Weight Distribution\nMean: {W.mean():.4f}, Std: {W.std():.4f}')
    ax.grid(True, alpha=0.3)
    
    # 3. Hidden state norm by position
    ax = axes[1, 0]
    hidden = analysis['hidden_states'].detach()
    positions = range(hidden.shape[1])
    norms = [hidden[:, pos, :].norm(dim=1).mean().item() for pos in positions]
    
    ax.plot(positions, norms, 'b-o')
    ax.set_xlabel('Position')
    ax.set_ylabel('Hidden State Norm')
    ax.set_title('Hidden State Norm Evolution')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary统计
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
Summary Statistics:

Logits:
  Successor mean: {analysis['logits']['successor_mean']:.4f}
  Non-successor mean: {analysis['logits']['non_successor_mean']:.4f}
  Gap: {analysis['logits']['successor_mean'] - analysis['logits']['non_successor_mean']:.4f}
  
  Successor std: {analysis['logits']['successor_std']:.4f}
  Non-successor std: {analysis['logits']['non_successor_std']:.4f}

Weights (for node tokens):
  Mean: {analysis['weights']['mean']:.4f}
  Std: {analysis['weights']['std']:.4f}
  Range: [{analysis['weights']['min']:.4f}, {analysis['weights']['max']:.4f}]

Interpretation:
  {"Positive weights → Selection mechanism" if analysis['weights']['mean'] > 0 else "Negative weights → Exclusion mechanism"}
  Logit gap indicates {"strong" if abs(analysis['logits']['successor_mean'] - analysis['logits']['non_successor_mean']) > 5 else "moderate"} discrimination
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'correct_analysis.png', dpi=150)
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
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("Loading model and data...")
    model, meta, val_data = load_model_and_data(args.checkpoint, args.data_dir)
    
    print("\nAnalyzing with correct understanding...")
    analysis = analyze_logits_correctly(model, meta, val_data)
    
    print(f"\nResults:")
    print(f"Successor logit mean: {analysis['logits']['successor_mean']:.4f}")
    print(f"Non-successor logit mean: {analysis['logits']['non_successor_mean']:.4f}")
    print(f"Logit gap: {analysis['logits']['successor_mean'] - analysis['logits']['non_successor_mean']:.4f}")
    print(f"\nNode weights mean: {analysis['weights']['mean']:.4f}")
    
    print("\nCreating visualizations...")
    visualize_correct_analysis(analysis, args.output_dir)
    
    print(f"\nDone! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()