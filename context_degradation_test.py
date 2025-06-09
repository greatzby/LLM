"""
Context降级测试 - 测试模型对噪声context的鲁棒性
运行方式: python context_degradation_test.py --num_nodes 100 --config 1_1_120
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import torch.nn.functional as F
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoints', type=int, nargs='+', 
                       default=[10000, 30000, 50000, 70000, 100000])
    parser.add_argument('--noise_levels', type=float, nargs='+', 
                       default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=500)
    return parser.parse_args()

def load_test_sequences(data_path, meta, max_samples=500):
    """加载测试序列"""
    stoi = meta['stoi']
    sequences = []
    
    with open(f'{data_path}/test.txt', 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            tokens = line.strip().split()
            sequence = []
            for token in tokens:
                if token in stoi:
                    sequence.append(stoi[token])
            
            if len(sequence) >= 4:
                sequences.append(sequence)
    
    return sequences

def add_noise_to_context(context, noise_level, vocab_size):
    """给context添加噪声"""
    if noise_level == 0:
        return context
    
    noisy_context = context.clone()
    batch_size, seq_len = context.shape
    
    # 计算要corrupted的token数量
    num_corrupted = int(seq_len * noise_level)
    
    for b in range(batch_size):
        # 保护前两个位置（source和target）
        if seq_len > 2:
            positions = list(range(2, seq_len))
            if positions and num_corrupted > 0:
                num_to_corrupt = min(num_corrupted, len(positions))
                corrupt_positions = random.sample(positions, num_to_corrupt)
                
                for pos in corrupt_positions:
                    # 替换为随机token（避免特殊token 0和1）
                    noisy_context[b, pos] = random.randint(2, vocab_size - 1)
    
    return noisy_context

def evaluate_with_noise(model, test_sequences, noise_level, vocab_size, device):
    """在带噪声的context下评估模型"""
    model.eval()
    
    tf_correct = 0
    tf_total = 0
    position_accuracies = {}
    
    with torch.no_grad():
        for sequence in test_sequences:
            if len(sequence) < 4:
                continue
            
            # 准备输入
            context = torch.tensor(sequence[:-1], device=device).unsqueeze(0)
            target = torch.tensor(sequence[1:], device=device).unsqueeze(0)
            
            # 添加噪声
            context = add_noise_to_context(context, noise_level, vocab_size)
            
            # 获取预测
            logits, _ = model(context)
            predictions = torch.argmax(logits, dim=-1)
            
            # 计算总体准确率
            correct_mask = (predictions == target).float()
            tf_correct += correct_mask.sum().item()
            tf_total += target.numel()
            
            # 计算每个位置的准确率
            for pos in range(min(10, correct_mask.shape[1])):
                if pos not in position_accuracies:
                    position_accuracies[pos] = []
                position_accuracies[pos].append(correct_mask[0, pos].item())
    
    tf_accuracy = tf_correct / tf_total if tf_total > 0 else 0
    
    # 计算每个位置的平均准确率
    pos_acc_means = {}
    for pos, accs in position_accuracies.items():
        pos_acc_means[f'position_{pos}'] = float(np.mean(accs))
    
    return {
        'tf_accuracy': float(tf_accuracy),
        'total_tokens': tf_total,
        'position_accuracies': pos_acc_means
    }

def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'context_degradation')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载meta信息
    dataset = 'simple_graph'
    data_path = f'data/{dataset}/{args.num_nodes}'
    meta_path = f'{data_path}/meta.pkl'
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    
    # 加载测试数据
    print("Loading test sequences...")
    test_sequences = load_test_sequences(data_path, meta, args.max_samples)
    print(f"Loaded {len(test_sequences)} test sequences")
    
    # 测试每个checkpoint
    results = {}
    
    for ckpt_iter in args.checkpoints:
        print(f"\nTesting checkpoint {ckpt_iter}")
        
        # 加载模型
        out_dir = f'out/{dataset}_{args.config}_{args.num_nodes}'
        ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_20.pt')
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping...")
            continue
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
            
            # 处理state dict
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
            model.eval()
            model.to(args.device)
            
            # 测试不同噪声水平
            noise_results = {}
            
            for noise_level in args.noise_levels:
                print(f"  Testing with noise level {noise_level}")
                
                # 多次运行取平均（因为噪声是随机的）
                runs = []
                position_accs_all = {}
                
                for run in range(args.num_runs):
                    run_result = evaluate_with_noise(model, test_sequences, 
                                                   noise_level, vocab_size, args.device)
                    runs.append(run_result['tf_accuracy'])
                    
                    # 收集位置准确率
                    for pos_key, acc in run_result['position_accuracies'].items():
                        if pos_key not in position_accs_all:
                            position_accs_all[pos_key] = []
                        position_accs_all[pos_key].append(acc)
                
                # 计算平均值
                noise_results[f'noise_{noise_level}'] = {
                    'mean_accuracy': float(np.mean(runs)),
                    'std_accuracy': float(np.std(runs)),
                    'min_accuracy': float(np.min(runs)),
                    'max_accuracy': float(np.max(runs)),
                    'position_accuracies': {
                        pos: float(np.mean(accs)) for pos, accs in position_accs_all.items()
                    }
                }
            
            results[ckpt_iter] = noise_results
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing checkpoint {ckpt_iter}: {e}")
            continue
    
    # 保存结果
    with open(os.path.join(output_dir, 'context_degradation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 创建可视化
    create_degradation_plots(results, args.noise_levels, output_dir)
    
    print(f"\nContext degradation test complete! Results saved to {output_dir}/")

def create_degradation_plots(results, noise_levels, output_dir):
    """创建降级测试的可视化"""
    if not results:
        print("No results to plot")
        return
    
    # 主要降级曲线
    plt.figure(figsize=(10, 6))
    
    for ckpt in sorted(results.keys()):
        accuracies = []
        errors = []
        
        for noise in noise_levels:
            key = f'noise_{noise}'
            if key in results[ckpt]:
                accuracies.append(results[ckpt][key]['mean_accuracy'])
                errors.append(results[ckpt][key]['std_accuracy'])
            else:
                accuracies.append(0)
                errors.append(0)
        
        plt.errorbar(noise_levels, accuracies, yerr=errors, 
                    marker='o', label=f'Iteration {ckpt}', capsize=5)
    
    plt.xlabel('Noise Level')
    plt.ylabel('Teacher Forcing Accuracy')
    plt.title('Model Robustness to Context Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, max(noise_levels) + 0.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'context_degradation_curves.png'), dpi=150)
    plt.close()
    
    # 相对降级率
    plt.figure(figsize=(10, 6))
    
    for ckpt in sorted(results.keys()):
        baseline = results[ckpt].get('noise_0.0', {}).get('mean_accuracy', 1.0)
        if baseline > 0:
            relative_accs = []
            
            for noise in noise_levels:
                key = f'noise_{noise}'
                if key in results[ckpt]:
                    acc = results[ckpt][key]['mean_accuracy']
                    relative_accs.append(acc / baseline)
                else:
                    relative_accs.append(1.0)
            
            plt.plot(noise_levels, relative_accs, marker='o', label=f'Iteration {ckpt}')
    
    plt.xlabel('Noise Level')
    plt.ylabel('Relative Accuracy (vs No Noise)')
    plt.title('Relative Performance Degradation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, max(noise_levels) + 0.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_degradation_curves.png'), dpi=150)
    plt.close()
    
    # 位置特定的降级（选择一个中间的噪声水平）
    mid_noise = 0.2
    positions = list(range(10))
    
    plt.figure(figsize=(12, 6))
    
    for ckpt in sorted(results.keys()):
        if f'noise_{mid_noise}' in results[ckpt]:
            pos_accs = results[ckpt][f'noise_{mid_noise}']['position_accuracies']
            
            accs = []
            for pos in positions:
                key = f'position_{pos}'
                if key in pos_accs:
                    accs.append(pos_accs[key])
                else:
                    accs.append(0)
            
            if accs:
                plt.plot(positions[:len(accs)], accs, marker='o', label=f'Iteration {ckpt}')
    
    plt.xlabel('Token Position')
    plt.ylabel('Accuracy')
    plt.title(f'Position-wise Accuracy with {mid_noise} Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_degradation.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()