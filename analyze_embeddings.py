#!/usr/bin/env python3
"""
Analyze node embeddings and their relationship to graph structure
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# Import from your existing model file
from model import GPT, GPTConfig

def analyze_embeddings(checkpoint_path, adjacency_path, num_nodes=100, save_dir='embedding_analysis'):
    """Analyze node embeddings and their relationship to graph structure"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with correct config
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Extract embeddings (token embeddings)
    # Note: Adding 2 for token offset (start/end tokens)
    embeddings = model.transformer.wte.weight.data[2:num_nodes+2].cpu().numpy()
    
    # Load adjacency matrix
    adjacency = np.load(adjacency_path)
    
    print(f"Computing pairwise similarities for {num_nodes} nodes...")
    
    # 1. Compute pairwise similarities
    similarities = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )
            similarities[i, j] = sim
    
    # 2. Separate edge and non-edge similarities
    edge_sims = []
    non_edge_sims = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Skip self-loops
                if adjacency[i, j] > 0:
                    edge_sims.append(similarities[i, j])
                else:
                    non_edge_sims.append(similarities[i, j])
    
    # 3. Compute statistics
    iteration = int(Path(checkpoint_path).stem.split('_')[0])
    
    results = {
        'iteration': iteration,
        'checkpoint': checkpoint_path,
        'edge_similarity': {
            'mean': float(np.mean(edge_sims)),
            'std': float(np.std(edge_sims)),
            'min': float(np.min(edge_sims)),
            'max': float(np.max(edge_sims))
        },
        'non_edge_similarity': {
            'mean': float(np.mean(non_edge_sims)),
            'std': float(np.std(non_edge_sims)),
            'min': float(np.min(non_edge_sims)),
            'max': float(np.max(non_edge_sims))
        },
        'similarity_gap': float(np.mean(edge_sims) - np.mean(non_edge_sims)),
        'embedding_norms': {
            'mean': float(np.mean([np.linalg.norm(e) for e in embeddings])),
            'std': float(np.std([np.linalg.norm(e) for e in embeddings])),
            'max': float(np.max([np.linalg.norm(e) for e in embeddings])),
            'min': float(np.min([np.linalg.norm(e) for e in embeddings]))
        }
    }
    
    # 4. Visualize
    plt.figure(figsize=(12, 5))
    
    # Similarity distributions
    plt.subplot(1, 2, 1)
    plt.hist(edge_sims, bins=50, alpha=0.7, label=f'Edge pairs (μ={np.mean(edge_sims):.3f})', density=True)
    plt.hist(non_edge_sims, bins=50, alpha=0.7, label=f'Non-edge pairs (μ={np.mean(non_edge_sims):.3f})', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Embedding Similarity Distribution - Iteration {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Embedding norms
    plt.subplot(1, 2, 2)
    norms = [np.linalg.norm(e) for e in embeddings]
    plt.hist(norms, bins=30)
    plt.xlabel('Embedding Norm')
    plt.ylabel('Count')
    plt.title(f'Distribution of Embedding Norms - Iteration {iteration}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'embedding_dist_{iteration}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"  Edge similarity: {results['edge_similarity']['mean']:.4f} ± {results['edge_similarity']['std']:.4f}")
    print(f"  Non-edge similarity: {results['non_edge_similarity']['mean']:.4f} ± {results['non_edge_similarity']['std']:.4f}")
    print(f"  Similarity gap: {results['similarity_gap']:.4f}")
    print(f"  Mean embedding norm: {results['embedding_norms']['mean']:.4f}")
    
    return results

def main():
    # Define checkpoints to analyze
    checkpoints = [
        'out/simple_graph_1_1_120_100/1000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/7000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/8000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/9000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/10000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/15000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/20000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/30000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/40000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/50000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/60000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/70000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/80000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/90000_ckpt_20.pt',
        'out/simple_graph_1_1_120_100/100000_ckpt_20.pt'
    ]
    
    adjacency_path = 'data/simple_graph/100/adjacency.npy'
    save_dir = 'embedding_analysis'
    
    # Check if files exist
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"Warning: {ckpt} not found, skipping...")
            checkpoints.remove(ckpt)
    
    if not os.path.exists(adjacency_path):
        print(f"Error: Adjacency matrix not found at {adjacency_path}")
        return
    
    all_results = []
    
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS")
    print("="*60 + "\n")
    
    for ckpt in checkpoints:
        try:
            results = analyze_embeddings(ckpt, adjacency_path, save_dir=save_dir)
            all_results.append(results)
            print()
        except Exception as e:
            print(f"Error processing {ckpt}: {e}")
            continue
    
    # Save all results
    results_path = os.path.join(save_dir, 'embedding_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot trends
    if len(all_results) > 0:
        print("\nGenerating trend plots...")
        
        iterations = [r['iteration'] for r in all_results]
        edge_sims = [r['edge_similarity']['mean'] for r in all_results]
        non_edge_sims = [r['non_edge_similarity']['mean'] for r in all_results]
        sim_gaps = [r['similarity_gap'] for r in all_results]
        norms = [r['embedding_norms']['mean'] for r in all_results]
        
        plt.figure(figsize=(14, 10))
        
        # Similarity evolution
        plt.subplot(2, 2, 1)
        plt.plot(iterations, edge_sims, 'o-', label='Edge pairs', linewidth=2, markersize=8)
        plt.plot(iterations, non_edge_sims, 'o-', label='Non-edge pairs', linewidth=2, markersize=8)
        plt.axvline(x=8000, color='red', linestyle='--', alpha=0.5, label='TF collapse starts')
        plt.axvline(x=40000, color='green', linestyle='--', alpha=0.5, label='Weights turn positive')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Cosine Similarity')
        plt.title('Embedding Similarity Evolution')
        plt.legend()
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Similarity gap
        plt.subplot(2, 2, 2)
        plt.plot(iterations, sim_gaps, 'o-', color='red', linewidth=2, markersize=8)
        plt.axvline(x=8000, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=40000, color='green', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Similarity Gap')
        plt.title('Edge vs Non-edge Similarity Gap')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Embedding norms
        plt.subplot(2, 2, 3)
        plt.plot(iterations, norms, 'o-', color='green', linewidth=2, markersize=8)
        plt.axvline(x=8000, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=40000, color='green', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Norm')
        plt.title('Embedding Norm Evolution')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Gap vs Norm correlation
        plt.subplot(2, 2, 4)
        plt.scatter(norms, sim_gaps, s=100, alpha=0.6)
        for i, iter_val in enumerate(iterations):
            if iter_val in [1000, 8000, 10000, 40000, 100000]:
                plt.annotate(f'{iter_val}', (norms[i], sim_gaps[i]), 
                           xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Mean Embedding Norm')
        plt.ylabel('Similarity Gap')
        plt.title('Embedding Norm vs Similarity Gap')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        evolution_path = os.path.join(save_dir, 'embedding_evolution.png')
        plt.savefig(evolution_path, dpi=150)
        plt.close()
        
        print(f"Evolution plots saved to {evolution_path}")
        
        # Print summary for key checkpoints
        print("\n" + "="*60)
        print("KEY CHECKPOINT SUMMARY")
        print("="*60)
        
        key_iters = [1000, 8000, 10000, 40000, 100000]
        for r in all_results:
            if r['iteration'] in key_iters:
                print(f"\nIteration {r['iteration']}:")
                print(f"  Similarity gap: {r['similarity_gap']:.4f}")
                print(f"  Edge sim: {r['edge_similarity']['mean']:.4f}")
                print(f"  Non-edge sim: {r['non_edge_similarity']['mean']:.4f}")
                print(f"  Norm: {r['embedding_norms']['mean']:.4f}")

if __name__ == "__main__":
    main()