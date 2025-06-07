#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

from model import GPTConfig, GPT

def plot_heatmap_with_adjacency(matrix, adjacency_matrix, title="WM_prime", save_dir=".", state_tag="state", show_first_n=20):
    """
    Plot a heatmap with red boxes marking the adjacency matrix 1's.
    """
    # Take only first show_first_n rows and columns
    matrix_subset = matrix[:show_first_n, :show_first_n]
    adjacency_subset = adjacency_matrix[:show_first_n, :show_first_n]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix_subset, cmap='viridis', aspect='auto')
    plt.colorbar()
    
    # Add red boxes for adjacency matrix 1's
    for i in range(show_first_n):
        for j in range(show_first_n):
            if adjacency_subset[i, j] == 1:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)
    
    plt.title(f"{title} (First {show_first_n}x{show_first_n})")
    plt.xlabel("Target Node (j)")
    plt.ylabel("Source Node (i)")
    plt.tight_layout()
    
    filename = f"{state_tag}_{title}_with_adjacency.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=150)
    print(f"Saved '{title}' heatmap with adjacency to: {full_path}")
    plt.close()

def plot_simple_heatmap(matrix, title="Heatmap", save_dir=".", state_tag="state", show_first_n=None):
    """
    Plot a simple heatmap from the given matrix.
    """
    if show_first_n is not None:
        matrix = matrix[:show_first_n, :show_first_n]
        title = f"{title} (First {show_first_n}x{show_first_n})"
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Target Node (j)")
    plt.ylabel("Source Node (i)")
    plt.tight_layout()
    
    filename = f"{state_tag}_{title.replace(' ', '_')}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=150)
    print(f"Saved '{title}' heatmap to: {full_path}")
    plt.close()

def verify_adjacency_matrix(adjacency_matrix):
    """验证邻接矩阵的属性"""
    # 检查是否对称（无向图）
    is_symmetric = np.allclose(adjacency_matrix, adjacency_matrix.T)
    print(f"\nAdjacency matrix is symmetric: {is_symmetric}")
    
    # 统计边数
    total_edges = np.sum(adjacency_matrix)
    
    if is_symmetric:
        num_edges = np.sum(adjacency_matrix[np.triu_indices_from(adjacency_matrix, k=1)])
        print(f"Graph type: Undirected")
        print(f"Number of edges: {num_edges}")
    else:
        print(f"Graph type: Directed")
        print(f"Number of directed edges: {total_edges}")
        # 检查有多少双向边
        bidirectional = 0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(i+1, adjacency_matrix.shape[1]):
                if adjacency_matrix[i,j] == 1 and adjacency_matrix[j,i] == 1:
                    bidirectional += 1
        print(f"Number of bidirectional edges: {bidirectional}")
    
    # 计算入度和出度
    out_degrees = np.sum(adjacency_matrix, axis=1)
    in_degrees = np.sum(adjacency_matrix, axis=0)
    
    print(f"Average out-degree: {np.mean(out_degrees):.2f}")
    print(f"Average in-degree: {np.mean(in_degrees):.2f}")
    print(f"Max out-degree: {np.max(out_degrees)}")
    print(f"Max in-degree: {np.max(in_degrees)}")
    
    return is_symmetric

def calculate_weight_gap(WM_matrix, adjacency_matrix, directed=True):
    """
    Calculate the average weight gap between edges and non-edges.
    For directed graphs: considers all (i,j) pairs where i≠j
    For undirected graphs: only considers i<j pairs
    """
    num_nodes = adjacency_matrix.shape[0]
    edge_weights = []
    non_edge_weights = []
    
    if directed:
        # For directed graphs, consider all pairs except self-loops
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    if adjacency_matrix[i, j] == 1:
                        edge_weights.append(WM_matrix[i, j])
                    else:
                        non_edge_weights.append(WM_matrix[i, j])
    else:
        # For undirected graphs, only consider upper triangular
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adjacency_matrix[i, j] == 1:
                    edge_weights.append(WM_matrix[i, j])
                else:
                    non_edge_weights.append(WM_matrix[i, j])
    
    avg_edge_weight = np.mean(edge_weights) if edge_weights else 0
    avg_non_edge_weight = np.mean(non_edge_weights) if non_edge_weights else 0
    weight_gap = avg_edge_weight - avg_non_edge_weight
    
    print(f"\nWeight gap analysis:")
    print(f"Number of edges: {len(edge_weights)}")
    print(f"Number of non-edges: {len(non_edge_weights)}")
    print(f"Average edge weight: {avg_edge_weight:.6f}")
    print(f"Average non-edge weight: {avg_non_edge_weight:.6f}")
    print(f"Weight gap: {weight_gap:.6f}")
    print(f"Edge/Non-edge ratio: {avg_edge_weight / (avg_non_edge_weight + 1e-10):.2f}")
    
    # Additional statistics
    if edge_weights:
        print(f"\nEdge weight statistics:")
        print(f"  Min: {np.min(edge_weights):.6f}")
        print(f"  Max: {np.max(edge_weights):.6f}")
        print(f"  Std: {np.std(edge_weights):.6f}")
    
    if non_edge_weights:
        print(f"\nNon-edge weight statistics:")
        print(f"  Min: {np.min(non_edge_weights):.6f}")
        print(f"  Max: {np.max(non_edge_weights):.6f}")
        print(f"  Std: {np.std(non_edge_weights):.6f}")
        print(f"  Percentage positive: {100 * np.sum(np.array(non_edge_weights) > 0) / len(non_edge_weights):.2f}%")
    
    return weight_gap, avg_edge_weight, avg_non_edge_weight

def analyze_weight_distribution(WM_matrix, adjacency_matrix, save_dir, state_tag):
    """
    Analyze and plot the distribution of edge and non-edge weights.
    """
    edge_weights = []
    non_edge_weights = []
    
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                if adjacency_matrix[i, j] == 1:
                    edge_weights.append(WM_matrix[i, j])
                else:
                    non_edge_weights.append(WM_matrix[i, j])
    
    # Plot weight distributions
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Histograms
    plt.subplot(1, 2, 1)
    bins = np.linspace(min(min(edge_weights), min(non_edge_weights)), 
                      max(max(edge_weights), max(non_edge_weights)), 50)
    plt.hist(edge_weights, bins=bins, alpha=0.6, label=f'Edges (n={len(edge_weights)})', color='green')
    plt.hist(non_edge_weights, bins=bins, alpha=0.6, label=f'Non-edges (n={len(non_edge_weights)})', color='red')
    plt.axvline(np.mean(edge_weights), color='darkgreen', linestyle='--', label=f'Edge mean: {np.mean(edge_weights):.4f}')
    plt.axvline(np.mean(non_edge_weights), color='darkred', linestyle='--', label=f'Non-edge mean: {np.mean(non_edge_weights):.4f}')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.title('Distribution of Edge vs Non-edge Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plots
    plt.subplot(1, 2, 2)
    data_to_plot = [edge_weights, non_edge_weights]
    bp = plt.boxplot(data_to_plot, labels=['Edges', 'Non-edges'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    plt.ylabel('Weight Value')
    plt.title('Weight Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{state_tag}_weight_distributions.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=150)
    print(f"Saved weight distribution plot to: {full_path}")
    plt.close()

def analyze_direct_output_weights(model, num_nodes, adjacency_matrix, token_offset=2):
    """
    Directly analyze the output layer weights without FFN transformation.
    This gives us the 'raw' weights before any non-linear transformations.
    """
    print("\n" + "="*60)
    print("ANALYZING DIRECT OUTPUT WEIGHTS (without FFN)")
    print("="*60)
    
    W_o = model.lm_head.weight  # (vocab_size, d_model)
    W_t = model.transformer.wte.weight  # (vocab_size, d_model)
    
    # For each node pair (i,j), compute the direct weight: W_t[i]^T @ W_o[j]
    num_nodes_actual = min(num_nodes, W_t.shape[0] - token_offset)
    direct_weights = np.zeros((num_nodes_actual, num_nodes_actual))
    
    with torch.no_grad():
        for i in range(num_nodes_actual):
            for j in range(num_nodes_actual):
                # Direct weight from node i to node j
                token_i = i + token_offset
                token_j = j + token_offset
                weight = torch.dot(W_t[token_i], W_o[token_j]).item()
                direct_weights[i, j] = weight
    
    # Calculate weight gap for direct weights
    edge_weights = []
    non_edge_weights = []
    
    for i in range(num_nodes_actual):
        for j in range(num_nodes_actual):
            if i != j and i < adjacency_matrix.shape[0] and j < adjacency_matrix.shape[1]:
                if adjacency_matrix[i, j] == 1:
                    edge_weights.append(direct_weights[i, j])
                else:
                    non_edge_weights.append(direct_weights[i, j])
    
    avg_edge = np.mean(edge_weights) if edge_weights else 0
    avg_non_edge = np.mean(non_edge_weights) if non_edge_weights else 0
    
    print(f"\nDirect weight statistics (without FFN):")
    print(f"Average edge weight: {avg_edge:.6f}")
    print(f"Average non-edge weight: {avg_non_edge:.6f}")
    print(f"Weight gap: {avg_edge - avg_non_edge:.6f}")
    print(f"Percentage of positive non-edge weights: {100 * np.sum(np.array(non_edge_weights) > 0) / len(non_edge_weights):.2f}%")
    
    return direct_weights

def load_adjacency_matrix(adj_path, num_nodes):
    """
    Load adjacency matrix from file. Supports .npy and .txt formats.
    """
    if adj_path.endswith('.npy'):
        adjacency = np.load(adj_path)
    elif adj_path.endswith('.txt'):
        adjacency = np.loadtxt(adj_path)
    else:
        raise ValueError(f"Unsupported adjacency matrix format: {adj_path}")
    
    # Ensure it's the right size
    if adjacency.shape[0] != num_nodes or adjacency.shape[1] != num_nodes:
        print(f"Warning: Adjacency matrix shape {adjacency.shape} doesn't match num_nodes {num_nodes}")
        # Resize if necessary
        min_size = min(adjacency.shape[0], num_nodes)
        adjacency = adjacency[:min_size, :min_size]
        if min_size < num_nodes:
            # Pad with zeros if needed
            padded = np.zeros((num_nodes, num_nodes))
            padded[:min_size, :min_size] = adjacency
            adjacency = padded
    
    return adjacency

def load_model(ckpt_path, device):
    """
    Load a GPT model from the specified checkpoint path.
    """
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config_args = checkpoint["model_args"]
    config = GPTConfig(**config_args)
    print("Model configuration:", config)
    
    model = GPT(config)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully and set to eval mode.")
    return model

def compute_WM_WV(model, num_nodes, token_offset=2):
    """
    Compute the WM' and WV' matrices based on the formulas from the paper.
    
    IMPORTANT: Node i corresponds to token (i + token_offset) due to special tokens:
    - Token 0: [PAD]
    - Token 1: \n
    - Token 2: Node 0
    - Token 3: Node 1
    - etc.
    """
    # Extract token embedding and output projection matrices
    W_t = model.transformer.wte.weight    # shape: (vocab_size, d_model)
    W_o = model.lm_head.weight            # shape: (vocab_size, d_model)
    ff_module = model.transformer.h[0].mlp   # FFN module

    # Retrieve the value matrix from the attention module
    full_attn = model.transformer.h[0].attn.c_attn.weight  # shape: (3*d_model, d_model)
    d_model = full_attn.shape[1]
    W_V = full_attn[2 * d_model: 3 * d_model, :]   # Extract Value portion

    WM_list = []
    WV_list = []
    
    with torch.no_grad():  # Ensure no gradient computation
        for i in range(num_nodes):
            # CRITICAL FIX: Use the correct token index for node i
            x = W_t[i + token_offset]  # Node i corresponds to token (i + token_offset)
            
            # Compute FFN(x)
            x_tensor = x.unsqueeze(0)  # shape: (1, d_model)
            FFN_x = ff_module(x_tensor)  # shape: (1, d_model)
            
            # Compute WM'(i,:)
            term1 = torch.matmul(FFN_x, W_o.t()).squeeze(0)   # shape: (vocab_size,)
            term2 = torch.matmul(x, W_o.t())                    # shape: (vocab_size,)
            wm_i = term1 + term2
            
            # Extract weights corresponding to nodes (not special tokens)
            # We want weights for tokens [token_offset, token_offset+num_nodes)
            wm_i_nodes = wm_i[token_offset:token_offset+num_nodes]
            WM_list.append(wm_i_nodes.unsqueeze(0))
            
            # Compute WV'(i,:)
            x_wv = torch.matmul(x, W_V)                        # shape: (d_model,)
            x_wv_tensor = x_wv.unsqueeze(0)                    # shape: (1, d_model)
            FFN_x_wv = ff_module(x_wv_tensor)                  # shape: (1, d_model)
            term3 = torch.matmul(x_wv, W_o.t())                # shape: (vocab_size,)
            term4 = torch.matmul(FFN_x_wv, W_o.t()).squeeze(0)   # shape: (vocab_size,)
            wv_i = term3 + term4
            
            # Extract weights corresponding to nodes
            wv_i_nodes = wv_i[token_offset:token_offset+num_nodes]
            WV_list.append(wv_i_nodes.unsqueeze(0))

    WM_mat = torch.cat(WM_list, dim=0)  # shape: (num_nodes, num_nodes)
    WV_mat = torch.cat(WV_list, dim=0)  # shape: (num_nodes, num_nodes)
    
    print(f"WM' matrix shape: {WM_mat.shape}")
    print(f"WV' matrix shape: {WV_mat.shape}")
    
    return WM_mat.cpu().detach().numpy(), WV_mat.cpu().detach().numpy()

def main():
    parser = argparse.ArgumentParser(
        description="Compute WM' and WV' matrices and analyze them with adjacency matrix."
    )
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use")
    parser.add_argument('--save_dir', type=str, default=".", help="Directory to save the heatmap images")
    parser.add_argument('--num_nodes', type=int, required=True, help="Number of nodes")
    parser.add_argument('--adjacency', type=str, help="Path to adjacency matrix file (.npy or .txt)")
    parser.add_argument('--show_first_n', type=int, default=20, help="Show first N rows/cols in visualization")
    parser.add_argument('--token_offset', type=int, default=2, help="Offset for node tokens (default: 2)")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = args.device
    model = load_model(args.ckpt, device)
    
    # Compute the WM' and WV' matrices
    print("\nComputing WM' and WV' matrices...")
    WM_mat, WV_mat = compute_WM_WV(model, args.num_nodes, args.token_offset)
    
    state_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    
    # If adjacency matrix is provided, plot with red boxes and calculate weight gap
    if args.adjacency:
        print("\nLoading adjacency matrix...")
        adjacency_matrix = load_adjacency_matrix(args.adjacency, args.num_nodes)
        
        # Verify if it's directed or undirected
        is_symmetric = verify_adjacency_matrix(adjacency_matrix)
        
        # Plot WM' with adjacency matrix overlay
        plot_heatmap_with_adjacency(WM_mat, adjacency_matrix, title="WM_prime", 
                                  save_dir=args.save_dir, state_tag=state_tag, 
                                  show_first_n=args.show_first_n)
        
        # Calculate and display weight gap for WM'
        print("\nCalculating weight gap for WM':")
        weight_gap, avg_edge, avg_non_edge = calculate_weight_gap(WM_mat, adjacency_matrix, 
                                                                 directed=not is_symmetric)
        
        # Analyze weight distribution
        analyze_weight_distribution(WM_mat, adjacency_matrix, args.save_dir, state_tag)
        
        # Analyze direct output weights (without FFN)
        direct_weights = analyze_direct_output_weights(model, args.num_nodes, 
                                                     adjacency_matrix, args.token_offset)
        
        # Save weight gap information
        gap_info = {
            "checkpoint": args.ckpt,
            "num_nodes": args.num_nodes,
            "weight_gap": float(weight_gap),
            "avg_edge_weight": float(avg_edge),
            "avg_non_edge_weight": float(avg_non_edge),
            "token_offset": args.token_offset,
            "graph_type": "directed" if not is_symmetric else "undirected"
        }
        gap_file = os.path.join(args.save_dir, f"{state_tag}_weight_gap.json")
        with open(gap_file, 'w') as f:
            json.dump(gap_info, f, indent=2)
        print(f"\nWeight gap info saved to: {gap_file}")
    else:
        # Plot simple heatmaps without adjacency overlay
        print("\nNo adjacency matrix provided, plotting simple heatmaps...")
        plot_simple_heatmap(WM_mat, title="WM_prime", save_dir=args.save_dir, 
                          state_tag=state_tag, show_first_n=args.show_first_n)
    
    # Always plot WV' as simple heatmap
    plot_simple_heatmap(WV_mat, title="WV_prime", save_dir=args.save_dir, 
                      state_tag=state_tag, show_first_n=args.show_first_n)

if __name__ == '__main__':
    main()