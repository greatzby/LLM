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
    plt.xlabel("Columns")
    plt.ylabel("Rows")
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
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    
    filename = f"{state_tag}_{title.replace(' ', '_')}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=150)
    print(f"Saved '{title}' heatmap to: {full_path}")
    plt.close()

def calculate_weight_gap(WM_matrix, adjacency_matrix):
    """
    Calculate the average weight gap between edges and non-edges.
    Following the paper: only consider i < j positions.
    """
    num_nodes = adjacency_matrix.shape[0]
    edge_weights = []
    non_edge_weights = []
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Only upper triangular (i < j)
            if adjacency_matrix[i, j] == 1:
                edge_weights.append(WM_matrix[i, j])
            else:
                non_edge_weights.append(WM_matrix[i, j])
    
    avg_edge_weight = np.mean(edge_weights) if edge_weights else 0
    avg_non_edge_weight = np.mean(non_edge_weights) if non_edge_weights else 0
    weight_gap = avg_edge_weight - avg_non_edge_weight
    
    print(f"Number of edges: {len(edge_weights)}")
    print(f"Number of non-edges: {len(non_edge_weights)}")
    print(f"Average edge weight: {avg_edge_weight:.4f}")
    print(f"Average non-edge weight: {avg_non_edge_weight:.4f}")
    print(f"Weight gap: {weight_gap:.4f}")
    
    return weight_gap, avg_edge_weight, avg_non_edge_weight

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

def compute_WM_WV(model, num_nodes):
    """
    Compute the WM' and WV' matrices based on the formulas from the paper.
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
            # Use the token embedding corresponding to node i
            x = W_t[i]  # shape: (d_model,)
            
            # Compute FFN(x)
            x_tensor = x.unsqueeze(0)  # shape: (1, d_model)
            FFN_x = ff_module(x_tensor)  # shape: (1, d_model)
            
            # Compute WM'(i,:)
            term1 = torch.matmul(FFN_x, W_o.t()).squeeze(0)   # shape: (vocab_size,)
            term2 = torch.matmul(x, W_o.t())                    # shape: (vocab_size,)
            wm_i = term1 + term2
            wm_i_slice = wm_i[:num_nodes]
            WM_list.append(wm_i_slice.unsqueeze(0))
            
            # Compute WV'(i,:)
            x_wv = torch.matmul(x, W_V)                        # shape: (d_model,)
            x_wv_tensor = x_wv.unsqueeze(0)                    # shape: (1, d_model)
            FFN_x_wv = ff_module(x_wv_tensor)                  # shape: (1, d_model)
            term3 = torch.matmul(x_wv, W_o.t())                # shape: (vocab_size,)
            term4 = torch.matmul(FFN_x_wv, W_o.t()).squeeze(0)   # shape: (vocab_size,)
            wv_i = term3 + term4
            wv_i_slice = wv_i[:num_nodes]
            WV_list.append(wv_i_slice.unsqueeze(0))

    WM_mat = torch.cat(WM_list, dim=0)  # shape: (num_nodes, num_nodes)
    WV_mat = torch.cat(WV_list, dim=0)  # shape: (num_nodes, num_nodes)
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
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = args.device
    model = load_model(args.ckpt, device)
    
    # Compute the WM' and WV' matrices
    print("\nComputing WM' and WV' matrices...")
    WM_mat, WV_mat = compute_WM_WV(model, args.num_nodes)
    
    state_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    
    # If adjacency matrix is provided, plot with red boxes and calculate weight gap
    if args.adjacency:
        print("\nLoading adjacency matrix...")
        adjacency_matrix = load_adjacency_matrix(args.adjacency, args.num_nodes)
        
        # Plot WM' with adjacency matrix overlay
        plot_heatmap_with_adjacency(WM_mat, adjacency_matrix, title="WM_prime", 
                                  save_dir=args.save_dir, state_tag=state_tag, 
                                  show_first_n=args.show_first_n)
        
        # Calculate and display weight gap for WM'
        print("\nCalculating weight gap for WM':")
        weight_gap, avg_edge, avg_non_edge = calculate_weight_gap(WM_mat, adjacency_matrix)
        
        # Save weight gap information
        gap_info = {
            "checkpoint": args.ckpt,
            "num_nodes": args.num_nodes,
            "weight_gap": float(weight_gap),
            "avg_edge_weight": float(avg_edge),
            "avg_non_edge_weight": float(avg_non_edge)
        }
        gap_file = os.path.join(args.save_dir, f"{state_tag}_weight_gap.json")
        with open(gap_file, 'w') as f:
            json.dump(gap_info, f, indent=2)
        print(f"Weight gap info saved to: {gap_file}")
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