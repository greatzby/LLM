#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import GPTConfig, GPT  # Ensure that your model.py file defines GPTConfig and GPT, and that they include transformer.wte, lm_head, transformer.h, attn, etc.

def plot_and_save_heatmap(matrix, title="Heatmap", save_dir=".", state_tag="state"):
    """
    Plot a heatmap from the given matrix using matplotlib and save it as an image file.
    
    Parameters:
      - matrix: a numpy array to visualize.
      - title: the title of the plot; this will also be used in the filename.
      - save_dir: the directory where the image will be saved.
      - state_tag: a label representing the current state (e.g., checkpoint name).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    filename = f"{state_tag}_{title.replace(' ', '_')}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    print(f"Saved '{title}' heatmap to: {full_path}")
    plt.close()

def load_model(ckpt_path, device):
    """
    Load a GPT model from the specified checkpoint path.
    Assumes the checkpoint contains "model_args" and "model" fields.
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
    
    Assumptions:
      - Token embedding matrix: W_t = model.transformer.wte.weight, with shape (vocab_size, d_model)
      - Output projection matrix: W_o = model.lm_head.weight, with shape (vocab_size, d_model)
      - FFN module: using model.transformer.h[0].mlp which maps an input (1, d_model) to (1, d_model)
      - The Value matrix W^V is taken from model.transformer.h[0].attn.c_attn.weight,
        which has shape (3*d_model, d_model). We extract the last d_model rows.
    
    For a node i (corresponding to the token at index i in the vocabulary), let:
         x = e_iᵀ · W_t = W_t[i]   (with shape (d_model,))
    
    Then the formulas are:
         WM'(i,:) = FFN(x) · W_o + x · W_o
         WV'(i,:) = (x · W^V) · W_o + FFN(x · W^V) · W_o
    
    To facilitate visualization, this code only extracts the first num_nodes elements from the
    output vectors to form a num_nodes x num_nodes square matrix.
    """
    # Extract token embedding and output projection matrices
    W_t = model.transformer.wte.weight    # shape: (vocab_size, d_model)
    W_o = model.lm_head.weight              # shape: (vocab_size, d_model)
    ff_module = model.transformer.h[0].mlp   # FFN module

    # Retrieve the value matrix from the attention module
    full_attn = model.transformer.h[0].attn.c_attn.weight  # shape: (3*d_model, d_model)
    d_model = full_attn.shape[1]
    W_V = full_attn[2 * d_model: 3 * d_model, :]   # Extract Value portion, shape: (d_model, d_model)

    WM_list = []
    WV_list = []
    
    for i in range(num_nodes):
        # Use the token embedding corresponding to node i (i.e., e_iᵀ * W_t)
        x = W_t[i]  # shape: (d_model,)
        
        # Compute FFN(x). ff_module expects a 2D tensor, so add a batch dimension.
        x_tensor = x.unsqueeze(0)  # shape: (1, d_model)
        FFN_x = ff_module(x_tensor)  # shape: (1, d_model)
        
        # Compute x·W_o and FFN(x)·W_o using W_o transposed.
        # Both terms yield a vector of shape (vocab_size,)
        term1 = torch.matmul(FFN_x, W_o.t()).squeeze(0)   # shape: (vocab_size,)
        term2 = torch.matmul(x, W_o.t())                    # shape: (vocab_size,)
        wm_i = term1 + term2                              # This forms the full WM'(i,:) vector
        
        # Take the first num_nodes entries for visualization
        wm_i_slice = wm_i[:num_nodes]
        WM_list.append(wm_i_slice.unsqueeze(0))
        
        # For WV': first compute x·W_V
        x_wv = torch.matmul(x, W_V)                        # shape: (d_model,)
        x_wv_tensor = x_wv.unsqueeze(0)                    # shape: (1, d_model)
        FFN_x_wv = ff_module(x_wv_tensor)                  # shape: (1, d_model)
        term3 = torch.matmul(x_wv, W_o.t())                # shape: (vocab_size,)
        term4 = torch.matmul(FFN_x_wv, W_o.t()).squeeze(0)   # shape: (vocab_size,)
        wv_i = term3 + term4                              # Full WV'(i,:) vector
        wv_i_slice = wv_i[:num_nodes]
        WV_list.append(wv_i_slice.unsqueeze(0))

    WM_mat = torch.cat(WM_list, dim=0)  # shape: (num_nodes, num_nodes)
    WV_mat = torch.cat(WV_list, dim=0)  # shape: (num_nodes, num_nodes)
    return WM_mat.cpu().detach().numpy(), WV_mat.cpu().detach().numpy()

def main():
    parser = argparse.ArgumentParser(
        description="Compute WM' and WV' matrices based on the paper formulas and save their heatmaps."
    )
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument('--save_dir', type=str, default=".", help="Directory to save the heatmap images")
    parser.add_argument('--num_nodes', type=int, required=True, help="Number of nodes (defines the square matrix dimension)")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = args.device
    model = load_model(args.ckpt, device)
    
    # Compute the WM' and WV' matrices
    WM_mat, WV_mat = compute_WM_WV(model, args.num_nodes)
    print("WM' matrix:\n", WM_mat)
    print("WV' matrix:\n", WV_mat)
    
    state_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    # Save heatmaps for the matrices
    plot_and_save_heatmap(WM_mat, title="WM_prime", save_dir=args.save_dir, state_tag=state_tag)
    plot_and_save_heatmap(WV_mat, title="WV_prime", save_dir=args.save_dir, state_tag=state_tag)

if __name__ == '__main__':
    main()